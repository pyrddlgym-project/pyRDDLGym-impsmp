"""Abstract interface for deterministic environment models"""
import jax
import jax.numpy as jnp
import jax.random

from pyRDDLGym_impsmp.models.base import BaseDeterministicModel

VALID_INITIALIZATION_STRATEGIES = (
    'constant',
    'normal',
    'uniform'
)


def generate_initial_states(key, config, batch_shape, state_dim):
    """Initializes a batch of policy rollouts"""
    if config['type'] == 'constant':
        val = config['params']['value']
        init_states = jnp.ones(shape=(*batch_shape, state_dim)) * val
    elif config['type'] == 'normal':
        mean = config['params']['mean']
        scale = config['params']['scale']
        init_states = mean + jax.random.normal(key, shape=(*batch_shape, state_dim)) * scale
    elif config['type'] == 'uniform':
        min = config['params']['min']
        max = config['params']['max']
        init_states = jax.random.uniform(key, shape=(*batch_shape, state_dim), minval=min, maxval=max)
    return init_states

def compute_sum_of_half_spaces_reward(next_state, W, B):
    return jnp.sum(jnp.sign(jnp.diag(jnp.tensordot(W, (next_state - B), axes=(1, 1)))))

class JAXSumOfHalfSpacesModel(BaseDeterministicModel):
    """An implementation of the Sum-of-Half-Spaces environment in Pure JAX, without
    using RDDL.

    In the Sum-of-Half-Spaces environment, the reward function is given by the
    expression

            R(x) = sum_{i=1}^N sgn( sum_{d=1}^|S| W_{id} * (x_d - B_{id} )

    where N is the number of summands parameter, W_{id}, B_{id} are fixed constants
    that define the problem, and x = (x_1, ..., x_|S|) is the state vector.

    The actions in the environment are translations of the state vector.
    """
    def __init__(self,
                 key,
                 state_dim,
                 action_dim,
                 horizon,
                 n_summands,
                 is_relaxed,
                 initial_state_config,
                 reward_shift,
                 relaxation_kwargs):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_summands = n_summands

        assert initial_state_config['type'] in VALID_INITIALIZATION_STRATEGIES
        self.initial_state_config = initial_state_config
        self.reward_shift_val = reward_shift

        key, *subkeys = jax.random.split(key, num=3)
        self.W = jax.random.normal(subkeys[0], shape=(n_summands, state_dim))
        self.B = jax.random.normal(subkeys[1], shape=(n_summands, state_dim)) * jnp.sqrt(5)

    def generate_initial_state_batched(self, key, batch_shape):
        """Generates the initial model states over a batch of generic shape

        Args:
            key: jax.random.PRNGKey
                Random generator state key
            batch_shape: Tuple of Int
                The desired shape
        """
        key, subkey = jax.random.split(key)
        init_states = generate_initial_states(subkey, self.initial_state_config, batch_shape, self.state_dim)
        return key, init_states

    def rollout_parametrized_policy(self, key, init_state, policy, theta, shift_reward=False):
        """Rolls out the policy with parameters theta."""
        def _step(carry, _):
            key, state = carry
            key, subkey = jax.random.split(key)
            action = policy.sample(subkey, theta, state)
            next_state = state + action
            reward = compute_sum_of_half_spaces_reward(next_state, self.W, self.B)
            return (key, next_state), (state, action, reward)

        (key, final_state), (states, actions, rewards) = jax.lax.scan(_step, init=(key, init_state), xs=None, length=self.horizon)
        return key, states, actions, rewards

    def rollout_parametrized_policy_batched(self, key, batch_init_states, policy, theta, shift_reward=False):
        """Performs a batch of roll outs of the policy with parameters theta"""
        B = batch_init_states.shape[0]
        key, *subkeys = jax.random.split(key, num=B+1)
        subkeys = jnp.asarray(subkeys)
        _, batch_states, batch_actions, batch_rewards = jax.vmap(
            self.rollout_parametrized_policy, (0, 0, None, None, None), (0, 0, 0, 0))(subkeys, batch_init_states, policy, theta, shift_reward)
        return key, batch_states, batch_actions, batch_rewards

    def evaluate_action_trajectory(self, key, init_state, actions, shift_reward=False):
        """Evaluates an action trajectory."""
        def _step(carry, action):
            key, state = carry
            key, subkey = jax.random.split(key)
            next_state = state + action
            reward = compute_sum_of_half_spaces_reward(next_state, self.W, self.B)
            return (key, next_state), (state, action, reward)

        (key, final_state), (states, actions, rewards) = jax.lax.scan(_step, init=(key, init_state), xs=actions)
        return key, states, actions, rewards

    def evaluate_action_trajectory_batched(self, key, batch_init_states, batch_actions, shift_reward=False):
        """Evaluates a batch of action trajectories."""
        assert batch_init_states.shape[0] == batch_actions.shape[0]
        B = batch_init_states.shape[0]
        key, *subkeys = jax.random.split(key, num=B+1)
        subkeys = jnp.asarray(subkeys)
        _, batch_states, batch_actions, batch_rewards = jax.vmap(
            self.evaluate_action_trajectory, (0, 0, 0, None), (0, 0, 0, 0))(subkeys, batch_init_states, batch_actions, shift_reward)
        return key, batch_states, batch_actions, batch_rewards

    def print_report(self, it):
        print(f'\tModel :: Sum-of-Half-Spaces (Pure JAX) ::'
              f' Dim={self.state_dim},',
              f' Summands={self.n_summands},',
              f' Horizon={self.horizon}')
