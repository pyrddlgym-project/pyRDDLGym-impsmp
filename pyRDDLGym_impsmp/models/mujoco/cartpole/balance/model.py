"""Interface to the MuJoCo CartPole (Balance) model"""
from pyRDDLGym_impsmp.models.base import BaseDeterministicModel

class MuJoCoCartpoleBalanceModel(BaseDeterministicModel):
    def __init__(self):
        pass

    def batch_generate_initial_state(self, key, batch_shape):
        """Generates the initial model states over a batch of generic shape

        Args:
            key: jax.random.PRNGKey
                Random generator state key
            batch_shape: Tuple of Int
                The desired shape
        """
        pass

    def rollout_parametrized_policy(self, key, init_states, theta, shift_reward=False):
        """Rolls out the policy with parameters theta."""
        pass

    def rollout_parametrized_policy_batched(self, key, batch_init_states, theta, shift_reward=False):
        """Performs a batch of roll outs of the policy with parameters theta"""
        pass

    def evaluate_action_trajectory(self, key, init_state, actions, shift_reward=False):
        """Evaluates an action trajectory."""
        pass

    def evaluate_action_trajectory_batched(self, key, batch_init_states, batch_actions, shift_reward=False):
        """Evaluates a batch of action trajectories."""
        pass
