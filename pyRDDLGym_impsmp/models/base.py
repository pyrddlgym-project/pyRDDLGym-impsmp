"""Abstract interface for deterministic environment models"""
import abc

class BaseDeterministicModel(abc.ABC):
    @abc.abstractmethod
    def generate_initial_state_batched(self, key, batch_shape):
        """Generates the initial model states over a batch of generic shape

        Args:
            key: jax.random.PRNGKey
                Random generator state key
            batch_shape: Tuple of Int
                The desired shape
        """
        pass

    @abc.abstractmethod
    def rollout_parametrized_policy(self, key, init_state, policy, theta, shift_reward=False):
        """Rolls out the policy with parameters theta."""
        pass

    @abc.abstractmethod
    def rollout_parametrized_policy_batched(self, key, batch_init_states, policy, theta, shift_reward=False):
        """Performs a batch of roll outs of the policy with parameters theta"""
        pass

    @abc.abstractmethod
    def evaluate_action_trajectory(self, key, init_state, actions, shift_reward=False):
        """Evaluates an action trajectory."""
        pass

    @abc.abstractmethod
    def evaluate_action_trajectory_batched(self, key, batch_init_states, batch_actions, shift_reward=False):
        """Evaluates a batch of action trajectories."""
        pass
