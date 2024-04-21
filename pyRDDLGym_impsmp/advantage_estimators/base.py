"""Common interface for Advantage estimators"""
import abc

class AdvEstimator(abc.ABC):
    @abc.abstractmethod
    def initialize_estimator_state(self, key, state_dim, action_dim):
        pass

    @abc.abstractmethod
    def estimate(self, key, states, actions, rewards, estimator_state):
        pass

    @abc.abstractmethod
    def print_report(self, it):
        pass
