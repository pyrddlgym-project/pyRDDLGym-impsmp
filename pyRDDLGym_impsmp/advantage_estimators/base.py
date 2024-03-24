"""Common interface for Advantage estimators"""
from abc import ABC, abstractmethod


class AdvEstimator(ABC):
    def __init__(self, key):
        pass

    @abstractmethod
    def initialize_estimator_state(self, key, state_dim, action_dim):
        pass

    @abstractmethod
    def estimate(self, key, states, actions, rewards, estimator_state):
        pass
