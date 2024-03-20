"""Uniform interface for different types of advantage estimators

For a useful reference, please see

J. Schulman, P. Moritz, S. Levine, M.I. Jordan, P. Abbeel. "High-Dimensional
Continuous Control Using Generalized Advantage Estimation". ICLR 2016
"""
from abc import ABC, abstractmethod
import numpy as np
import jax.numpy as jnp

class AdvEstimator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def estimate(self, states, actions, rewards):
        pass



class TotalTrajRewardAdvEstimator(AdvEstimator):
    """The advantages are estimated using the total trajectory reward, i.e.
           A(s_t, a_t) = \sum_{t'=0}^T  r_{t'}
       for all t
    """
    def estimate(self, states, actions, rewards):
        T = rewards.shape[1]
        trajrews = jnp.sum(rewards, axis=1)
        advantages = jnp.repeat(trajrews[:,jnp.newaxis], T, axis=1)
        return advantages


class FutureTrajRewardAdvEstimator(AdvEstimator):
    """The advantages are estimated using the trajectory rewards beginning
       with the current timestep, i.e.
           A(s_t, a_t) = \sum_{t'=t}^T r_{t'}
       for all t
    """
    def estimate(self, states, actions, rewards):
        flipped_rewards = jnp.flip(rewards, axis=1)
        future_trajrews = jnp.cumsum(flipped_rewards, axis=1)
        advantages = jnp.flip(future_trajrews, axis=1)
        return advantages


class FutureTrajRewardWConstantBaselineAdvEstimator(AdvEstimator):
    """The advantages are estimated using the trajectory rewards beginning
       with the current timestep, with respect to a constant baseline, i.e.
           A(s_t, a_t) = (\sum_{t'=t}^T r_{t'}) - b
    """
    def __init__(self, base_val):
        self.base_val = base_val

    def estimate(self, states, actions, rewards):
        flipped_rewards = jnp.flip(rewards, axis=1)
        future_trajrews = jnp.cumsum(flipped_rewards, axis=1)
        advantages = jnp.flip(future_trajrews, axis=1)
        advantages = advantages - self.base_val
        return advantages


class QFunctionAdvEstimator(AdvEstimator):
    """The advantages are computed using a learned Q-function (state-action value function)
           A(s_t, a_t) = Q(s_t, a_t)
       where Q is updated as
    """
    def __init__(self, Q_fn_config):
        pass

    def estimate(self, states, actions, rewards):
        raise NotImplementedError


class AFunctionAdvEstimator(AdvEstimator):
    """The advantages are computed using learned Q-function (state-action value function) and
       V-function (state value function) as
           A(s_t, a_t) = Q(s_t, a_t) - V(s_t)
       where Q and V are updated as
    """
    def estimate(self, states, actions, rewards):
        raise NotImplementedError


class TDResidualAdvEstimator(AdvEstimator):
    """ . """
    def estimate(self, states, actions, rewards):
        raise NotImplementedError
