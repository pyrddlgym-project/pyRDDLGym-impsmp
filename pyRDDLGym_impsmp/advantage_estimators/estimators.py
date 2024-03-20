"""Uniform interface for different types of advantage estimators

For a useful reference, please see

J. Schulman, P. Moritz, S. Levine, M.I. Jordan, P. Abbeel. "High-Dimensional
Continuous Control Using Generalized Advantage Estimation". ICLR 2016
"""
from abc import ABC, abstractmethod
import numpy as np
import jax.numpy as jnp
import jax.nn
import optax
import haiku as hk

class AdvEstimator(ABC):
    def __init__(self, key, state_dim, action_dim):
        pass

    @abstractmethod
    def estimate(self, key, states, actions, rewards):
        pass


#TODO: Add gamma

class TotalTrajRewardAdvEstimator(AdvEstimator):
    """The advantages are estimated using the total trajectory reward, i.e.
           A(s_t, a_t) = \sum_{t'=0}^T  r_{t'}
       for all t
    """
    def estimate(self, key, states, actions, rewards):
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
    def estimate(self, key, states, actions, rewards):
        flipped_rewards = jnp.flip(rewards, axis=1)
        flipped_future_trajrews = jnp.cumsum(flipped_rewards, axis=1)
        advantages = jnp.flip(flipped_future_trajrews, axis=1)
        return advantages


class FutureTrajRewardWConstantBaselineAdvEstimator(AdvEstimator):
    """The advantages are estimated using the trajectory rewards beginning
       with the current timestep, with respect to a constant baseline, i.e.
           A(s_t, a_t) = (\sum_{t'=t}^T r_{t'}) - b
    """
    def __init__(self, key, state_dim, action_dim, base_val):
        self.base_val = base_val

    def estimate(self, key, states, actions, rewards):
        flipped_rewards = jnp.flip(rewards, axis=1)
        flipped_future_trajrews = jnp.cumsum(flipped_rewards, axis=1)
        advantages = jnp.flip(flipped_future_trajrews, axis=1)
        advantages = advantages - self.base_val
        return advantages

class FutureTrajRewardWLearnedBaselineAdvEstimator(AdvEstimator):
    """The advantages are estimated using the trajectory rewards beginning
       with the current timestep, with respect to a constant baseline, i.e.
           A(s_t, a_t) = (\sum_{t'=t}^T r_{t'}) - V(s_t)
       where V is a learned state value function, which is updated to
       minimize the loss
           |B|^{-1} \sum_B \sum_{t=0}^T (V(s_t) - (\sum_{t'=t}^T r_t))^2
       (B denotes the sample of rollouts)
    """
    def __init__(self, key, state_dim, action_dim, num_hidden_nodes_V):
        def V(state):
            """The V-function is parametrized by a MLP"""
            hidden_layers = []
            for n in num_hidden_nodes_V:
                hidden_layers.extend([hk.Linear(n), jax.nn.relu])
            mlp = hk.Sequential(hidden_layers + [hk.Linear(1)])
            V_val = mlp(state)
            return V_val

        self.V = hk.transform(V)
        dummy_state = jnp.ones(state_dim)
        self.V_theta = self.V.init(key, dummy_state)
        self.dloss = jax.grad(self.loss, argnums=3)

        self.optimizer = optax.adam(learning_rate=1e-3)
        self.opt_state = self.optimizer.init(self.V_theta)

    def loss(self, key, future_trajrews, states, V_theta):
        state_vals = self.V.apply(V_theta, key, states)[..., 0]
        err = future_trajrews - state_vals
        sqerr = err*err
        cmlt_sqerr = jnp.sum(sqerr, axis=1)
        return jnp.mean(cmlt_sqerr, axis=0)

    def estimate(self, key, states, actions, rewards):
        flipped_rewards = jnp.flip(rewards, axis=1)
        flipped_future_trajrews = jnp.cumsum(flipped_rewards, axis=1)
        future_trajrews = jnp.flip(flipped_future_trajrews, axis=1)

        # compute advantages
        state_vals = self.V.apply(self.V_theta, key, states)[..., 0]
        advantages = future_trajrews - state_vals

        # update V
        grads = self.dloss(key, future_trajrews, states, self.V_theta)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.V_theta = optax.apply_updates(self.V_theta, updates)

        return advantages



class QFunctionAdvEstimator(AdvEstimator):
    """The advantages are computed using a learned Q-function (state-action value function)
           A(s_t, a_t) = Q(s_t, a_t)
       where Q is updated as
    """
    def __init__(self, key, state_dim, action_dim, num_hidden_nodes):
        def Q(state_action_pair):
            """The Q-function is parameterized by a MLP"""
            hidden_layers = []
            for n in num_hidden_nodes:
                hidden_layers.extend([hk.Linear(n), jax.nn.relu])
            mlp = hk.Sequential(hidden_layers + [hk.Linear(1)])
            Q_val = mlp(state_action_pair)
            return Q_val
        self.Q = hk.transform(Q)
        dummy_state_action_pair = jnp.ones(state_dim + action_dim)
        self.Q_theta = self.Q.init(key, dummy_state_action_pair)

    def TD_loss(

    def estimate(self, key, states, actions, rewards):
        state_action_pairs = jnp.concatenate([states, actions], axis=2)
        Q_vals = self.Q.apply(self.Q_theta, key, state_action_pairs)[..., 0]

        # TODO: update Q_vals using TD-loss

        return Q_vals # == advantages

class AFunctionAdvEstimator(AdvEstimator):
    """The advantages are computed using learned Q-function (state-action value function) and
       V-function (state value function) as
           A(s_t, a_t) = Q(s_t, a_t) - V(s_t)
       where Q and V are updated as
    """
    def estimate(self, key, states, actions, rewards):
        raise NotImplementedError


class TDResidualAdvEstimator(AdvEstimator):
    """ . """
    def estimate(self, key, states, actions, rewards):
        raise NotImplementedError
