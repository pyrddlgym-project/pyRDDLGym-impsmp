"""Uniform interface for different types of advantage estimators

For a useful reference, please see

J. Schulman, P. Moritz, S. Levine, M.I. Jordan, P. Abbeel. "High-Dimensional
Continuous Control Using Generalized Advantage Estimation". ICLR 2016
"""
from abc import ABC, abstractmethod
from collections import deque
import numpy as np
import jax.numpy as jnp
import jax.nn
import optax
import haiku as hk
import pyRDDLGym_impsmp.registry


def compute_future_discounted_traj_rewards(rewards, gamma):
    """Given a batch of reward sequences of the form
           r_0, r_1, r_2, ..., r_{T-1}
       and a discount factor gamma, computes the quantities
           G_t = sum_{t'=t}^{T-1} gamma^{t'-t} r_{t'}
       for each sequence in the batch
    """
    B, T = rewards.shape
    gammas = jnp.roll(jnp.cumprod(gamma * jnp.ones(T)), 1)
    gammas = gammas.at[0].set(1.0)
    discounted_rewards = rewards * gammas
    flipped_discounted_rewards = jnp.flip(discounted_rewards, axis=1)
    flipped_future_disc_trajrews = jnp.cumsum(flipped_discounted_rewards, axis=1)
    future_disc_trajrews = jnp.flip(flipped_future_disc_trajrews, axis=1) / gammas
    return future_disc_trajrews


class AdvEstimator(ABC):
    def __init__(self, key):
        pass

    @abstractmethod
    def initialize_estimator_state(self, key, state_dim, action_dim):
        pass

    @abstractmethod
    def estimate(self, key, states, actions, rewards, estimator_state):
        pass



class TotalTrajRewardAdvEstimator(AdvEstimator):
    """The advantages are estimated using the total trajectory reward, i.e.
           A(s_t, a_t) = \sum_{t'=0}^T  r_{t'}
       for all t
    """
    def initialize_estimator_state(self, key, state_dim, action_dim):
        return {}

    def estimate(self, key, states, actions, rewards, estimator_state):
        T = rewards.shape[1]
        trajrews = jnp.sum(rewards, axis=1)
        advantages = jnp.repeat(trajrews[:,jnp.newaxis], T, axis=1)
        return advantages, estimator_state


class FutureTrajRewardAdvEstimator(AdvEstimator):
    """The advantages are estimated using the trajectory rewards beginning
       with the current timestep, i.e.
           A(s_t, a_t) = \sum_{t'=t}^T r_{t'}
       for all t
    """
    def __init__(self, key, gamma):
        self.gamma = gamma
        assert 0.0 < self.gamma <= 1.0

    def initialize_estimator_state(self, key, state_dim, action_dim):
        return {}

    def estimate(self, key, states, actions, rewards, estimator_state):
        advantages = compute_future_discounted_traj_rewards(rewards, self.gamma)
        return advantages, estimator_state


class FutureTrajRewardWConstantBaselineAdvEstimator(AdvEstimator):
    """The advantages are estimated using the trajectory rewards beginning
       with the current timestep, with respect to a constant baseline, i.e.
           A(s_t, a_t) = (\sum_{t'=t}^T r_{t'}) - b
       where b is a fixed constant provided at initialization
    """
    def __init__(self, key, gamma, base_val):
        self.gamma = gamma
        self.base_val = base_val

    def initialize_estimator_state(self, key, state_dim, action_dim):
        return {}

    def estimate(self, key, states, actions, rewards, estimator_state):
        future_disc_trajrews = compute_future_discounted_traj_rewards(rewards, self.gamma)
        advantages = future_disc_trajrews - self.base_val
        return advantages, estimator_state


class FutureTrajRewardWRunningAvgBaselineAdvEstimator(AdvEstimator):
    """The advantages are estimated using the trajectory rewards beginning
       with the current timestep, with respect to a running average baseline, i.e.
           A(s_t, a_t) = (\sum_{t'=t}^T r_{t'}) - b
       where b is a running average of state-values
    """
    def __init__(self, key, gamma):
        self.gamma = gamma

    def initialize_estimator_state(self, key, state_dim, action_dim):
        estimator_state = {
            'base_val': 0.0,
            'num_seen_state_vals': 0
        }
        return estimator_state

    def estimate(self, key, states, actions, rewards):
        future_disc_trajrews = compute_future_discounted_traj_rewards(rewards, self.gamma)

        # update the baseline
        B, T = rewards.shape
        num_new_state_vals = B * T
        num_seen_state_vals = estimator_state['num_seen_state_vals']
        estimator_state['base_val'] = (num_seen_state_vals * estimator_state['base_val'] + jnp.sum(future_disc_trajrews)) / (num_seen_state_vals + num_new_state_vals)
        estimator_state['num_seen_state_vals'] += num_new_state_vals

        advantages = future_disc_trajrews - estimator_state['base_val']
        return advantages, estimator_state


class FutureTrajRewardWLearnedBaselineAdvEstimator(AdvEstimator):
    """The advantages are estimated using the trajectory rewards beginning
       with the current timestep, with respect to a constant baseline, i.e.
           A(s_t, a_t) = (\sum_{t'=t}^T gamma^{t'-t} r_{t'}) - V(s_t)
       where V is a learned state value function, which is updated to
       minimize the loss
           |B|^{-1} \sum_B \sum_{t=0}^T (V(s_t) - (\sum_{t'=t}^T gamma^{t'-t} r_t))^2
       (B denotes the sample of rollouts)
    """
    def __init__(self, key, gamma, num_hidden_nodes_V, optimizer):
        self.gamma = gamma

        def V(state):
            """The V-function is parametrized by a MLP"""
            hidden_layers = []
            for n in num_hidden_nodes_V:
                hidden_layers.extend([hk.Linear(n), jax.nn.relu])
            mlp = hk.Sequential(hidden_layers + [hk.Linear(1)])
            V_val = mlp(state)
            return V_val

        self.V = hk.transform(V)

        optimizer_cls = pyRDDLGym_impsmp.registry.registry.optimizer_lookup_table[optimizer['type']]
        self.optimizer = optimizer_cls(**optimizer['params'])

    def initialize_estimator_state(self, key, state_dim, action_dim):
        estimator_state = {}

        dummy_state = jnp.ones(state_dim)
        estimator_state['V_theta'] = self.V.init(key, dummy_state)
        estimator_state['opt_state'] = self.optimizer.init(estimator_state['V_theta'])
        return estimator_state

    def loss(self, key, future_disc_trajrews, states, V_theta):
        state_vals = self.V.apply(V_theta, key, states)[..., 0]
        err = future_disc_trajrews - state_vals
        sqerr = err * err
        cmlt_sqerr = jnp.mean(sqerr, axis=1)
        return jnp.mean(cmlt_sqerr, axis=0)

    def estimate(self, key, states, actions, rewards, estimator_state):
        future_disc_trajrews = compute_future_discounted_traj_rewards(rewards, self.gamma)

        # compute advantages
        state_vals = self.V.apply(estimator_state['V_theta'], key, states)[..., 0]
        advantages = future_disc_trajrews - state_vals

        # update V to minimize squared error
        dloss = jax.grad(self.loss, argnums=3)
        grads = dloss(key, future_disc_trajrews, states, estimator_state['V_theta'])
        updates, estimator_state['opt_state'] = self.optimizer.update(grads, estimator_state['opt_state'])
        estimator_state['V_theta'] = optax.apply_updates(estimator_state['V_theta'], updates)

        return advantages, estimator_state


class QFunctionAdvEstimator(AdvEstimator):
    """The advantages are computed using a learned Q-function (state-action value function)
           A(s_t, a_t) = Q(s_t, a_t)
       where Q is updated to minimize the squared TD-loss.

       Sensitive to learning rates!
    """
    def __init__(self, key, gamma, num_hidden_nodes_Q, target_update_freq, grad_clip_val, optimizer):
        self.gamma = gamma
        self.grad_clip_val = grad_clip_val

        def Q(state_action_pair):
            """The Q-function is parameterized by a MLP"""
            hidden_layers = []
            for n in num_hidden_nodes_Q:
                hidden_layers.extend([hk.Linear(n), jax.nn.relu])
            mlp = hk.Sequential(hidden_layers + [hk.Linear(1)])
            Q_val = mlp(state_action_pair)
            return Q_val
        self.Q = hk.transform(Q)

        optimizer_cls = pyRDDLGym_impsmp.registry.registry.optimizer_lookup_table[optimizer['type']]
        self.optimizer = optimizer_cls(**optimizer['params'])

        # idea of target network from DQN, but repurposed for policy evaluation,
        # not policy improvement
        self.target_update_freq = target_update_freq

    def initialize_estimator_state(self, key, state_dim, action_dim):
        estimator_state = {}
        dummy_state_action_pair = jnp.ones(state_dim + action_dim)
        estimator_state['Q_theta'] = self.Q.init(key, dummy_state_action_pair)
        estimator_state['Q_theta_target'] = jax.tree_util.tree_map(lambda Q_theta_term: jnp.copy(Q_theta_term), estimator_state['Q_theta'])
        estimator_state['opt_state'] = self.optimizer.init(estimator_state['Q_theta'])
        estimator_state['target_update_counter'] = 0
        return estimator_state

    def update_Q_target(self, estimator_state):
        """Functional approach to periodically updating the target Q-values.
           Otherwise conflicts with jit.
        """
        estimator_state['Q_theta_target'] = jax.tree_util.tree_map(lambda Q_theta_term: jnp.copy(Q_theta_term), estimator_state['Q_theta'])
        estimator_state['target_update_counter'] = 0
        return estimator_state

    def skip_update_Q_target(self, estimator_state):
        """Called when it is not time to call 'update_Q_target'"""
        return estimator_state

    def TD_loss(self, key, gamma, state_action_pairs, rewards, Q_theta_target, Q_theta):
        """The TD error is given by
               [r(s_t, a_t) + gamma * Q(s_{t+1}, a_{t+1})] - Q(s_t, a_t)
           The TD loss is the square of TD error summed over the rollout
           and averaged across the rollout sample
        """
        Q_vals = self.Q.apply(Q_theta, key, state_action_pairs)[..., 0]
        Q_val_targets = self.Q.apply(Q_theta_target, key, state_action_pairs)[..., 0]
        TD_err = (rewards[..., :-1] + gamma * Q_val_targets[..., 1:]) - Q_vals[..., :-1]
        sq_TD_err = TD_err * TD_err
        cmlt_sq_TD_err = jnp.sum(sq_TD_err, axis=1)
        loss = jnp.mean(cmlt_sq_TD_err, axis=0)
        return loss

    def estimate(self, key, states, actions, rewards, estimator_state):
        B, T = rewards.shape
        state_action_pairs = jnp.concatenate([states, actions], axis=2)
        advantages = self.Q.apply(estimator_state['Q_theta'], key, state_action_pairs)[..., 0]

        # update Q to minimize TD-loss
        dTD_loss = jax.grad(self.TD_loss, argnums=5)
        grads = dTD_loss(key, self.gamma, state_action_pairs, rewards, estimator_state['Q_theta_target'], estimator_state['Q_theta'])
        grads = jax.tree_util.tree_map(lambda grad_term: jnp.clip(grad_term, -self.grad_clip_val, self.grad_clip_val), grads)
        updates, estimator_state['opt_state'] = self.optimizer.update(grads, estimator_state['opt_state'])
        estimator_state['Q_theta'] = optax.apply_updates(estimator_state['Q_theta'], updates)

        estimator_state['target_update_counter'] += T
        estimator_state = jax.lax.cond(
            estimator_state['target_update_counter'] > self.target_update_freq,
            self.update_Q_target,
            self.skip_update_Q_target,
            estimator_state)

        return advantages, estimator_state

class AFunctionAdvEstimator(AdvEstimator):
    """The advantages are computed using learned Q-function (state-action value function) and
       V-function (state value function) as
           A(s_t, a_t) = Q(s_t, a_t) - V(s_t)
       where Q and V are updated to minimize the TD loss and squared error with the observed
       cumulative rewards, respectively.
    """
    def estimate(self, key, states, actions, rewards):
        raise NotImplementedError


class TDResidualAdvEstimator(AdvEstimator):
    """ . """
    def estimate(self, key, states, actions, rewards):
        raise NotImplementedError
