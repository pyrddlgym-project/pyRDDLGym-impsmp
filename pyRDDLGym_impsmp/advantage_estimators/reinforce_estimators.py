"""Uniform interface for different types of advantage estimators

For a useful reference, please see

J. Schulman, P. Moritz, S. Levine, M.I. Jordan, P. Abbeel. "High-Dimensional
Continuous Control Using Generalized Advantage Estimation". ICLR 2016
"""
from pyRDDLGym_impsmp.advantage_estimators.base import AdvEstimator
from collections import deque
import numpy as np
import jax.numpy as jnp
import jax.nn
import optax
import haiku as hk
import pyRDDLGym_impsmp.registry


def compute_future_discounted_traj_rewards(rewards, gamma):
    """Given a reward sequence of the form
           r_0, r_1, r_2, ..., r_{T-1}
    and a discount factor gamma, computes the quantities
           G_t = sum_{t'=t}^{T-1} gamma^{t'-t} r_{t'}
    """
    T = rewards.shape
    gammas = jnp.roll(jnp.cumprod(gamma * jnp.ones(T)), 1)
    gammas = gammas.at[0].set(1.0)
    discounted_rewards = rewards * gammas
    flipped_discounted_rewards = jnp.flip(discounted_rewards)
    flipped_future_disc_trajrews = jnp.cumsum(flipped_discounted_rewards)
    future_disc_trajrews = jnp.flip(flipped_future_disc_trajrews) / gammas
    return future_disc_trajrews



class TotalTrajRewardAdvEstimator(AdvEstimator):
    """The advantages are estimated using the total trajectory reward, i.e.
           A(s_t, a_t) = \sum_{t'=0}^T  r_{t'}
    for all t
    """
    def __init__(self, key):
        pass

    def initialize_estimator_state(self, key, state_dim, action_dim):
        return key, {}

    def estimate(self, key, states, actions, rewards, estimator_state):
        T = rewards.shape[0]
        trajrews = jnp.sum(rewards)
        advantages = jnp.repeat(trajrews, T)
        return key, advantages, estimator_state

    def print_report(self, it):
        print('\tAdv.est :: Type=Total trajectory reward')


class FutureTrajRewardAdvEstimator(AdvEstimator):
    """The advantages are estimated using the trajectory rewards beginning
    with the current timestep, and possibly discounting, i.e.
           A(s_t, a_t) = \sum_{t'=t}^T gamma^{t'-t} r_{t'}
    for all t
    """
    def __init__(self, key, gamma):
        self.gamma = gamma
        assert 0.0 < self.gamma <= 1.0

    def initialize_estimator_state(self, key, state_dim, action_dim):
        return key, {}

    def estimate(self, key, states, actions, rewards, estimator_state):
        advantages = compute_future_discounted_traj_rewards(rewards, self.gamma)
        return key, advantages, estimator_state

    def print_report(self, it):
        print(f'\tAdv.est :: Type=Future trajectory reward :: Gamma={self.gamma:.2f}')


class FutureTrajRewardWConstantBaselineAdvEstimator(AdvEstimator):
    """The advantages are estimated using the trajectory rewards beginning
    with the current timestep, with respect to a constant baseline, i.e.
           A(s_t, a_t) = (\sum_{t'=t}^T r_{t'}) - b
    where b is a fixed constant provided at initialization
    """
    def __init__(self, key, gamma, base_val):
        self.gamma = gamma
        self.base_val = base_val
        assert 0.0 < self.gamma <= 1.0

    def initialize_estimator_state(self, key, state_dim, action_dim):
        return key, {}

    def estimate(self, key, states, actions, rewards, estimator_state):
        future_disc_trajrews = compute_future_discounted_traj_rewards(rewards, self.gamma)
        advantages = future_disc_trajrews - self.base_val
        return key, advantages, estimator_state

    def print_report(self, it):
        print(f'\tAdv.est :: Type=Future trajectory reward with constant baseline :: Gamma={self.gamma:.2f} :: Baseline={self.base_val:.2f}')



# TODO: Update all of the classes below to make un-batched
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
        return key, estimator_state

    def estimate(self, key, states, actions, rewards):
        future_disc_trajrews = compute_future_discounted_traj_rewards(rewards, self.gamma)

        # update the baseline
        B, T = rewards.shape
        num_new_state_vals = B * T
        num_seen_state_vals = estimator_state['num_seen_state_vals']
        estimator_state['base_val'] = (num_seen_state_vals * estimator_state['base_val'] + jnp.sum(future_disc_trajrews)) / (num_seen_state_vals + num_new_state_vals)
        estimator_state['num_seen_state_vals'] += num_new_state_vals

        advantages = future_disc_trajrews - estimator_state['base_val']
        return key, advantages, estimator_state


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
            return V_val[..., 0]

        self.V = hk.transform(V)

        optimizer_cls = pyRDDLGym_impsmp.registry.registry.optimizer_lookup_table[optimizer['type']]
        self.optimizer = optimizer_cls(**optimizer['params'])

    def initialize_estimator_state(self, key, state_dim, action_dim):
        estimator_state = {}

        key, subkey = jax.random.split(key)

        dummy_state = jnp.ones(state_dim)
        estimator_state['V_theta'] = self.V.init(subkey, dummy_state)
        estimator_state['opt_state'] = self.optimizer.init(estimator_state['V_theta'])
        return key, estimator_state

    def loss(self, key, future_disc_trajrews, states, V_theta):
        state_vals = self.V.apply(V_theta, key, states)
        err = future_disc_trajrews - state_vals
        sqerr = err * err
        cmlt_sqerr = jnp.mean(sqerr, axis=1)
        return jnp.mean(cmlt_sqerr, axis=0)

    def estimate(self, key, states, actions, rewards, estimator_state):
        future_disc_trajrews = compute_future_discounted_traj_rewards(rewards, self.gamma)

        # compute advantages
        key, subkey = jax.random.split(key)
        state_vals = self.V.apply(estimator_state['V_theta'], subkey, states)
        advantages = future_disc_trajrews - state_vals

        # update V to minimize squared error
        dloss = jax.grad(self.loss, argnums=3)
        grads = dloss(key, future_disc_trajrews, states, estimator_state['V_theta'])
        updates, estimator_state['opt_state'] = self.optimizer.update(grads, estimator_state['opt_state'])
        estimator_state['V_theta'] = optax.apply_updates(estimator_state['V_theta'], updates)

        return key, advantages, estimator_state


class QFunctionAdvEstimator(AdvEstimator):
    """The advantages are computed using a learned Q-function (state-action value function)
           A(s_t, a_t) = Q(s_t, a_t)
    where Q is updated to minimize the squared TD-loss.

    Sensitive to learning rates!
    """
    def __init__(self, key, gamma, num_hidden_nodes_Q, target_update_freq, grad_clip_val, optimizer):
        self.gamma = gamma
        self.grad_clip_val = grad_clip_val

        def Q(states, actions):
            """The Q-function is parameterized by a MLP"""
            hidden_layers = []
            for n in num_hidden_nodes_Q:
                hidden_layers.extend([hk.Linear(n), jax.nn.relu])
            mlp = hk.Sequential(hidden_layers + [hk.Linear(1)])
            state_action_pairs = jnp.concatenate([states, actions], axis=-1)
            Q_val = mlp(state_action_pairs)
            return Q_val[..., 0]
        self.Q = hk.transform(Q)

        optimizer_cls = pyRDDLGym_impsmp.registry.registry.optimizer_lookup_table[optimizer['type']]
        self.optimizer = optimizer_cls(**optimizer['params'])

        # idea of target network from DQN, but repurposed for policy evaluation,
        # not policy improvement
        self.target_update_freq = target_update_freq

    def initialize_estimator_state(self, key, state_dim, action_dim):
        estimator_state = {}

        key, subkey = jax.random.split(key)

        dummy_state, dummy_action = jnp.ones(state_dim), jnp.ones(action_dim)
        estimator_state['Q_theta'] = self.Q.init(subkey, dummy_state, dummy_action)
        estimator_state['Q_theta_target'] = jax.tree_util.tree_map(lambda Q_theta_term: jnp.copy(Q_theta_term), estimator_state['Q_theta'])
        estimator_state['opt_state'] = self.optimizer.init(estimator_state['Q_theta'])
        estimator_state['target_update_counter'] = 0
        return key, estimator_state

    def dQ_a(self, key, states, actions, estimator_state):
        """Calculates the partials of the current Q-function with respect to the actions"""
        dQ_a_termwise = jax.grad(self.Q.apply, argnums=3)
        dQ_a_over_batch = jax.vmap(dQ_a_termwise, in_axes=(None, None, 0, 0), out_axes=0)
        dQ_a_over_batch_and_time = jax.vmap(dQ_a_over_batch, in_axes=(None, None, 0, 0), out_axes=0)
        return dQ_a_over_batch_and_time(estimator_state['Q_theta'], key, states, actions)

    def TD_loss(self, key, gamma, states, actions, rewards, Q_theta_target, Q_theta):
        """The TD error is given by
               [r(s_t, a_t) + gamma * Q(s_{t+1}, a_{t+1})] - Q(s_t, a_t)
        The TD loss is the square of TD error summed over the rollout
        and averaged across the rollout sample
        """
        Q_vals = self.Q.apply(Q_theta, key, states, actions)
        Q_val_targets = self.Q.apply(Q_theta_target, key, states, actions)
        TD_err = (rewards[..., :-1] + gamma * Q_val_targets[..., 1:]) - Q_vals[..., :-1]
        sq_TD_err = TD_err * TD_err
        cmlt_sq_TD_err = jnp.sum(sq_TD_err, axis=1)
        loss = jnp.mean(cmlt_sq_TD_err, axis=0)
        return loss

    def estimate(self, key, states, actions, rewards, estimator_state):
        B, T = rewards.shape
        advantages = self.Q.apply(estimator_state['Q_theta'], key, states, actions)

        # update Q to minimize TD-loss
        key, subkey = jax.random.split(key)
        dTD_loss = jax.grad(self.TD_loss, argnums=6)
        grads = dTD_loss(subkey, self.gamma, states, actions, rewards, estimator_state['Q_theta_target'], estimator_state['Q_theta'])
        grads = jax.tree_util.tree_map(lambda grad_term: jnp.clip(grad_term, -self.grad_clip_val, self.grad_clip_val), grads)
        updates, estimator_state['opt_state'] = self.optimizer.update(grads, estimator_state['opt_state'])
        estimator_state['Q_theta'] = optax.apply_updates(estimator_state['Q_theta'], updates)

        estimator_state['target_update_counter'] += T
        estimator_state = jax.lax.cond(
            estimator_state['target_update_counter'] > self.target_update_freq,
            self.update_Q_target,
            self.skip_update_Q_target,
            estimator_state)

        return key, advantages, estimator_state

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


class AFunctionAdvEstimator(AdvEstimator):
    """The advantages are computed using learned Q-function (state-action value function) and
    V-function (state value function) as
            A(s_t, a_t) = Q(s_t, a_t) - V(s_t)
    where Q and V are updated to minimize the TD loss and squared error with the observed
    cumulative rewards, respectively.
    """
    def __init__(self, key, gamma, num_hidden_nodes_V, num_hidden_nodes_Q, target_update_freq, grad_clip_val, optimizer):
        self.gamma = gamma
        self.grad_clip_val = grad_clip_val

        def V(states):
            """The V-function is parameterized by a MLP"""
            hidden_layers = []
            for n in num_hidden_nodes_V:
                hidden_layers.extend([hk.Linear(n), jax.nn.relu])
            mlp = hk.Sequential(hidden_layers + [hk.Linear(1)])
            V_val = mlp(states)
            return V_val[..., 0]

        def Q(states, actions):
            """The Q-function is parameterized by a MLP"""
            hidden_layers = []
            for n in num_hidden_nodes_Q:
                hidden_layers.extend([hk.Linear(n), jax.nn.relu])
            mlp = hk.Sequential(hidden_layers + [hk.Linear(1)])
            state_action_pairs = jnp.concatenate([states, actions], axis=-1)
            Q_val = mlp(state_action_pairs)
            return Q_val[..., 0]

        self.V = hk.transform(V)
        self.Q = hk.transform(Q)

        optimizer_cls = pyRDDLGym_impsmp.registry.registry.optimizer_lookup_table[optimizer['type']]
        self.V_optimizer = optimizer_cls(**optimizer['params'])
        self.Q_optimizer = optimizer_cls(**optimizer['params'])

        # idea of target network from DQN, but repurposed for policy evaluation,
        # not policy improvement
        self.target_update_freq = target_update_freq

    def initialize_estimator_state(self, key, state_dim, action_dim):
        estimator_state = {}

        key, subkey = jax.random.split(key)

        dummy_state, dummy_action = jnp.ones(state_dim), jnp.ones(action_dim)
        estimator_state['V_theta'] = self.V.init(subkey, dummy_state)
        estimator_state['Q_theta'] = self.Q.init(subkey, dummy_state, dummy_action)
        estimator_state['Q_theta_target'] = jax.tree_util.tree_map(lambda Q_theta_term: jnp.copy(Q_theta_term), estimator_state['Q_theta'])
        estimator_state['V_opt_state'] = self.V_optimizer.init(estimator_state['V_theta'])
        estimator_state['Q_opt_state'] = self.Q_optimizer.init(estimator_state['Q_theta'])
        estimator_state['target_update_counter'] = 0
        return key, estimator_state

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

    def regr_loss(self, key, future_disc_trajrews, states, V_theta):
        """Regression loss to fit the V state-value function"""
        state_vals = self.V.apply(V_theta, key, states)
        err = future_disc_trajrews - state_vals
        sqerr = err * err
        cmlt_sqerr = jnp.mean(sqerr, axis=1)
        return jnp.mean(cmlt_sqerr, axis=0)

    def TD_loss(self, key, gamma, states, actions, rewards, Q_theta_target, Q_theta):
        """The TD error is given by
               [r(s_t, a_t) + gamma * Q(s_{t+1}, a_{t+1})] - Q(s_t, a_t)
        The TD loss is the square of TD error summed over the rollout
        and averaged across the rollout sample
        """
        Q_vals = self.Q.apply(Q_theta, key, states, actions)
        Q_val_targets = self.Q.apply(Q_theta_target, key, states, actions)
        TD_err = (rewards[..., :-1] + gamma * Q_val_targets[..., 1:]) - Q_vals[..., :-1]
        sq_TD_err = TD_err * TD_err
        cmlt_sq_TD_err = jnp.sum(sq_TD_err, axis=1)
        loss = jnp.mean(cmlt_sq_TD_err, axis=0)
        return loss

    def estimate(self, key, states, actions, rewards, estimator_state):
        B, T = rewards.shape
        future_disc_trajrews = compute_future_discounted_traj_rewards(rewards, self.gamma)

        V_vals = self.V.apply(estimator_state['V_theta'], key, states)
        Q_vals = self.Q.apply(estimator_state['Q_theta'], key, states, actions)

        # compute advantages
        advantages = Q_vals - V_vals

        key, subkey = jax.random.split(key)

        # update V to minimize squared error
        dloss = jax.grad(self.regr_loss, argnums=3)
        grads = dloss(subkey, future_disc_trajrews, states, estimator_state['V_theta'])
        grads = jax.tree_util.tree_map(lambda grad_term: jnp.clip(grad_term, -self.grad_clip_val, self.grad_clip_val), grads)
        updates, estimator_state['V_opt_state'] = self.V_optimizer.update(grads, estimator_state['V_opt_state'])
        estimator_state['V_theta'] = optax.apply_updates(estimator_state['V_theta'], updates)

        # update Q to minimize TD-loss
        dTD_loss = jax.grad(self.TD_loss, argnums=6)
        grads = dTD_loss(subkey, self.gamma, states, actions, rewards, estimator_state['Q_theta_target'], estimator_state['Q_theta'])
        grads = jax.tree_util.tree_map(lambda grad_term: jnp.clip(grad_term, -self.grad_clip_val, self.grad_clip_val), grads)
        updates, estimator_state['Q_opt_state'] = self.Q_optimizer.update(grads, estimator_state['Q_opt_state'])
        estimator_state['Q_theta'] = optax.apply_updates(estimator_state['Q_theta'], updates)

        estimator_state['target_update_counter'] += T
        estimator_state = jax.lax.cond(
            estimator_state['target_update_counter'] > self.target_update_freq,
            self.update_Q_target,
            self.skip_update_Q_target,
            estimator_state)

        return key, advantages, estimator_state



class TDResidualAdvEstimator(AdvEstimator):
    """The advantages are estimated using the TD error computed using
    a learned V state-value function, i.e.
           A(s_t, a_t) = (r(s_t, a_t) + gamma V(s'_t)) - V(s_t)
    where V is a learned state value function. V is updated to minimize
    the loss
           |B|^{-1} \sum_B \sum_{t=0}^T (V(s_t) - (\sum_{t'=t}^T gamma^{t'-t} r_t))^2
    (B denotes the sample of rollouts)
    """
    def __init__(self, key, gamma, num_hidden_nodes_V, grad_clip_val, optimizer):
        self.gamma = gamma
        self.grad_clip_val = grad_clip_val

        def V(state):
            """The V-function is parametrized by a MLP"""
            hidden_layers = []
            for n in num_hidden_nodes_V:
                hidden_layers.extend([hk.Linear(n), jax.nn.relu])
            mlp = hk.Sequential(hidden_layers + [hk.Linear(1)])
            V_val = mlp(state)
            return V_val[..., 0]

        self.V = hk.transform(V)

        optimizer_cls = pyRDDLGym_impsmp.registry.registry.optimizer_lookup_table[optimizer['type']]
        self.optimizer = optimizer_cls(**optimizer['params'])

    def initialize_estimator_state(self, key, state_dim, action_dim):
        estimator_state = {}

        key, subkey = jax.random.split(key)

        dummy_state = jnp.ones(state_dim)
        estimator_state['V_theta'] = self.V.init(subkey, dummy_state)
        estimator_state['opt_state'] = self.optimizer.init(estimator_state['V_theta'])
        return key, estimator_state

    def loss(self, key, future_disc_trajrews, states, V_theta):
        state_vals = self.V.apply(V_theta, key, states)
        err = future_disc_trajrews - state_vals
        sqerr = err * err
        cmlt_sqerr = jnp.mean(sqerr, axis=1)
        return jnp.mean(cmlt_sqerr, axis=0)

    def estimate(self, key, states, actions, rewards, estimator_state):
        future_disc_trajrews = compute_future_discounted_traj_rewards(rewards, self.gamma)

        # compute advantages
        state_vals = self.V.apply(estimator_state['V_theta'], key, states)
        state_vals = jnp.pad(state_vals, ((0, 0), (0, 1)))
        advantages = (rewards + self.gamma * state_vals[..., 1:]) - state_vals[..., :-1]

        key, subkey = jax.random.split(key)

        # update V to minimize squared error
        dloss = jax.grad(self.loss, argnums=3)
        grads = dloss(subkey, future_disc_trajrews, states, estimator_state['V_theta'])
        grads = jax.tree_util.tree_map(lambda grad_term: jnp.clip(grad_term, -self.grad_clip_val, self.grad_clip_val), grads)
        updates, estimator_state['opt_state'] = self.optimizer.update(grads, estimator_state['opt_state'])
        estimator_state['V_theta'] = optax.apply_updates(estimator_state['V_theta'], updates)

        return key, advantages, estimator_state
