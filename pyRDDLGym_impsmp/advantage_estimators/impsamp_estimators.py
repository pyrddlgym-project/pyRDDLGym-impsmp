from pyRDDLGym_impsmp.advantage_estimators.base import AdvEstimator
from collections import deque
import numpy as np
import jax.numpy as jnp
import jax.nn
import optax
import haiku as hk
import pyRDDLGym_impsmp.registry


class SamplingModelQFunctionAdvEstimator(AdvEstimator):
    """
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

    def estimate(self, key, states, actions, _, estimator_state):
        print(states.shape)
        print(actions.shape)
        advantages = self.Q.apply(estimator_state['Q_theta'], key, states, actions)
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

    def update_theta(self, key):
        """TODO: make this work"""
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
