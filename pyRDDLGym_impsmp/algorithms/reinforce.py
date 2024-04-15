import numpy as np
import jax
import jax.numpy as jnp
import optax
import functools
from time import perf_counter as timer

def scalar_mult(alpha, v):
    pad_axes = tuple(range(len(alpha.shape), len(v.shape)))
    alpha = jnp.expand_dims(alpha, axis=pad_axes)
    return alpha * v

weighting_map = jax.vmap(
    scalar_mult, in_axes=0, out_axes=0)


@functools.partial(jax.jit, static_argnames=('batch_size', 'epsilon', 'policy', 'model', 'adv_estimator'))
def compute_reinforce_dJ_hat_estimate(key, theta, batch_size, epsilon, policy, model, adv_estimator, adv_estimator_state):
    """Computes an estimate of dJ^pi over a sample of rolled out trajectories

        B = {tau_0, tau_1, ..., tau_{|B|-1}}

    using the REINFORCE formula

        dJ hat = 1/|B| * sum_{tau in B} [ R(tau) * grad log pi(tau) ]

    where
        dJ denotes the gradient of J^pi with respect to theta,
        R(tau) = sum_{t=1}^T R(s_t, a_t)
        grad log pi(tau) = sum_{t=1}^T grad log pi(s_t, a_t)

    Args:
        key: JAX random key
        theta: Current policy pi parameters
        batch_size: Total number of samples
        epsilon: Small numerical stability constant
        policy: Object carrying static policy parameters
                (the values of the parameters are passed separately
                 in theta)
        model: Training RDDL model

    Returns:
        key: Mutated JAX random key
        dJ_hat: dJ estimator
        batch_stats: Dictionary of statistics for the current sample
            'dJ': Individual summands of the dJ estimator
            'actions': Sampled actions
    """
    jacobian = jax.jacrev(policy.pdf, argnums=1)

    def _compute_dJ_summand(init, _):
        """Generates a single summand in the expression for the dJ estimator"""
        key, adv_estimator_state = init
        key, init_states = model.batch_generate_initial_state(key, (model.state_dim,))
        key, states, actions, rewards = model.rollout_parametrized_policy(key, init_states, theta)
        key, advantages, adv_estimator_state = adv_estimator.estimate(key, states, actions, rewards, adv_estimator_state)
        pi_inv = 1 / (policy.pdf(key, theta, states, actions) + epsilon)
        dpi = jacobian(key, theta, states, actions)
        dlogpi = jax.tree_util.tree_map(lambda dpi_term: weighting_map(pi_inv, dpi_term), dpi)
        adv_weighted_dlogpi = jax.tree_util.tree_map(lambda dlogpi_term: weighting_map(advantages, dlogpi_term), dlogpi)

        dJ_summands = jax.tree_util.tree_map(lambda adv_weighted_dlogpi_term: jnp.sum(adv_weighted_dlogpi_term, axis=1), adv_weighted_dlogpi)
        carry = (key, adv_estimator_state)
        result = (actions, rewards, dJ_summands)
        return carry, result

    carry, result = jax.lax.scan(
        _compute_dJ_summand,
        (key, adv_estimator_state), [None] * batch_size, length=batch_size)
    (key, adv_estimator_state) = carry
    (actions, rewards, dJ_summands) = result

    dJ_hat = jax.tree_util.tree_map(lambda term: jnp.mean(term, axis=(0,1)), dJ_summands)

    batch_stats = {}
    return key, dJ_hat, adv_estimator_state, batch_stats

@functools.partial(jax.jit, static_argnames=('eval_batch_size', 'policy', 'model'))
def evaluate_policy(key, it, algo_stats, eval_batch_size, theta, policy, model):
    key, init_states = model.batch_generate_initial_state(key, (eval_batch_size, model.state_dim))
    key, states, actions, rewards = model.rollout_parametrized_policy_batched(key, init_states, theta)
    rewards = jnp.sum(rewards, axis=1) # sum rewards along the time axis
    key, subkey = jax.random.split(key)
    policy_mean, policy_cov = policy.apply(subkey, theta, states)

    algo_stats['reward_mean']  = algo_stats['reward_mean'].at[it].set(jnp.mean(rewards))
    algo_stats['reward_std']   = algo_stats['reward_std'].at[it].set(jnp.std(rewards))
    algo_stats['reward_sterr'] = algo_stats['reward_sterr'].at[it].set(algo_stats['reward_std'][it] / jnp.sqrt(eval_batch_size))
    return key, algo_stats

def print_reinforce_report(it, algo_stats, subt0, subt1):
    """Prints out the results for the current REINFORCE iteration to console"""
    print(f'Iter {it} :: REINFORCE :: Runtime={subt1-subt0}s')
    print(f'Eval. reward={algo_stats["reward_mean"][it]:.3f} \u00B1 {algo_stats["reward_sterr"][it]:.3f}\n')


def reinforce(key, n_iters, config, bijector, policy, sampler, optimizer, models, adv_estimator, adv_estimator_state):
    """Runs the REINFORCE algorithm"""
    train_model = models['train_model']
    eval_model = models.get('eval_model', train_model)

    state_dim = train_model.state_dim
    action_dim = train_model.action_dim
    batch_size = config['batch_size']
    eval_batch_size = config['eval_batch_size']

    epsilon = config.get('epsilon', 1e-12)
    save_dJ = config.get('save_dJ', False)
    verbose = config.get('verbose', False)

    # initialize stats collection
    algo_stats = {
        'action_dim': action_dim,
        'batch_size': batch_size,
        'eval_batch_size': eval_batch_size,
        'reward_mean':        jnp.empty(shape=(n_iters,)),
        'reward_std':         jnp.empty(shape=(n_iters,)),
        'reward_sterr':       jnp.empty(shape=(n_iters,)),
    }

    if save_dJ:
        algo_stats['dJ'] = jnp.empty(shape=(n_iters, batch_size, policy.n_params))

    # initialize optimizer
    opt_state = optimizer.init(policy.theta)

    # run REINFORCE
    for it in range(n_iters):
        subt0 = timer()

        key, algo_stats = evaluate_policy(key, it, algo_stats, eval_batch_size, policy.theta, policy, eval_model)

        key, dJ_hat, adv_estimator_state, batch_stats = compute_reinforce_dJ_hat_estimate(
            key, policy.theta,
            batch_size, epsilon, policy, train_model,
            adv_estimator, adv_estimator_state)

        updates, opt_state = optimizer.update(dJ_hat, opt_state)
        policy.theta = optax.apply_updates(policy.theta, updates)

        # update statistics and print out report for current iteration
        if verbose:
            print_reinforce_report(it, algo_stats, subt0, timer())

    algo_stats.update({
        'algorithm': 'REINFORCE',
        'n_iters': n_iters,
        'config': config,
    })

    return key, algo_stats
