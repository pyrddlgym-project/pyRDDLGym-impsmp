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

def flatten_dJ(dJ, batch_size, n_params):
    ip0 = 0
    flat_dJ = jnp.empty(shape=(batch_size, n_params))
    for leaf in jax.tree_util.tree_leaves(dJ):
        leaf = leaf.reshape(batch_size, n_params)
        ip1 = ip0 + leaf[0].flatten().shape[0]
        flat_dJ = flat_dJ.at[:, ip0:ip1].set(leaf)
        ip0 = ip1
    return flat_dJ

@functools.partial(
    jax.jit,
    static_argnames=('n_shards', 'batch_size', 'epsilon', 'policy', 'model', 'adv_estimator'))
def compute_reinforce_dJ_hat_estimate(key, theta, batch_size, n_shards, epsilon, policy, model, adv_estimator):
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
        n_shards: Number of shards to divide the batch into
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
    batch_stats = {}
    jacobian = jax.jacrev(policy.pdf, argnums=1)

    def compute_dJ_hat_summands_for_shard(key, _):
        """Compute dJ hat for the current sample, dividing the computation up
        into shards of size model.n_rollouts (this can be useful for fitting
        the computation into GPU VRAM, for example)
        """
        key, subkey = jax.random.split(key)
        states, actions, rewards = model.rollout(subkey, theta)
        advantages = adv_estimator.estimate(subkey, states, actions, rewards)
        pi_inv = 1 / (policy.pdf(subkey, theta, states, actions) + epsilon)
        dpi = jacobian(subkey, theta, states, actions)
        dlogpi = jax.tree_util.tree_map(lambda dpi_term: weighting_map(pi_inv, dpi_term), dpi)
        A_weighted_dlogpi = jax.tree_util.tree_map(lambda dlogpi_term: weighting_map(advantages, dlogpi_term), dlogpi)

        dJ_summands = jax.tree_util.tree_map(lambda A_weighted_dlogpi_term: jnp.sum(A_weighted_dlogpi_term, axis=1), A_weighted_dlogpi)
        return key, (actions, rewards, dJ_summands)

    key, (actions, rewards, dJ_summands) = jax.lax.scan(
        compute_dJ_hat_summands_for_shard,
        key, [None] * n_shards, length=n_shards)

    dJ_hat = jax.tree_util.tree_map(lambda term: jnp.mean(term, axis=(0,1)), dJ_summands)

    #batch_stats['dJ'] = flatten_dJ(dJ_summands, batch_size, policy.n_params)
    #batch_stats['actions'] = actions.reshape(batch_size, policy.action_dim)

    return key, dJ_hat, batch_stats

@functools.partial(jax.jit, static_argnames=('eval_n_shards', 'eval_batch_size', 'policy', 'model'))
def evaluate_policy(key, it, algo_stats, eval_n_shards, eval_batch_size, theta, policy, model):
    key, subkey = jax.random.split(key)
    states, actions, rewards = model.rollout(subkey, theta)
    rewards = jnp.sum(rewards, axis=1) # sum rewards along the time axis
    policy_mean, policy_cov = policy.apply(subkey, theta, states)

    algo_stats['reward_mean']  = algo_stats['reward_mean'].at[it].set(jnp.mean(rewards))
    algo_stats['reward_std']   = algo_stats['reward_std'].at[it].set(jnp.std(rewards))
    algo_stats['reward_sterr'] = algo_stats['reward_sterr'].at[it].set(algo_stats['reward_std'][it] / jnp.sqrt(model.n_rollouts))
    algo_stats['policy_mean']  = algo_stats['policy_mean'].at[it].set(jnp.mean(policy_mean, axis=(0,1)))
    algo_stats['policy_cov']   = algo_stats['policy_cov'].at[it].set(jnp.diag(jnp.mean(policy_cov, axis=(0,1))))
    return key, algo_stats

@functools.partial(
    jax.jit,
    static_argnames=(
        'batch_size',
        'save_dJ')
    )
def update_reinforce_stats(key, it, algo_stats, batch_stats, batch_size, save_dJ):
    """Updates the REINFORCE statistics using the statistics returned
    from the computation of dJ_hat for the current sample as well as
    the current policy mean/cov """
    #dJ_hat = jnp.mean(batch_stats['dJ'], axis=0)
    #dJ_covar = jnp.cov(batch_stats['dJ'], rowvar=False)

    if save_dJ:
        algo_stats['dJ']                 = algo_stats['dJ'].at[it].set(batch_stats['dJ'])

    #algo_stats['dJ_hat_max']         = algo_stats['dJ_hat_max'].at[it].set(jnp.max(dJ_hat))
    #algo_stats['dJ_hat_min']         = algo_stats['dJ_hat_min'].at[it].set(jnp.min(dJ_hat))
    #algo_stats['dJ_hat_norm']        = algo_stats['dJ_hat_norm'].at[it].set(jnp.linalg.norm(dJ_hat))
    #algo_stats['dJ_covar_max']       = algo_stats['dJ_covar_max'].at[it].set(jnp.max(dJ_covar))
    #algo_stats['dJ_covar_min']       = algo_stats['dJ_covar_min'].at[it].set(jnp.min(dJ_covar))
    #algo_stats['dJ_covar_diag_max']  = algo_stats['dJ_covar_diag_max'].at[it].set(jnp.max(jnp.diag(dJ_covar)))
    #algo_stats['dJ_covar_diag_min']  = algo_stats['dJ_covar_diag_min'].at[it].set(jnp.min(jnp.diag(dJ_covar)))
    #algo_stats['dJ_covar_diag_mean'] = algo_stats['dJ_covar_diag_mean'].at[it].set(jnp.mean(jnp.diag(dJ_covar)))
    #algo_stats['transformed_policy_sample_mean'] = algo_stats['transformed_policy_sample_mean'].at[it].set(jnp.squeeze(jnp.mean(batch_stats['actions'], axis=0)))
    #algo_stats['transformed_policy_sample_cov']  = algo_stats['transformed_policy_sample_cov'].at[it].set(jnp.squeeze(jnp.std(batch_stats['actions'], axis=0))**2)
    return key, algo_stats

def print_reinforce_report(it, algo_stats, subt0, subt1):
    """Prints out the results for the current REINFORCE iteration to console"""
    print(f'Iter {it} :: REINFORCE :: Runtime={subt1-subt0}s')
    print(f'Untransformed parametrized policy [Mean, Diag(Cov)] =')
    print(algo_stats['policy_mean'][it])
    print(algo_stats['policy_cov'][it])
    print(f'Transformed action sample statistics [[Means], [StDevs]] =')
    print(algo_stats['transformed_policy_sample_mean'][it])
    print(algo_stats['transformed_policy_sample_cov'][it])
    print(algo_stats['dJ_hat_min'][it], '<= dJ <=', algo_stats['dJ_hat_max'][it], f':: dJ norm={algo_stats["dJ_hat_norm"][it]}')
    print('dJ covar:', algo_stats['dJ_covar_diag_min'][it], '<= Mean', algo_stats["dJ_covar_diag_mean"][it], ' <=', algo_stats["dJ_covar_diag_max"][it])
    print(f'Eval. reward={algo_stats["reward_mean"][it]:.3f} \u00B1 {algo_stats["reward_sterr"][it]:.3f}\n')


def reinforce(key, n_iters, config, bijector, policy, sampler, optimizer, models, train_adv_estimator):
    """Runs the REINFORCE algorithm"""

    action_dim = policy.action_dim
    batch_size = config['batch_size']
    eval_batch_size = config['eval_batch_size']

    train_model = models['train_model']
    eval_model = models['eval_model']

    # compute the necessary number of shards
    n_shards = int(batch_size // train_model.n_rollouts)
    eval_n_shards = int(eval_batch_size // eval_model.n_rollouts)
    assert n_shards > 0, (
         '[reinforce] Please check that batch_size > train_model.n_rollouts.'
        f' batch_size={batch_size}, train_model.n_rollouts={train_model.n_rollouts}')
    assert eval_n_shards > 0, (
        '[reinforce] Please check that eval_batch_size > eval_model.n_rollouts.'
        f' eval_batch_size={eval_batch_size}, eval_model.n_rollouts={eval_model.n_rollouts}')

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
        'policy_mean':        jnp.empty(shape=(n_iters, action_dim)),
        'policy_cov':         jnp.empty(shape=(n_iters, action_dim)),
        'transformed_policy_sample_mean': jnp.empty(shape=(n_iters, action_dim)),
        'transformed_policy_sample_cov':  jnp.empty(shape=(n_iters, action_dim)),
        'dJ_hat_max':         jnp.empty(shape=(n_iters,)),
        'dJ_hat_min':         jnp.empty(shape=(n_iters,)),
        'dJ_hat_norm':        jnp.empty(shape=(n_iters,)),
        'dJ_covar_max':       jnp.empty(shape=(n_iters,)),
        'dJ_covar_min':       jnp.empty(shape=(n_iters,)),
        'dJ_covar_diag_max':  jnp.empty(shape=(n_iters,)),
        'dJ_covar_diag_min':  jnp.empty(shape=(n_iters,)),
        'dJ_covar_diag_mean': jnp.empty(shape=(n_iters,)),
    }

    if save_dJ:
        algo_stats['dJ'] = jnp.empty(shape=(n_iters, batch_size, policy.n_params))

    # initialize optimizer
    opt_state = optimizer.init(policy.theta)

    # run REINFORCE
    for it in range(n_iters):
        subt0 = timer()

        key, algo_stats = evaluate_policy(key, it, algo_stats, eval_n_shards, eval_batch_size, policy.theta, policy, eval_model)

        key, dJ_hat, batch_stats = compute_reinforce_dJ_hat_estimate(
            key, policy.theta,
            batch_size, n_shards, epsilon, policy, train_model, train_adv_estimator)

        updates, opt_state = optimizer.update(dJ_hat, opt_state)
        policy.theta = optax.apply_updates(policy.theta, updates)

        # update statistics and print out report for current iteration
        key, algo_stats = update_reinforce_stats(key, it, algo_stats, batch_stats, batch_size, save_dJ)
        if verbose:
            print_reinforce_report(it, algo_stats, subt0, timer())

    algo_stats.update({
        'algorithm': 'REINFORCE',
        'n_iters': n_iters,
        'config': config,
    })

    import matplotlib.pyplot as plt
    plt.plot(range(n_iters), algo_stats['reward_mean'])
    plt.savefig('plot.png')
    print('saved')

    return key, algo_stats
