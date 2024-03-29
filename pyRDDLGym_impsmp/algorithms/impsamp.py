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


@functools.partial(jax.jit, static_argnames=('policy', 'model', 'adv_estimator'))
def unnormalized_instr_density(key, theta, policy, model, adv_estimator, adv_estimator_state, states, actions):
    """Computes the value of the (unnormalized) instrumental probability density
    at the provided batch of states and actions
    """
    dpi = policy.diagonal_of_jacobian(key, theta, states, actions)
    key, Q_values, _ = adv_estimator.estimate(key, states, actions, None, adv_estimator_state)
    density = jnp.abs(dpi * Q_values[..., 0])
    return density

@functools.partial(jax.jit, static_argnames=('policy', 'model', 'adv_estimator'))
def unnormalized_log_instr_density(key, theta, policy, model, adv_estimator, adv_estimator_state, states, actions):
    density = unnormalized_instr_density(key, theta, policy, model, adv_estimator, adv_estimator_state, states, actions)
    return jnp.log(density)


@functools.partial(jax.jit, static_argnames=('n_shards', 'epsilon', 'policy', 'model'))
def compute_impsamp_dJ_hat_estimate(key, theta, s, a, cmlt_r, sampling_model_Q_vals, n_shards, epsilon, policy, model):
    """**** TODO: Update the docstring ****

        Computes an estimate of dJ^pi over a sample of rolled out trajectories

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
    dpi_B = jax.vmap(policy.diagonal_of_jacobian, (None, None, 0, 0), 0)
    dpi_BC = jax.vmap(dpi_B, (None, None, 2, 2), 2)

    key, subkey = jax.random.split(key)
    dpi = dpi_BC(subkey, theta, s, a)

    Q_estimate_ratio = cmlt_r / (sampling_model_Q_vals + epsilon)
    dpi_sgn = dpi / (jnp.abs(dpi) + epsilon)

    flat_dJ_hat = Q_estimate_ratio * dpi_sgn

    # Take averages over B (Sample batch axis) and C (Sampler chain axis),
    # leaving P (Policy parameter axis)
    flat_dJ_hat = jnp.mean(flat_dJ_hat, axis=(0, 2))

    # Put back into a dm-haiku tree shape in order to be useful
    # for updating the parameter vector theta
    dJ_hat = policy.unflatten_dJ(flat_dJ_hat)

    batch_stats = {}
    return key, dJ_hat, batch_stats

@functools.partial(jax.jit, static_argnames=('eval_n_shards', 'eval_batch_size', 'policy', 'model'))
def evaluate_policy(key, it, algo_stats, eval_n_shards, eval_batch_size, theta, policy, model):
    key, init_states = model.batch_generate_initial_state(key, (model.n_rollouts, model.state_dim))
    key, states, actions, rewards = model.rollout(key, init_states, theta)
    rewards = jnp.sum(rewards, axis=1) # sum rewards along the time axis
    key, subkey = jax.random.split(key)
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
def update_impsamp_stats(key, it, algo_stats, batch_stats, batch_size, save_dJ):
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

def print_impsamp_report(it, algo_stats, subt0, subt1):
    """Prints out the results for the current training iteration to console"""
    print(f'Iter {it} :: Importance Sampling :: Runtime={subt1-subt0}s')
    print(f'Untransformed parametrized policy [Mean, Diag(Cov)] =')
    print(algo_stats['policy_mean'][it])
    print(algo_stats['policy_cov'][it])
    print(f'Transformed action sample statistics [[Means], [StDevs]] =')
    print(algo_stats['transformed_policy_sample_mean'][it])
    print(algo_stats['transformed_policy_sample_cov'][it])
    print(algo_stats['dJ_hat_min'][it], '<= dJ <=', algo_stats['dJ_hat_max'][it], f':: dJ norm={algo_stats["dJ_hat_norm"][it]}')
    print('dJ covar:', algo_stats['dJ_covar_diag_min'][it], '<= Mean', algo_stats["dJ_covar_diag_mean"][it], ' <=', algo_stats["dJ_covar_diag_max"][it])
    print(f'Eval. reward={algo_stats["reward_mean"][it]:.3f} \u00B1 {algo_stats["reward_sterr"][it]:.3f}\n')


def impsamp(key, n_iters, config, bijector, policy, sampler, optimizer, models, adv_estimator, adv_estimator_state):
    """Runs the REINFORCE with Importance Sampling algorithm"""

    # Shorthands for commonly-used parameters
    # B = Sample batch
    # P = Number of policy parameters
    # C = Number of sampler chains
    B = config['batch_size']
    P = policy.n_params
    C = config['n_parallel_sampler_chains']
    n_samples = B * P * C

    action_dim = policy.action_dim
    n_parallel_sampler_chains = config['n_parallel_sampler_chains']
    eval_batch_size = config['eval_batch_size']

    sampling_model = models['sampling_model']
    train_model = models['train_model']
    eval_model = models['eval_model']

    horizon = train_model.horizon

    # compute the necessary number of shards
    n_shards = int(n_samples // train_model.n_rollouts)
    eval_n_shards = int(eval_batch_size // eval_model.n_rollouts)
    assert n_shards > 0, (
         '[impsamp] Please check that n_samples > train_model.n_rollouts.'
        f' n_samples={n_samples}, train_model.n_rollouts={train_model.n_rollouts}')
    assert eval_n_shards > 0, (
        '[impsamp] Please check that eval_batch_size > eval_model.n_rollouts.'
        f' eval_batch_size={eval_batch_size}, eval_model.n_rollouts={eval_model.n_rollouts}')

    epsilon = config.get('epsilon', 1e-12)
    save_dJ = config.get('save_dJ', False)
    verbose = config.get('verbose', False)

    # initialize stats collection
    algo_stats = {
        'action_dim': action_dim,
        'batch_size': B,
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
        algo_stats['dJ'] = jnp.empty(shape=(n_iters, B, policy.n_params))

    # initialize optimizer
    opt_state = optimizer.init(policy.theta)

    # initialize bijector
    unconstraining_bijector = [
        bijector
    ]

    # functions that compute a model step over B (Sample batch axis),
    # P (Policy parameter axis) and C (Sampler chain index axis)
    train_model_step_parallel_B = jax.vmap(train_model.step, (None, 0, 0, None, None), (0, 0))
    train_model_step_parallel_BP = jax.vmap(train_model_step_parallel_B, (None, 0, 0, None, None), (0, 0))
    train_model_step_parallel_BPC = jax.vmap(train_model_step_parallel_BP, (None, 0, 0, None, None), (0, 0))
    train_model_step_parallel_BPC = jax.jit(train_model_step_parallel_BPC)

    sampling_model_step_parallel_B = jax.vmap(sampling_model.step, (None, 0, 0, None, None), (0, 0))
    sampling_model_step_parallel_BP = jax.vmap(sampling_model_step_parallel_B, (None, 0, 0, None, None), (0, 0))
    sampling_model_step_parallel_BPC = jax.vmap(sampling_model_step_parallel_BP, (None, 0, 0, None, None), (0, 0))
    sampling_model_step_parallel_BPC = jax.jit(sampling_model_step_parallel_BPC)

    state_shape = (train_model.n_rollouts, policy.n_params, n_parallel_sampler_chains, train_model.state_dim)

    # run REINFORCE with Importance Sampling
    for it in range(n_iters):
        subt0 = timer()

        key, algo_stats = evaluate_policy(key, it, algo_stats, eval_n_shards, eval_batch_size, policy.theta, policy, eval_model)

        states_traj = []
        actions_traj = []
        rewards_traj = []
        sampling_model_rewards_traj = []

        # generate the initial states
        key, states = train_model.batch_generate_initial_state(key, state_shape)
        states_traj.append(states)

        # sample rollouts one step at a time using instrumental densities
        for t in range(horizon):
            key, subkey = jax.random.split(key)

            unnorm_log_density = functools.partial(
                unnormalized_log_instr_density,
                subkey,
                policy.theta,
                policy,
                sampling_model,
                adv_estimator,
                adv_estimator_state)

            # sample actions from instrumental density
            key = sampler.generate_step_size(key)
            key = sampler.prep(key,
                               it,
                               target_log_prob_fn=unnorm_log_density,
                               unconstraining_bijector=unconstraining_bijector)
            key = sampler.generate_initial_state(key, it, states)
            try:
                key, actions, accepted_matrix = sampler.sample(key, policy.theta, policy.n_params, states)
            except FloatingPointError as e:
                warnings.warn(f'[impsamp] Iteration {it}. Caught FloatingPointError exception during sampling')
                raise e

            key, subkey = jax.random.split(key)
            states, rewards = train_model_step_parallel_BPC(subkey, states, actions, policy.theta, True)
            _, sampling_rewards = sampling_model_step_parallel_BPC(subkey, states, actions, policy.theta, True)

            states_traj.append(states)
            actions_traj.append(actions)
            rewards_traj.append(rewards)
            sampling_model_rewards_traj.append(sampling_rewards)


        # compute the training model dJ estimate using initial actions and rewards from the generated trajectories
        s = states_traj[0]
        a = actions_traj[0]
        cmlt_rewards = jnp.sum(jnp.asarray(rewards_traj), axis=0)

        key, sampling_model_Q_vals, adv_estimator_state = adv_estimator.estimate(
            key, s, a, None, adv_estimator_state)

        key, dJ_hat, batch_stats = compute_impsamp_dJ_hat_estimate(
            key, policy.theta, s, a, cmlt_rewards, sampling_model_Q_vals,
            n_shards, epsilon, policy, train_model)

        # update the policy
        updates, opt_state = optimizer.update(dJ_hat, opt_state)
        policy.theta = optax.apply_updates(policy.theta, updates)

        # update the sampling Q-function using policy evaluation (minimize TD residual)
        for s, a, r, s_next, a_next in zip(states_traj, actions_traj, sampling_model_rewards_traj, states_traj[1:], actions_traj[1:]):
            key, adv_estimator_state = adv_estimator.policy_eval(
                key, s, a, r, s_next, a_next, adv_estimator_state)

        # update statistics and print out report for current iteration
        key, algo_stats = update_impsamp_stats(key, it, algo_stats, batch_stats, B, save_dJ)

        subt1 = timer()
        if verbose:
            print_impsamp_report(it, algo_stats, subt0, subt1)

    algo_stats.update({
        'algorithm': 'ImpSamp',
        'n_iters': n_iters,
        'config': config,
    })

    import matplotlib.pyplot as plt
    plt.plot(range(n_iters), algo_stats['reward_mean'])
    plt.savefig('plot.png')
    print('saved')

    return key, algo_stats
