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


@functools.partial(jax.jit, static_argnames=('policy', 'model'))
def unnormalized_instr_density_vector(key, policy, model, theta, init_states, actions):
    """The instrumental density for parameter i is defined as

        rho_i = | \tilde{R(tau_i)} * (\partial \pi / \partial \theta_i) (tau_i, theta) |

    Where \tilde R denotes the cumulative reward over trajectory tau_i in the
    sampling model, and pi denotes the parametrized policy with parameters
    theta. Please note that each parameter theta_i has its own sample trajectory
    (denoted by tau_i).

    Args:
        key: jax.random.PRNGKey
            The random generator state key
        policy:
            Class carrying static policy parameters
        model:
            Interface to the RDDL environment model
        theta: Dict
            Policy parameters (Dynamic, therefore passed separately from the policy class;
            the static and dynamic parameters are split to enable JIT compilation.)
            The policy number of parameters is denoted by n_params
        init_states: jnp.array shape=(n_params, state_dim)
        actions: jnp.array shape=(n_params, horizon, action_dim)
            For each parameter, the initial state and a trajectory of actions

    Returns:
        jnp.array shape=(n_params,)
            (rho_0(tau_0), rho_1(tau_1), ..., rho_N(tau_N))
        where N=n_params
    """
    key, *subkeys = jax.random.split(key, num=policy.n_params+1)
    subkeys = jnp.asarray(subkeys)
    _, states, actions, rewards = jax.vmap(
        model.evaluate_action_trajectory, (0, 0, 0), (0, 0, 0, 0))(subkeys, init_states, actions)

    dpi = policy.diagonal_of_jacobian_traj(key, theta, states, actions)
    adv = jnp.sum(rewards, axis=1) #advantage estimate
    density_vector = jnp.abs(adv * dpi)
    return density_vector

@functools.partial(jax.jit, static_argnames=('policy', 'model'))
def unnormalized_log_instr_density_vector(key, policy, model, theta, init_state, actions):
    """Please see the unnormalized_instr_density_vector docstring."""
    density_vector = unnormalized_instr_density_vector(key, policy, model, theta, init_state, actions)
    return jnp.log(density_vector)


@functools.partial(jax.jit, static_argnames=('epsilon', 'policy', 'light_model', 'train_model', 'Z_est_sample_size'))
def compute_impsamp_dJ_hat_estimate(
    key, epsilon, policy, light_model, train_model, Z_est_sample_size,
    theta, init_states, actions):
    """***** TODO:
      Update the docstring
      Add splitting by sign
     *******
    """
    B, P, T, A = actions.shape

    def _compute_unnorm_dJ_term(key, x):
        init_states, actions = x
        key, *subkeys = jax.random.split(key, num=(2 * P) + 1)
        light_subkeys, train_subkeys = jnp.asarray(subkeys[:P]), jnp.asarray(subkeys[P:])

        _, light_s, light_a, light_r = jax.vmap(
            light_model.evaluate_action_trajectory, (0, 0, 0), (0, 0, 0, 0))(light_subkeys, init_states, actions)
        light_adv_est = jnp.sum(light_r, axis=1)
        light_dpi = policy.diagonal_of_jacobian_traj(key, theta, light_s, light_a)

        _, train_s, train_a, train_r = jax.vmap(
            train_model.evaluate_action_trajectory, (0, 0, 0), (0, 0, 0, 0))(train_subkeys, init_states, actions)
        train_adv_est = jnp.sum(train_r, axis=1)
        train_dpi = policy.diagonal_of_jacobian_traj(key, theta, train_s, train_a)

        dJ_term = train_adv_est * train_dpi / (jnp.abs(light_adv_est * light_dpi) + epsilon)
        return key, dJ_term

    key, unnorm_dJ_terms = jax.lax.scan(_compute_unnorm_dJ_term, init=key, xs=(init_states, actions))
    flat_unnorm_dJ_hat = jnp.mean(unnorm_dJ_terms, axis=0)

    # estimate the partition functions Z_i for each instrumental density
    Z_init_states = init_states.reshape(B * P, -1)
    Z_init_states = jnp.repeat(Z_init_states, np.ceil(Z_est_sample_size / (B * P)).astype(np.uint32), axis=0)
    def _compute_Z_estimate_term(key, x):
        init_state = x

        key, states, actions, rewards = light_model.rollout_parametrized_policy(key, init_state, policy.theta)

        key, *subkeys = jax.random.split(key, num=3)
        pi = policy.pdf_traj(subkeys[0], policy.theta, states[jnp.newaxis, ...], actions[jnp.newaxis, ...])

        adv_est = jnp.sum(rewards, axis=0)
        states_PTS = jnp.stack([states] * P)
        actions_PTA = jnp.stack([actions] * P)
        dpi = policy.diagonal_of_jacobian_traj(subkeys[1], policy.theta, states_PTS, actions_PTA)
        rho_vec = jnp.abs(adv_est * dpi)

        Z_term = rho_vec / pi
        return key, Z_term

    key, Z_terms = jax.lax.scan(_compute_Z_estimate_term, init=key, xs=Z_init_states)
    Zs = jnp.mean(Z_terms, axis=0)

    # normalize the dJ hat estimate using the partition functions,
    # and convert format back into a dm-haiku tree shape in order
    # to be useful for updating the parameter vector theta
    flat_dJ_hat = flat_unnorm_dJ_hat / (Zs + epsilon)
    dJ_hat = policy.unflatten_dJ(flat_dJ_hat)

    batch_stats = {}
    return key, dJ_hat, batch_stats


@functools.partial(jax.jit, static_argnames=('eval_batch_size', 'policy', 'model'))
def evaluate_policy(key, it, algo_stats, eval_batch_size, theta, policy, model):
    key, init_states = model.batch_generate_initial_state(key, (eval_batch_size, model.state_dim))
    key, *subkeys = jax.random.split(key, num=eval_batch_size+1)
    subkeys = jnp.asarray(subkeys)
    _, states, actions, rewards = jax.vmap(
        model.rollout_parametrized_policy, (0, 0, None), (0, 0, 0, 0))(subkeys, init_states, theta)
    rewards = jnp.sum(rewards, axis=1) # sum rewards along the time axis
    key, subkey = jax.random.split(key)
    policy_mean, policy_cov = policy.apply(subkey, theta, states)

    algo_stats['reward_mean']  = algo_stats['reward_mean'].at[it].set(jnp.mean(rewards))
    algo_stats['reward_std']   = algo_stats['reward_std'].at[it].set(jnp.std(rewards))
    algo_stats['reward_sterr'] = algo_stats['reward_sterr'].at[it].set(algo_stats['reward_std'][it] / jnp.sqrt(eval_batch_size))
    algo_stats['policy_mean']  = algo_stats['policy_mean'].at[it].set(jnp.mean(policy_mean, axis=(0,1)))
    algo_stats['policy_cov']   = algo_stats['policy_cov'].at[it].set(jnp.diag(jnp.mean(policy_cov, axis=(0,1))))
    return key, algo_stats

@functools.partial(jax.jit, static_argnames=('batch_size', 'save_dJ'))
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

    sampling_model = models['sampling_model']
    train_model = models['train_model']
    eval_model = models['eval_model']

    # Shorthands for common parameters
    # B = Sample batch size
    # C = Number of sampler chains
    # P = Number of policy parameters
    # T = Horizon
    # A = Action space dimension
    # S = State space dimension
    B = config['batch_size']
    C = config['n_parallel_sampler_chains']
    P = policy.n_params
    T = train_model.horizon
    A = policy.action_dim
    S = policy.state_dim

    eval_batch_size = config['eval_batch_size']

    Z_estimator_config = config['Z_estimator_config']

    epsilon = config.get('epsilon', 1e-12)
    save_dJ = config.get('save_dJ', False)
    verbose = config.get('verbose', False)

    # initialize stats collection
    algo_stats = {
        'action_dim': A,
        'state_dim': S,
        'batch_size': B,
        'eval_batch_size': eval_batch_size,
        'reward_mean':        jnp.empty(shape=(n_iters,)),
        'reward_std':         jnp.empty(shape=(n_iters,)),
        'reward_sterr':       jnp.empty(shape=(n_iters,)),
        'policy_mean':        jnp.empty(shape=(n_iters, A)),
        'policy_cov':         jnp.empty(shape=(n_iters, A)),
        'transformed_policy_sample_mean': jnp.empty(shape=(n_iters, A)),
        'transformed_policy_sample_cov':  jnp.empty(shape=(n_iters, A)),
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
        algo_stats['dJ'] = jnp.empty(shape=(n_iters, B, P))

    # initialize optimizer
    opt_state = optimizer.init(policy.theta)

    # initialize bijector
    unconstraining_bijector = [
        bijector
    ]

    init_state_shape = (B, C, P, S)

    # run REINFORCE with Importance Sampling
    for it in range(n_iters):
        subt0 = timer()

        key, algo_stats = evaluate_policy(key, it, algo_stats, eval_batch_size, policy.theta, policy, eval_model)

        # generate the initial states
        key, init_states = train_model.batch_generate_initial_state(key, init_state_shape)

        # sample action trajectories from the instrumental density
        key, subkey = jax.random.split(key)
        unnorm_log_density_vector = functools.partial(
            unnormalized_log_instr_density_vector,
            subkey,
            policy,
            sampling_model,
            policy.theta)

        key = sampler.generate_step_size(key)
        key = sampler.prep(key,
                           it,
                           target_log_prob_fn=unnorm_log_density_vector,
                           unconstraining_bijector=unconstraining_bijector)
        key = sampler.generate_initial_state(key, it, init_states)
        try:
            key, actions, accepted_matrix = sampler.sample(key, policy.theta, init_states)
        except FloatingPointError as e:
            warnings.warn(f'[impsamp] Iteration {it}. Caught FloatingPointError exception during sampling')
            break

        init_states = init_states[:, 0] # remove the C axis.TODO: Figure out what the right thing to do is
        key, dJ_hat, batch_stats = compute_impsamp_dJ_hat_estimate(
            key, epsilon, policy, sampling_model, train_model, Z_estimator_config['n_samples'],
            policy.theta, init_states, actions)

        # update the policy
        updates, opt_state = optimizer.update(dJ_hat, opt_state)
        policy.theta = optax.apply_updates(policy.theta, updates)

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

    return key, algo_stats
