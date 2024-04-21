"""Implementation of REINFORCE with Importance Sampling.

NOTE: This particular implementation does not split the densities into positive
and negative parts. It also does not do subsampling. This implementation is
mostly useful as a working simplest version of the algorithm.

For better-performing implementations, please see

             "impsamp_signed_density"
                       or
             "impsamp_split_by_sign"
"""
import jax
import jax.numpy as jnp
import optax
import numpy as np
import functools
from time import perf_counter as timer



@functools.partial(jax.jit, static_argnames=('policy', 'model'))
def unnormalized_instr_density_vector(key, policy, model, theta, init_model_states, actions):
    """The instrumental density for parameter i is defined as

    rho_i = | \tilde{R(tau_i)} * (\partial \pi / \partial \theta_i) (tau_i, theta) |

    Where \tilde R denotes the cumulative reward over trajectory tau_i
    in the sampling model, and pi denotes the parametrized policy with
    parameters theta. Please note that each parameter theta_i has its
    own sample trajectory (denoted by tau_i).

    Args:
        key: jax.random.PRNGKey
            The random generator state key
        policy:
            Class carrying static policy parameters
        model:
            Interface to the RDDL environment model
        theta: Dict
            Policy parameters
            (Dynamic, therefore passed separately from the policy class;
            the static and dynamic parameters are split to enable JIT
            compilation.)
            The policy number of parameters is denoted by n_params
        init_model_states: jnp.array shape=(n_params, state_dim)
        actions: jnp.array shape=(n_params, horizon, action_dim)
            For each parameter, the initial state and a trajectory of
            actions

    Returns:
        jnp.array shape=(n_params,)
            (rho_0(tau_0), rho_1(tau_1), ..., rho_N(tau_N))
        where N=n_params
    """
    # evaluate tau_i for parameter i in batch in i
    key, states, actions, rewards = model.evaluate_action_trajectory_batched(
        key, init_model_states, actions)
    dpi = policy.diagonal_of_jacobian_traj(key, theta, states, actions)
    adv = jnp.sum(rewards, axis=1) #advantage estimate
    density_vector = jnp.abs(adv * dpi)
    return density_vector

@functools.partial(jax.jit, static_argnames=('policy', 'model'))
def unnormalized_log_instr_density_vector(key, policy, model, theta, init_state, actions):
    """Please see the unnormalized_instr_density_vector docstring."""
    density_vector = unnormalized_instr_density_vector(key, policy, model, theta, init_state, actions)
    return jnp.log(density_vector)


@functools.partial(jax.jit, static_argnames=('epsilon', 'policy', 'sampling_model', 'train_model', 'Z_est_sample_size'))
def compute_impsamp_dJ_hat_estimate(
    key, epsilon, policy, sampling_model, train_model, Z_est_sample_size,
    theta, init_model_states, actions):
    """Given a sample of initial states and actions, computes the Importance Sampling
    estimate of dJ using the formula

        dJ/dtheta_i ~ 1/|B| \sum_j r(tau_j) / \tilde |r(tau_j)| * sign(dpi/dthetai)

    Args:
        key: jax.random.PRNGKey
            The random generator state key
        epsilon: Float
            Small number to avoid division by zero
        policy:
            Class carrying static policy parameters
        sampling_model:
        train_model:
            Interfaces to the light and training environment model
        Z_est_sample_size: Int
            Number of samples to draw for estimating the partition functions
        theta: Dict
            Policy parameters
            (Dynamic, therefore passed separately from the policy class;
            the static and dynamic parameters are split to enable JIT
            compilation.)
            The policy number of parameters is denoted by n_params
        init_model_states: jnp.array shape=(batch_size, n_chains, n_params, state_dim)
        actions: jnp.array shape=(batch_size, n_chains, n_params, horizon, action_dim)
            For each sample, chain, parameter, the initial state and a trajectory
            of actions

    Returns:
        key: jax.random.PRNGKey
            Mutated random generator state
        dJ_hat: Dict
            PyTree of dJ/dtheta_i. Same PyTree structure as theta.
        batch_stats: Dict
            Collected statistics
    """
    # Please see the "impsamp" function below for the meaning
    # of the parameter shorthands
    B, P, T, A = actions.shape

    def _compute_unnorm_dJ_term(key, x):
        init_model_states, actions = x

        key, light_s, light_a, light_r = sampling_model.evaluate_action_trajectory_batched(
            key, init_model_states, actions)
        light_adv_est = jnp.sum(light_r, axis=1)
        light_dpi = policy.diagonal_of_jacobian_traj(key, theta, light_s, light_a)

        key, train_s, train_a, train_r = train_model.evaluate_action_trajectory_batched(
            key, init_model_states, actions)
        train_adv_est = jnp.sum(train_r, axis=1)
        train_dpi = policy.diagonal_of_jacobian_traj(key, theta, train_s, train_a)

        dJ_term = train_adv_est * train_dpi / (jnp.abs(light_adv_est * light_dpi) + epsilon)
        return key, dJ_term

    key, unnorm_dJ_terms = jax.lax.scan(_compute_unnorm_dJ_term, init=key, xs=(init_model_states, actions))
    flat_unnorm_dJ_hat = jnp.mean(unnorm_dJ_terms, axis=0)

    # estimate the partition functions Z_i for each instrumental density
    Z_init_states = init_model_states.reshape(B * P, -1)
    Z_init_states = jnp.repeat(Z_init_states, np.ceil(Z_est_sample_size / (B * P)).astype(np.uint32), axis=0)
    def _compute_Z_estimate_term(key, x):
        init_state = x

        key, states, actions, rewards = sampling_model.rollout_parametrized_policy(key, init_state, policy.theta)

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

    # normalize the dJ hat estimate using the partition functions
    flat_dJ_hat = flat_unnorm_dJ_hat / (Zs + epsilon)

    # convert the format back into a dm-haiku PyTree in order
    # to be useful for updating the parameter vector theta
    dJ_hat = policy.unflatten_dJ(flat_dJ_hat)

    batch_stats = {}
    return key, dJ_hat, batch_stats


@functools.partial(jax.jit, static_argnames=('eval_batch_size', 'policy', 'model'))
def evaluate_policy(key, it, algo_stats, eval_batch_size, theta, policy, model):
    key, init_model_states = model.generate_initial_state_batched(key, (eval_batch_size,))
    key, states, actions, rewards = model.rollout_parametrized_policy_batched(
        key, init_model_states, theta)
    rewards = jnp.sum(rewards, axis=1) # sum rewards along the time axis

    algo_stats['reward_mean']  = algo_stats['reward_mean'].at[it].set(jnp.mean(rewards))
    algo_stats['reward_std']   = algo_stats['reward_std'].at[it].set(jnp.std(rewards))
    algo_stats['reward_sterr'] = algo_stats['reward_sterr'].at[it].set(
        algo_stats['reward_std'][it] / jnp.sqrt(eval_batch_size))
    return key, algo_stats

def print_impsamp_report(it, algo_stats, sampler, subt0, subt1):
    """Prints out the results for the current training iteration to console"""
    print(f'Iter {it} :: Importance Sampling :: Runtime={subt1-subt0}s')
    sampler.print_report(it)
    print(f'Eval. reward={algo_stats["reward_mean"][it]:.3f} \u00B1 {algo_stats["reward_sterr"][it]:.3f}\n')


def impsamp(key, n_iters, checkpoint_freq,
            config, bijector, policy, sampler, optimizer, models, adv_estimator, adv_estimator_state):
    """Runs the REINFORCE with Importance Sampling algorithm"""

    sampling_model = models['sampling_model']
    train_model = models['train_model']
    eval_model = models.get('eval_model', train_model)

    # Shorthands for common parameters
    # B = Sample batch size
    # P = Number of policy parameters
    # T = Horizon
    # S = State space dimension
    # A = Action space dimension
    B = config['batch_size']
    P = policy.n_params
    T = train_model.horizon
    S = policy.state_dim
    A = policy.action_dim

    eval_batch_size = config['eval_batch_size']

    Z_estimator_config = config['Z_estimator_config']

    epsilon = config.get('epsilon', 1e-12)
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
    }

    # initialize optimizer
    opt_state = optimizer.init(policy.theta)

    # initialize bijector
    unconstraining_bijector = [
        bijector
    ]

    init_model_state_shape = (B, P, S)
    actions = None

    # run REINFORCE with Importance Sampling
    for it in range(n_iters):
        subt0 = timer()

        key, algo_stats = evaluate_policy(key, it, algo_stats, eval_batch_size, policy.theta, policy, eval_model)

        # generate the initial states
        key, init_model_states = train_model.generate_initial_state_batched(key, init_model_state_shape[:-1])

        # sample action trajectories from the instrumental density
        key, subkey = jax.random.split(key)
        unnorm_log_density_vector = functools.partial(
            unnormalized_log_instr_density_vector,
            subkey,
            policy,
            sampling_model,
            policy.theta)

        key, sampler_step_size = sampler.generate_step_size(key)
        key, sampler_init_states = sampler.generate_initial_state(key, it, init_model_states, actions)
        key = sampler.prep(key,
                           it,
                           target_log_prob_fn=unnorm_log_density_vector,
                           unconstraining_bijector=unconstraining_bijector,
                           step_size=sampler_step_size)

        try:
            key, (init_model_states, actions), accepted_matrix = sampler.sample(
                key, policy.theta, init_model_states, sampler_step_size, sampler_init_states)
        except FloatingPointError as e:
            warnings.warn(f'[impsamp] Iteration {it}. Caught FloatingPointError exception during sampling')
            break

        key, dJ_hat, batch_stats = compute_impsamp_dJ_hat_estimate(
            key, epsilon, policy, sampling_model, train_model, Z_estimator_config['n_samples'],
            policy.theta, init_model_states, actions)

        # update the policy
        updates, opt_state = optimizer.update(dJ_hat, opt_state)
        policy.theta = optax.apply_updates(policy.theta, updates)

        subt1 = timer()
        if verbose:
            print_impsamp_report(it, algo_stats, sampler, subt0, subt1)

    algo_stats.update({
        'algorithm': 'ImpSamp',
        'n_iters': n_iters,
        'config': config,
        'policy_theta': policy.theta
    })

    #TODO: Remove, this is temporary
    import matplotlib.pyplot as plt
    plt.plot(range(n_iters), algo_stats['reward_mean'])
    plt.savefig('/tmp/impsamp_plot.png')
    #DONE

    return key, algo_stats
