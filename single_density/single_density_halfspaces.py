import jax
import numpy as np
import jax.numpy as jnp
import jax.scipy.stats as st

import optax
import blackjax

from functools import partial
import matplotlib.pyplot as plt

key_rng = jax.random.key(42)


# "Sum-of-half-spaces" environment
# A hyperplane in R^d may be described by
#     n . (x - b) = 0
# where n - normal vector, b - translation vector, x - variable, . denotes dot product in R^d
#
# Sum of half spaces reward: r(x) = sum_i sgn(n_i . (x - b_i))
#
# where sgn: R -> R, sgn(x) = -1 if x < 0, 0 if x == 0, +1 if x > 0
#
# The idea is to generate a collection of hyperplanes {H0, H1, ..., Hn}. Points to one side of
# a hyperplane Hi have reward +1, and on the other side -1 (points ON the hyperplane have reward 0).
# Denote the reward of a point x with respect to hyperplane Hi b Ri. Then the reward of the point x
# is sum_i R_i.
#
# Smoothed reward:           r_sm(x) = sum_i tanh((n_i . (x - b_i)) / env_smoothing_weight)

env_dim = 2
env_n_summands = 10
env_smoothing_weight = 1.0

if env_dim == 2 and env_n_summands == 10:
    # Definition of "Instance 0" of the Dim=2, Summands=10 RDDL
    #
    # This special case is hard-coded because there is a reference
    # visualization of the reward function at
    #
    #     http://iliasmirnov.com/ISPG/2023_11_14_sum_of_half_spaces/dim2_sum10/img0.png
    #
    # normal vectors
    env_n = jnp.array(
        [[-0.6498891485751617,  0.2919621952260027],
         [0.6786194558029103,  -0.6731776871798324],
         [0.5186441418263966,   0.7688779421225117],
         [0.9046459743664846,  -0.19908859352762293],
         [1.3787654118345662,  -0.3343831688835496],
         [0.15299290103060892, -0.4208794744431309],
         [0.9567609279084787,   0.14615876772409975],
         [0.7434759943923458,   1.5750369384499119],
         [0.07260930861610973, -0.14729616541211463],
         [0.36572742545177667, -0.6180098661319744]])

    # translation vectors
    env_b = jnp.array(
        [[0.9221341845334698, -1.5875599610512214],
         [-1.5407858463031727, -1.0830704454939821],
         [-0.9714550617144435, -6.3226717384079],
         [0.4627735557073612,  -0.9854401248393589],
         [-1.5657090993934906,  2.054459931192971],
         [-2.3975399618755096,  0.9660592961761412],
         [-1.7790604424242211, -0.7348306237622735],
         [0.5942818372802966,  -0.9727294659982499],
         [2.9721688297435516,  -0.11274469824590543],
         [-0.13144172918627878, 1.6276454625934715]])
else:
    key_rng, key_n, key_b = jax.random.split(key_rng, num=3)
    shape = (env_n_summands, env_dim)
    env_n = jax.random.normal(key_n, shape)
    env_b = 2 * jax.random.normal(key_b, shape)

env = (env_n, env_b)

def r(a, env):
    env_n, env_b = env
    x = jnp.einsum('ij,ij->i', env_n, (a - env_b))
    x = jnp.sign(x)
    return jnp.sum(x)

def r_smoothed(a, env):
    env_n, env_b = env
    x = jnp.einsum('ij,ij->i', env_n, (a - env_b))
    x = jnp.tanh(x / env_smoothing_weight)
    return jnp.sum(x)


# assume that the policy is distributed as a multivariate normal with parameters
# mu = (mu_x1, mu_x2, ..., mu_xd) and sigma = ((sigma_x1x1, 0, ..., 0), ..., (0, 0, ..., sigma_xdxd))

def initialize_theta_params(key_rng):
    key_rng, key_mu, key_sigma = jax.random.split(key_rng, num=3)
    params_mu    = 5  * jax.random.normal(key_mu, shape=(env_dim,))
    params_sigma = 10 * jax.random.normal(key_sigma, shape=(env_dim,))
    return key_rng, jnp.concatenate((params_mu, params_sigma))

key_rng, pi_theta_P0 = initialize_theta_params(key_rng)


def pi_sample(key, pi_theta_P, shape):
    # shape should end in (..., env_dim)
    x = jax.random.normal(key, shape)
    # make sure the variance is positive
    var = jax.nn.softplus(pi_theta_P[env_dim:])
    return pi_theta_P[0:env_dim] + jnp.sqrt(var) * x

def pi_pdf(x, pi_theta_P):
    # shape of x should end in (..., env_dim)
    var = jax.nn.softplus(pi_theta_P[env_dim:])
    cov = jnp.diag(var)
    return st.multivariate_normal.pdf(x, pi_theta_P[:env_dim], cov)

grad_pi_pdf = jax.jit(jax.grad(pi_pdf, argnums=1))


@partial(jax.jit, static_argnames=['optimizer', 'batch_size'])
def reinforce_step(key, pi_theta_P, optimizer, opt_state, batch_size, lr):
    key, key_sample, key_r = jax.random.split(key, num=3)
    a_batch_BA  = pi_sample(key_sample, pi_theta_P, shape=(batch_size, env_dim))
    pdf_B       = pi_pdf(a_batch_BA, pi_theta_P)
    grad_pdf_BP = jnp.apply_along_axis(grad_pi_pdf, 1, a_batch_BA, pi_theta_P)
    rewards_B   = jnp.apply_along_axis(r_smoothed, 1, a_batch_BA, env)

    reinforce_summands_BP = (grad_pdf_BP / (pdf_B[...,jnp.newaxis] + 1e-8)) * rewards_B[...,jnp.newaxis]
    reinforce_update_P    = jnp.mean(reinforce_summands_BP, axis=0)

    updates_P, opt_state = optimizer.update(reinforce_update_P, opt_state)
    pi_theta_P = optax.apply_updates(pi_theta_P, updates_P)
    #pi_theta_P = pi_theta_P - lr * reinforce_update_P

    # Estimate rewards on a controlled number of samples
    stats_rewards_est_a_BA = pi_sample(key_r, pi_theta_P, shape=(1024, env_dim))
    stats_rewards_est_B = jnp.apply_along_axis(r_smoothed, 1, stats_rewards_est_a_BA, env)
    stats_rewards_est = jnp.mean(stats_rewards_est_B)

    stats_step = [stats_rewards_est]

    return key, pi_theta_P, opt_state, stats_step


def reinforce(key, pi_theta_P, optimizer, opt_state, n_iter, batch_size, lr):
    stats_reward = []
    stats_params = []

    for _ in range(n_iter):
        key, pi_theta_P, opt_state, stats_step = reinforce_step(key, pi_theta_P, optimizer, opt_state, batch_size, lr)
        stats_reward.append(stats_step[0])
        stats_pi_theta = pi_theta_P
        # record the interpretable variance, not the parameters
        stats_pi_theta_P = stats_pi_theta.at[env_dim:].set(jax.nn.softplus(stats_pi_theta[env_dim:]))
        stats_params.append(stats_pi_theta_P)

    stats_run = (stats_reward, stats_params)

    return key, pi_theta_P, opt_state, stats_run


M = 1
def rho_pdf_unnorm(a_A, pi_theta_P):
    reward = r_smoothed(a_A, env)
    pdf = pi_pdf(a_A, pi_theta_P)
    grad_pdf_P = grad_pi_pdf(a_A, pi_theta_P)
    norm_dlogpi = M * jnp.linalg.norm(grad_pdf_P / M, ord=2) / (pdf + 1e-8)
    rho_val = jnp.abs(reward) * norm_dlogpi
    return rho_val

def log_rho_pdf_unnorm(a_A, pi_theta_P):
    reward = r_smoothed(a_A, env)
    pdf = pi_pdf(a_A, pi_theta_P)
    grad_pdf_P = grad_pi_pdf(a_A, pi_theta_P)
    norm_dlogpi = M * jnp.linalg.norm(grad_pdf_P / M, ord=2) / (pdf + 1e-8)
    log_rho_val = jnp.log(jnp.abs(reward)) + jnp.log(norm_dlogpi)
    return log_rho_val


rej_sampling_M = 1e3

#@partial(jax.jit, static_argnames=('optimizer', 'proposal_size', 'batch_size'))
def ISPG_step_rej(it, key, pi_theta_P, optimizer, opt_state, last_sample, proposal_size, batch_size):
    key, key_u, key_sample, key_infer, key_r, key_Z, key_vre, key_bias = jax.random.split(key, num=8)

    u_S = jax.random.uniform(key_u, shape=(proposal_size,))
    proposed_SA = pi_sample(key_sample, pi_theta_P, (proposal_size, env_dim))
    pi_S  = pi_pdf(proposed_SA, pi_theta_P)
    rho_S = jnp.apply_along_axis(rho_pdf_unnorm, 1, proposed_SA, pi_theta_P)

    accepted_S = u_S < (rho_S / (rej_sampling_M * pi_S))

    accepted_ind_S = jnp.argwhere(accepted_S, size=proposal_size, fill_value=-1)[..., 0]
    accepted_ind_B = accepted_ind_S[:batch_size]

    a_batch_BA = jnp.take(proposed_SA, accepted_ind_B, axis=0)

    grad_pdf_BP     = jnp.apply_along_axis(grad_pi_pdf, 1, a_batch_BA, pi_theta_P)
    grad_pdf_norm_B = jnp.linalg.norm(grad_pdf_BP, ord=2, axis=1)
    rewards_B       = jnp.apply_along_axis(r_smoothed, 1, a_batch_BA, env)

    scalars_B = rewards_B / (jnp.abs(rewards_B) * grad_pdf_norm_B + 1e-8)
    ISPG_summands_unnorm_BP = scalars_B[:, jnp.newaxis] * grad_pdf_BP

    # compute estimate of the partition function Z_rho
    # by sampling from pi and using importance sampling again
    Z_est_a_BA = pi_sample(key_Z, pi_theta_P, shape=(4096, env_dim))
    Z_est_pi_pdf_B = pi_pdf(Z_est_a_BA, pi_theta_P)
    Z_est_rho_pdf_unnorm_B = jnp.apply_along_axis(rho_pdf_unnorm, 1, Z_est_a_BA, pi_theta_P)

    Z_est_B = Z_est_pi_pdf_B / (Z_est_rho_pdf_unnorm_B + 1e-8)
    Z_est = jnp.mean(Z_est_B)

    ISPG_update_P = Z_est * jnp.mean(ISPG_summands_unnorm_BP, axis=0)

    # estimate the variance reduction
    ISPG_var = Z_est * Z_est * jnp.trace(jnp.cov(ISPG_summands_unnorm_BP, rowvar=False))

    a_batch_vre_BA = pi_sample(key_vre, pi_theta_P, shape=a_batch_BA.shape)
    pdf_B          = pi_pdf(a_batch_vre_BA, pi_theta_P)
    grad_pdf_BP    = jnp.apply_along_axis(grad_pi_pdf, 1, a_batch_vre_BA, pi_theta_P)
    rewards_B      = jnp.apply_along_axis(r_smoothed, 1, a_batch_vre_BA, env)
    reinforce_summands_BP = (grad_pdf_BP / (pdf_B[...,jnp.newaxis] + 1e-8)) * rewards_B[...,jnp.newaxis]
    REINFORCE_var = jnp.trace(jnp.cov(reinforce_summands_BP, rowvar=False))

    mag = jnp.linalg.norm(ISPG_update_P, ord=2)
    vre = REINFORCE_var / ISPG_var

    # estimate the bias
    a_batch_bias_BA = pi_sample(key_bias, pi_theta_P, shape=(16384, env_dim))
    pdf_B           = pi_pdf(a_batch_bias_BA, pi_theta_P)
    grad_pdf_BP     = jnp.apply_along_axis(grad_pi_pdf, 1, a_batch_bias_BA, pi_theta_P)
    rewards_B       = jnp.apply_along_axis(r_smoothed, 1, a_batch_bias_BA, env)
    reinforce_summands_BP = (grad_pdf_BP / (pdf_B[...,jnp.newaxis] + 1e-8)) * rewards_B[...,jnp.newaxis]
    REINFORCE_bias_est_P = jnp.mean(reinforce_summands_BP, axis=0)

    bias_est_P = ISPG_update_P - REINFORCE_bias_est_P
    bias_est   = jnp.max(jnp.abs(bias_est_P))

    # update the policy pi parameters
    updates_P, opt_state = optimizer.update(ISPG_update_P, opt_state)
    pi_theta_P = optax.apply_updates(pi_theta_P, updates_P)
    #    pi_theta_P = pi_theta_P - lr * ISPG_update_P

    stats_rewards_est_a_BA = pi_sample(key_r, pi_theta_P, shape=(1024, env_dim))
    stats_rewards_est_B = jnp.apply_along_axis(r_smoothed, 1, stats_rewards_est_a_BA, env)
    stats_rewards_est = jnp.mean(stats_rewards_est_B)
    acc_rate = jnp.sum(accepted_S) / batch_size
    stats_step = [stats_rewards_est, Z_est, vre, mag, acc_rate, bias_est]

    return key, pi_theta_P, opt_state, a_batch_BA, stats_step

def ISPG_rej(key, pi_theta_P, optimizer, opt_state, n_iter, proposal_size, batch_size, lr):
    stats_reward = []
    stats_policy = []
    stats_Z_est  = []
    stats_vre    = []
    stats_mag    = []
    stats_acc    = []
    stats_bias   = []

    key, key_sample = jax.random.split(key)
    sample = pi_sample(key_sample, pi_theta_P, (batch_size, env_dim))

    for it in range(n_iter):
        if it > 0:
            print('it={},  rew={:>2.3f},    variance reduction ~ {:>6.0f},      bias ~ {:>6.4f},    lr*mag={:>6.4f},    acc={:>3.2f}'.format(it,
                stats_reward[-1],
                stats_vre[-1],
                stats_bias[-1],
                lr*stats_mag[-1],
                stats_acc[-1]))


        key, pi_theta_P, opt_state, sample, stats_step = ISPG_step_rej(it, key, pi_theta_P, optimizer, opt_state, sample, proposal_size, batch_size)

        stats_reward.append(stats_step[0])
        stats_pi_theta_P = pi_theta_P
        # record the interpretable variance, not the parameters
        stats_pi_theta_P = stats_pi_theta_P.at[env_dim:].set(jax.nn.softplus(stats_pi_theta_P[env_dim:]))
        stats_policy.append(stats_pi_theta_P)
        stats_Z_est.append(stats_step[1])
        stats_vre.append(stats_step[2])
        stats_mag.append(stats_step[3])
        stats_acc.append(stats_step[4])
        stats_bias.append(stats_step[5])
        if stats_acc[-1] < 1.0:
            print('[ISPG_rej] Insufficiently many accepted samples -- early exit')
            break
        if jnp.isnan(stats_step[1]):
            print(f'[ISPG_rej] NaN on iteration {it} --- early exit')
            break

    stats_run = (stats_reward, stats_policy, stats_Z_est, stats_vre, stats_mag)
    return key, pi_theta_P, opt_state, stats_run



@partial(jax.jit, static_argnames=('optimizer', 'batch_size'))
def ISPG_step_hmc(it, key, pi_theta_P, optimizer, opt_state, last_sample, batch_size):
    key, key_sample, key_infer, key_r, key_Z, key_vre = jax.random.split(key, num=6)

    hmc_target_log_prob = lambda a: log_rho_pdf_unnorm(a, pi_theta_P)
    hmc_inv_mass_matrix = np.ones(env_dim) * 0.1
    hmc_step_size = 0.1
    hmc_num_leapfrog_steps = 32
    hmc_num_burnin_steps = 32

    hmc = blackjax.hmc(hmc_target_log_prob, hmc_step_size, hmc_inv_mass_matrix, hmc_num_leapfrog_steps)

    def inference_loop(rng_key, kernel, initial_state, num_samples, num_chains):
        @jax.jit
        def one_step(states, rng_key):
            keys = jax.random.split(rng_key, num_chains)
            states, _ = jax.vmap(kernel)(keys, states)
            return states, states

        keys = jax.random.split(rng_key, num_samples)
        _, states = jax.lax.scan(one_step, initial_state, keys)

        return states

    #initial_pos =  10 * jax.random.normal(key_sample, (batch_size, env_dim))
    initial_pos = pi_sample(key_sample, pi_theta_P, (batch_size, env_dim))
    initial_state = jax.vmap(hmc.init, in_axes=(0))(initial_pos)

    hmc_kernel = jax.jit(hmc.step)

    n_kept_after_burnin = 1
    states = inference_loop(key_infer, hmc_kernel, initial_state, hmc_num_burnin_steps+n_kept_after_burnin, batch_size)

    a_batch_BA = states.position[-n_kept_after_burnin:, :]
    a_batch_BA = a_batch_BA.reshape(-1, env_dim)

    grad_pdf_BP     = jnp.apply_along_axis(grad_pi_pdf, 1, a_batch_BA, pi_theta_P)
    grad_pdf_norm_B = jnp.linalg.norm(grad_pdf_BP, ord=2, axis=1)
    rewards_B       = jnp.apply_along_axis(r_smoothed, 1, a_batch_BA, env)

    scalars_B = rewards_B / (jnp.abs(rewards_B) * grad_pdf_norm_B + 1e-8)
    ISPG_summands_unnorm_BP = scalars_B[:, jnp.newaxis] * grad_pdf_BP

    # compute estimate of the partition function Z_rho
    # by sampling from pi and using importance sampling again
    Z_est_a_BA = pi_sample(key_Z, pi_theta_P, shape=(4096*4, env_dim))
    Z_est_pi_pdf_B = pi_pdf(Z_est_a_BA, pi_theta_P)
    Z_est_rho_pdf_unnorm_B = jnp.apply_along_axis(rho_pdf_unnorm, 1, Z_est_a_BA, pi_theta_P)

    Z_est_B = Z_est_pi_pdf_B / (Z_est_rho_pdf_unnorm_B + 1e-8)
    Z_est = jnp.mean(Z_est_B)

    ISPG_update_P = Z_est * jnp.mean(ISPG_summands_unnorm_BP, axis=0)


    # estimate the variance reduction
    ISPG_var = Z_est * Z_est * jnp.trace(jnp.cov(ISPG_summands_unnorm_BP, rowvar=False))

    a_batch_vre_BA = pi_sample(key_vre, pi_theta_P, shape=a_batch_BA.shape)
    pdf_B          = pi_pdf(a_batch_vre_BA, pi_theta_P)
    grad_pdf_BP    = jnp.apply_along_axis(grad_pi_pdf, 1, a_batch_vre_BA, pi_theta_P)
    rewards_B      = jnp.apply_along_axis(r_smoothed, 1, a_batch_vre_BA, env)

    reinforce_summands_BP = (grad_pdf_BP / (pdf_B[...,jnp.newaxis] + 1e-8)) * rewards_B[...,jnp.newaxis]
    REINFORCE_var = jnp.trace(jnp.cov(reinforce_summands_BP, rowvar=False))

    mag = jnp.linalg.norm(ISPG_update_P, ord=2)
    vre = REINFORCE_var / ISPG_var

    updates_P, opt_state = optimizer.update(ISPG_update_P, opt_state)
    pi_theta_P = optax.apply_updates(pi_theta_P, updates_P)
    #    pi_theta_P = pi_theta_P - lr * ISPG_update_P


    stats_rewards_est_a_BA = pi_sample(key_r, pi_theta_P, shape=(1024, env_dim))
    stats_rewards_est_B = jnp.apply_along_axis(r_smoothed, 1, stats_rewards_est_a_BA, env)
    stats_rewards_est = jnp.mean(stats_rewards_est_B)
    stats_step = [stats_rewards_est, Z_est, vre, mag, None]

    return key, pi_theta_P, opt_state, a_batch_BA, stats_step

def ISPG_hmc(key, pi_theta_P, optimizer, opt_state, n_iter, batch_size, lr):
    stats_reward = []
    stats_policy = []
    stats_Z_est  = []
    stats_vre    = []
    stats_mag    = []
    stats_acc    = []

    key, key_sample = jax.random.split(key)
    sample = pi_sample(key_sample, pi_theta_P, (batch_size, env_dim))

    for it in range(n_iter):
        print('it={},  variance reduction ~ {},   lr*mag={}'.format(it,
            stats_vre[-1] if stats_vre else 'None',
            lr*stats_mag[-1] if stats_mag else 'None'))


        key, pi_theta_P, opt_state, sample, stats_step = ISPG_step_hmc(it, key, pi_theta_P, optimizer, opt_state, sample, batch_size)

        stats_reward.append(stats_step[0])
        stats_pi_theta_P = pi_theta_P
        # record the interpretable variance, not the parameters
        stats_pi_theta_P = stats_pi_theta_P.at[env_dim:].set(jax.nn.softplus(stats_pi_theta_P[env_dim:]))
        stats_policy.append(stats_pi_theta_P)
        stats_Z_est.append(stats_step[1])
        stats_vre.append(stats_step[2])
        stats_mag.append(stats_step[3])
        stats_acc.append(stats_step[4])
        if jnp.isnan(stats_step[1]):
            print(f'NaN on iteration {it} --- early exit')
            break

    stats_run = (stats_reward, stats_policy, stats_Z_est, stats_vre, stats_mag)
    return key, pi_theta_P, opt_state, stats_run


# ==== Experiment config =====
n_iter = 10

# Compare r vs r_smoothed at a point
key_rng, key_a = jax.random.split(key_rng)
a = 5 * jax.random.normal(key_rng, shape=(1, env_dim))
print('r=',          r(a, env))
print('r_smoothed=', r_smoothed(a, env))

key_rng, key_ISPG, *keys_PG = jax.random.split(key_rng, num=5)

# benchmark PG (REINFORCE)
b_small  = 2
PG_lr    = [1.0, 1e-2,  1.0]
PG_b     = [b_small,  4096, 4096]
PG_stats = []

for key, b, lr in zip(keys_PG, PG_b, PG_lr):
    pi_theta_P = jnp.copy(pi_theta_P0)
    optimizer = optax.sgd(lr)
#    optimizer = optax.adam(lr)
    opt_state = optimizer.init(pi_theta_P)
    _, pi_theta_P, _, stats_run = reinforce(key, pi_theta_P, optimizer, opt_state, n_iter, b, lr)
    PG_stats.append(stats_run)


# benchmark ISPG
pi_theta_P = jnp.copy(pi_theta_P0)
ISPG_num_chains = b_small
ISPG_proposal_size = ISPG_num_chains * 128
ISPG_lr = 1e0
#optimizer = optax.adam(ISPG_lr)
optimizer = optax.sgd(ISPG_lr)
opt_state = optimizer.init(pi_theta_P)
key_ISPG, pi_theta_P, _, stats_run_ISPG = ISPG_rej(
    key_ISPG, pi_theta_P, optimizer, opt_state, n_iter, ISPG_proposal_size, ISPG_num_chains, ISPG_lr)
#key_ISPG, pi_theta_P, _, stats_run_ISPG = ISPG_hmc(
#    key_ISPG, pi_theta_P, optimizer, opt_state, n_iter, ISPG_num_chains, ISPG_lr)

ISPG_rew, ISPG_pi_theta, ISPG_Z_est, ISPG_vre, ISPG_mag = stats_run_ISPG

#print(ISPG_rew[:10])
#print(ISPG_pi_theta[:10])
#print(ISPG_Z_est[:10])


fig, ax = plt.subplots()

PG_labels = [f'PG b={b} lr={lr}' for b, lr in zip(PG_b, PG_lr)]
ISPG_label = f'ISPG b={ISPG_num_chains} lr={ISPG_lr}'

for stats, label in zip(PG_stats, PG_labels):
    ax.plot(range(n_iter), stats[0], label=label)

ax.plot(range(n_iter), stats_run_ISPG[0], label=ISPG_label)

ax.set_title(f'Smoothed sum-of-half-spaces env. dim={env_dim}  smoothing_weight={env_smoothing_weight}')

ax.legend()
plt.show()
