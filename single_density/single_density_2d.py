import jax
import numpy as np
import jax.numpy as jnp
import jax.scipy.stats as st

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
# Smoothed reward:           r_sm(x) = sum_i tanh((n_i . (x - b_i)) / smoothing_weight)
#
#
# Definition of "Instance 0" of the Dim=2, Summands=10 RDDL
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

env = (env_n, env_b)

def r(a, env):
    env_n, env_b = env
    x = jnp.einsum('ij,ij->i', env_n, (a - env_b))
    x = jnp.sign(x)
    return jnp.sum(x)

smoothing_weight = 0.1

def r_smoothed(a, env):
    env_n, env_b = env
    x = jnp.einsum('ij,ij->i', env_n, (a - env_b))
    x = jnp.tanh(x / smoothing_weight)
    return jnp.sum(x)


# assume that the policy is distributed as a multivariate normal with parameters
# mu = (mu_x, mu_y) and sigma = ((sigma_xx, 0), (0, sigma_yy))

def initialize_theta_params(key_rng):
    key_rng, key_mu, key_sigma = jax.random.split(key_rng, num=3)
    params_mu    = 5 *  jax.random.normal(key_mu, shape=(2,))
    params_sigma = 10 * jax.random.normal(key_sigma, shape=(2,))
    return key_rng, jnp.concatenate((params_mu, params_sigma))

key_rng, pi_theta_P0 = initialize_theta_params(key_rng)


def pi_sample(key, pi_theta_P, shape):
    # shape should end in (..., 2)
    x = jax.random.normal(key, shape)
    return pi_theta_P[0:2] + pi_theta_P[2:] * x

def pi_pdf(x, pi_theta_P):
    # shape of x should end in (..., 2)
    return st.multivariate_normal.pdf(x, pi_theta_P[0:2], jnp.diag(pi_theta_P[2:]))

grad_pi_pdf = jax.jit(jax.grad(pi_pdf, argnums=1))

@partial(jax.jit, static_argnames=['batch_size', 'lr'])
def reinforce_step(key, pi_theta_P, batch_size, lr):
    key, key_sample = jax.random.split(key, num=2)
    a_batch_BA  = pi_sample(key_sample, pi_theta_P, shape=(batch_size, 2))
    pdf_B       = pi_pdf(a_batch_BA, pi_theta_P)
    grad_pdf_BP = jnp.apply_along_axis(grad_pi_pdf, 1, a_batch_BA, pi_theta_P)
    rewards_B   = jnp.apply_along_axis(r_smoothed, 1, a_batch_BA, env)

    reinforce_summands_BP = (grad_pdf_BP / (pdf_B[...,jnp.newaxis] + 1e-9)) * rewards_B[...,jnp.newaxis]
    reinforce_update_P    = jnp.mean(reinforce_summands_BP, axis=0)

    pi_theta_P = pi_theta_P - lr * reinforce_update_P

    reward     = jnp.mean(rewards_B)
    stats_step = [reward]

    return key, pi_theta_P, stats_step


def reinforce(key, pi_theta_P, n_iter, batch_size, lr):
    stats_reward   = []
    stats_pi_theta_P = []
    stats_run = (stats_reward, stats_pi_theta_P)

    for _ in range(n_iter):
        key, pi_theta_P, stats_step = reinforce_step(key, pi_theta_P, batch_size, lr)
        stats_run[0].append(stats_step[0])
        stats_run[1].append(pi_theta_P)

    return key, pi_theta_P, stats_run


M = 1
def rho_pdf_unnorm(a_A, pi_theta_P):
    reward = r_smoothed(a_A, env)
    pdf = pi_pdf(a_A, pi_theta_P)
    grad_pdf_P = grad_pi_pdf(a_A, pi_theta_P)
    norm_dlogpi = M * jnp.linalg.norm(grad_pdf_P / M, ord=2) / (pdf + 1e-6)
    rho_val = jnp.abs(reward) * norm_dlogpi
    return rho_val

def log_rho_pdf_unnorm(a_A, pi_theta_P):
    reward = r_smoothed(a_A, env)
    pdf = pi_pdf(a_A, pi_theta_P)
    grad_pdf_P = grad_pi_pdf(a_A, pi_theta_P)
    norm_dlogpi = M * jnp.linalg.norm(grad_pdf_P / M, ord=2) / (pdf + 1e-6)
    log_rho_val = jnp.log(jnp.abs(reward)) + jnp.log(norm_dlogpi)
    return log_rho_val

@partial(jax.jit, static_argnames=('batch_size', 'lr'))
def ISPG_step_hmc(key, pi_theta_P, batch_size, lr):
    key, key_sample, key_infer, key_r, key_Z = jax.random.split(key, num=5)

    hmc_target_log_prob = lambda a: log_rho_pdf_unnorm(a, pi_theta_P)
    hmc_inv_mass_matrix = np.ones(2) * 0.1
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

    initial_pos = pi_sample(key_sample, pi_theta_P, (batch_size, 2))
    initial_state = jax.vmap(hmc.init, in_axes=(0))(initial_pos)

    hmc_kernel = jax.jit(hmc.step)

    states = inference_loop(key_infer, hmc_kernel, initial_state, hmc_num_burnin_steps+1, batch_size)

    a_batch_BA = states.position[-1, :]

    grad_pdf_BP     = jnp.apply_along_axis(grad_pi_pdf, 1, a_batch_BA, pi_theta_P)
    grad_pdf_norm_B = jnp.linalg.norm(grad_pdf_BP, ord=2, axis=1)
    rewards_B       = jnp.apply_along_axis(r_smoothed, 1, a_batch_BA, env)

    scalars_B = rewards_B / (jnp.abs(rewards_B) * grad_pdf_norm_B + 1e-9)
    ISPG_summands_unnorm_BP = scalars_B[:, jnp.newaxis] * grad_pdf_BP

    Z_est_a_BA = pi_sample(key_Z, pi_theta_P, shape=(4096, 2))
    Z_est_pi_pdf_B = pi_pdf(Z_est_a_BA, pi_theta_P)
    Z_est_rho_pdf_unnorm_B = jnp.apply_along_axis(rho_pdf_unnorm, 1, Z_est_a_BA, pi_theta_P)

    Z_est_B = Z_est_pi_pdf_B / (Z_est_rho_pdf_unnorm_B + 1e-6)
    Z_est = jnp.mean(Z_est_B)

    ISPG_update_P = Z_est * jnp.mean(ISPG_summands_unnorm_BP, axis=0)

    pi_theta_P = pi_theta_P - lr * ISPG_update_P

    stats_rewards_est_a_BA = pi_sample(key_r, pi_theta_P, shape=(32, 2))
    stats_rewards_est_B = jnp.apply_along_axis(r_smoothed, 1, stats_rewards_est_a_BA, env)
    stats_rewards_est = jnp.mean(stats_rewards_est_B)
    stats_step = [stats_rewards_est, Z_est]

    return key, pi_theta_P, stats_step


def ISPG(key, pi_theta_P, n_iter, batch_size, lr):
    stats_reward     = []
    stats_pi_theta_P = []
    stats_Z_est      = []
    stats_run = (stats_reward, stats_pi_theta_P, stats_Z_est)

    for it in range(n_iter):
        print(it)
        key, pi_theta_P, stats_step = ISPG_step_hmc(key, pi_theta_P, batch_size, lr)
        stats_run[0].append(stats_step[0])
        stats_run[1].append(pi_theta_P)
        stats_run[2].append(stats_step[1])
        if jnp.isnan(stats_step[1]):
            print(f'NaN on iteration {it} --- early exit')
            break

    return key, pi_theta_P, stats_run

n_iter = 200

pi_theta_P = pi_theta_P0
key_rng, pi_theta_P, stats_run_PG32 = reinforce(key_rng, pi_theta_P, n_iter, 32, 1e-2)

pi_theta_P = pi_theta_P0
key_rng, pi_theta_P, stats_run_PG4k = reinforce(key_rng, pi_theta_P, n_iter, 4096, 1e-2)

pi_theta_P = pi_theta_P0
key_rng, pi_theta_P, stats_run_PG4k_2 = reinforce(key_rng, pi_theta_P, n_iter, 4096, 1.0)

pi_theta_P = pi_theta_P0
key_rng, pi_theta_P, stats_run_ISPG = ISPG(key_rng, pi_theta_P, n_iter, 32, 1e-2)

ISPG_rew, ISPG_pi_theta, ISPG_Z_est = stats_run_ISPG

print(ISPG_rew[:20])
print(ISPG_pi_theta[:20])
print(ISPG_Z_est[:20])

print('Final policy params:')
print('PG:')
print(stats_run_PG4k[1][-5:])
print('ISPG:')
print(ISPG_pi_theta[-5:])


fig, ax = plt.subplots()

ax.plot(range(n_iter), stats_run_PG32[0], label='PG b=32 lr=1e-2')
ax.plot(range(n_iter), stats_run_PG4k[0], label='PG b=4096 lr=1e-2')
ax.plot(range(n_iter), stats_run_PG4k_2[0], label='PG b=4096 lr=1.0')
ax.plot(range(n_iter), stats_run_ISPG[0], label='ISPG b=32 lr=1e-2')

ax.set_title(f'Smoothed sum-of-half-spaces env.  smoothing_weight={smoothing_weight}')

ax.legend()
plt.show()
