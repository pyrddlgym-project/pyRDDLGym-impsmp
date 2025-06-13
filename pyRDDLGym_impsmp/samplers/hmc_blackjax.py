import jax
import numpy as np
import jax.numpy as jnp
import functools
from tensorflow_probability.substrates import jax as tfp

import blackjax

VALID_STEP_SIZE_DISTR_TYPES = (
    'constant',
    'uniform',
    'discrete_uniform',
    'log_uniform',
)

VALID_INIT_STRATEGY_TYPES = (
    'uniform',
    'normal',
    'rollout_cur_policy',
)

VALID_REINIT_STRATEGY_TYPES = (
    'uniform',
    'normal',
    'prev_sample',
    'rollout_cur_policy',
)

@functools.partial(jax.jit, static_argnames=(
    'policy',
    'model',
    'num_leapfrog_steps',
    'num_burnin_steps',
    'num_adaptation_steps'))
def sample_hmc(key,
               theta,
               init_model_states,
               sampler_step_size,
               sampler_init_state,
               policy,
               model,
               num_leapfrog_steps,
               num_burnin_steps,
               num_adaptation_steps):

    # B: batch size (number of parallel chains)
    # P: number of policy parameters
    # T: rollout horizon
    # S: state space dim
    # A: action space dim

    B, P, T, A = sampler_init_state.shape
    _, _, S    = init_model_states.shape


    def unnorm_log_density_vector(key, policy, model, theta, init_model_states, actions):
        """The signed, unnormalized instrumental density for parameter i is defined as

            \tilde{R(tau_i)} * (\partial \pi / \partial \theta_i) (tau_i, theta)

        Where \tilde R denotes the cumulative reward over trajectory tau_i
        in the sampling model, and pi denotes the parametrized policy with
        parameters theta. Note that each parameter theta_i has its own sample
        trajectory (denoted by tau_i).

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
        density_vector = adv * dpi
        return jnp.sum(jnp.log(jnp.abs(density_vector)))

    key, subkey = jax.random.split(key)
    target_log_prob_fn = functools.partial(
        unnorm_log_density_vector,
        subkey,
        policy,
        model,
        theta)

    target_log_prob = lambda x: target_log_prob_fn(**x)

    # TODO: Get rid of hardcoded values
    #inv_mass_matrix =  np.ones(264) * 0.1
    inv_mass_matrix =  np.ones(96) * 0.1

    hmc = blackjax.hmc(target_log_prob, sampler_step_size, inv_mass_matrix, num_leapfrog_steps)

    def inference_loop(rng_key, kernel, initial_state, num_samples, num_chains):
        @jax.jit
        def one_step(states, rng_key):
            keys = jax.random.split(rng_key, num_chains)
            states, _ = jax.vmap(kernel)(keys, states)
            return states, states

        keys = jax.random.split(rng_key, num_samples)
        _, states = jax.lax.scan(one_step, initial_state, keys)

        return states

    initial_pos = {'init_model_states': init_model_states,
                   'actions': sampler_init_state}
    initial_state = jax.vmap(hmc.init, in_axes=(0))(initial_pos)

    hmc_kernel = jax.jit(hmc.step)

    key, subkey = jax.random.split(key)
    states = inference_loop(subkey, hmc_kernel, initial_state, num_burnin_steps+1, B)

    sampled_init_model_states = states.position['init_model_states'][-1]
    sampled_actions           = states.position['actions'][-1]

    accepted_matrix = jnp.ones((1, B, P))
    return key, accepted_matrix, (sampled_init_model_states, sampled_actions)




class HMCSampler:
    def __init__(self,
                 n_iters,
                 batch_size,
                 state_dim,
                 action_dim,
                 model,
                 policy,
                 config):

        # Shorthands for common parameters
        # B: Sample batch size (Number of parallel chains)
        # P: Number of policy parameters
        # T: Environment horizon
        # S: State space dimension
        # A: Action space dimension
        self.B = batch_size
        self.P = policy.n_params
        self.T = model.horizon
        self.S = state_dim
        self.A = action_dim
        self.model = model
        self.policy = policy
        self.config = config

        if self.config['init_strategy']['type'] == 'normal':
            self.config['init_strategy']['params']['std'] = jnp.sqrt(self.config['init_strategy']['params']['var'])

        self.reinit_strategy = self.config['reinit_strategy']

        assert self.config['step_size_distribution']['type'] in VALID_STEP_SIZE_DISTR_TYPES
        assert self.config['init_strategy']['type'] in VALID_INIT_STRATEGY_TYPES
        assert self.config['reinit_strategy']['type'] in VALID_REINIT_STRATEGY_TYPES

        self.stats = {
            'step_size': -1.0,
            'acceptance_rate': -1.0
        }

    def generate_step_size(self, key):
        """The initial HMC step size may follow a schedule or may be drawn
        from a random distribution to avoid getting stuck in a periodic pattern.
        The step-size is then optionally adjusted using tfp.
        """
        key, subkey = jax.random.split(key)

        config = self.config['step_size_distribution']
        type = config['type']
        params = config['params']
        if type == 'constant':
            step_size = params['value']
        elif type == 'uniform':
            step_size = jax.random.uniform(subkey, minval=params['min'], maxval=params['max'])
        elif type == 'discrete_uniform':
            index = jax.random.randint(subkey, shape=(), minval=0, maxval=len(params['values']))
            step_size = params['values'][index]
        elif type == 'log_uniform':
            log_step_size = jax.random.uniform(subkey, shape=(), minval=jnp.log(params['min']), maxval=jnp.log(params['max']))
            step_size = jnp.exp(log_step_size)
        return key, step_size

    def generate_initial_state(self, key, it, init_model_state, prev_sample):
        """The HMC chains may be initialized with a variety of strategies."""
        key, subkey = jax.random.split(key)

        if it == 0:
            config = self.config['init_strategy']
            type = config['type']
            params = config['params']
        else:
            config = self.config['reinit_strategy']
            type = config['type']
            params = config['params']

        if type == 'prev_sample':
            init_state = prev_sample

        elif type == 'uniform':
            shape = (self.B, self.P, self.T, self.A)
            init_state = jax.random.uniform(
                subkey,
                shape=shape,
                minval=params['min'],
                maxval=params['max'])

        elif type == 'normal':
            init_state = jax.random.normal(subkey, shape=shape)
            init_state = params['mean'] + init_state * params['std']

        elif type == 'rollout_cur_policy':
            total_rollouts = self.B * self.P
            parallel_rollout_keys = jnp.asarray(jax.random.split(subkey, num=total_rollouts))
            flat_init_model_state = init_model_state.reshape(total_rollouts, self.S)
            parallel_rollout = jax.vmap(self.model.rollout_parametrized_policy, (0, 0, None, None), (0, 0, 0, 0))
            _, _, sampled_actions, _ = parallel_rollout(parallel_rollout_keys, flat_init_model_state, self.policy, self.policy.theta)
            init_state = sampled_actions.reshape(self.B, self.P, self.T, self.A)

        return key, init_state

    def prep(self,
             key,
             it,
             target_log_prob_fn,
             unconstraining_bijector,
             step_size):
        """Initialize the HMC sampler for the current iteration."""
        parallel_log_density_B = jax.vmap(target_log_prob_fn, (0, 0), 0)
        self.target_log_prob_fn = parallel_log_density_B
        return key

    def sample(self, key, theta, init_model_states, sampler_step_size, sampler_init_state):
        num_leapfrog_steps = self.config['num_leapfrog_steps']
        num_burnin_steps = self.config['burnin_per_chain']
        num_adaptation_steps = self.config.get('num_adaptation_steps')

        if num_adaptation_steps is None:
            num_adaptation_steps = int(num_burnin_steps * 0.8)

        key, accepted_matrix, (init_model_states, sampled_actions) = sample_hmc(
            key, theta, init_model_states, sampler_step_size, sampler_init_state,
            self.policy, self.model, num_leapfrog_steps, num_burnin_steps, num_adaptation_steps)

        self.stats['step_size'] = sampler_step_size
        self.stats['acceptance_rate'] = jnp.mean(accepted_matrix)

        # for HMC, the `non-accepted` samples are nevertheless used for computing the integral estimate
        # so, return the all-ones vector as the last argument
        return key, (init_model_states, sampled_actions), jnp.ones(accepted_matrix[0].shape, dtype=jnp.bool_)

    def print_report(self, it):
        print(f'\tHMC :: Batch={self.B} :: Init={self.config["init_strategy"]["type"]} :: Reinit={self.config["reinit_strategy"]["type"]}')
        print(f'\t       Step size={self.stats["step_size"]} :: Burnin={self.config["burnin_per_chain"]} :: Num.leapfrog={self.config["num_leapfrog_steps"]} :: Acceptance rate={self.stats["acceptance_rate"]:.3f}')
