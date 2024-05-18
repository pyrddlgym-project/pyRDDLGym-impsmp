import jax
import jax.numpy as jnp
import functools
from tensorflow_probability.substrates import jax as tfp


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

@functools.partial(jax.jit, static_argnames=('target_log_prob_fn', 'num_leapfrog_steps', 'num_burnin_steps', 'num_adaptation_steps'))
def sample_hmc(key,
               theta,
               target_log_prob_fn,
               init_model_states,
               sampler_step_size,
               sampler_init_state,
               num_leapfrog_steps,
               num_burnin_steps,
               num_adaptation_steps):

    hmc_sampler = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        step_size=sampler_step_size,
        num_leapfrog_steps=num_leapfrog_steps)

    hmc_sampler_with_adaptive_step_size = tfp.mcmc.SimpleStepSizeAdaptation(
        inner_kernel=hmc_sampler,
        num_adaptation_steps=num_adaptation_steps)

    #@@@@ BEGIN
    #TODO: Fix the bijector (only apply to 2nd arg)
    #self.sampler = tfp.mcmc.TransformedTransitionKernel(
    #    inner_kernel=hmc_sampler_with_adaptive_step_size,
    #    bijector=unconstraining_bijector)
    #@@@@ END

    key, subkey = jax.random.split(key)
    (init_model_states, sampled_actions), accepted_matrix = tfp.mcmc.sample_chain(
        num_results=1,
        current_state=(init_model_states, sampler_init_state),
        num_burnin_steps=num_burnin_steps,
        kernel=hmc_sampler_with_adaptive_step_size,
        trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
        seed=subkey)

        #@@@@ BEGIN
        #TODO: Revert to previous version below once bijector fixed
        #trace_fn=lambda _, pkr: pkr.inner_results.inner_results.is_accepted)
        #@@@@ END

    return key, accepted_matrix, (init_model_states[0], sampled_actions[0])


@functools.partial(jax.jit, static_argnames=('target_log_prob_fn', 'max_tree_depth', 'num_burnin_steps', 'num_adaptation_steps'))
def sample_nuts(key,
                theta,
                target_log_prob_fn,
                init_model_states,
                sampler_step_size,
                sampler_init_state,
                max_tree_depth,
                num_burnin_steps,
                num_adaptation_steps):

    nuts = tfp.mcmc.NoUTurnSampler(
        target_log_prob_fn=target_log_prob_fn,
        step_size=sampler_step_size,
        max_tree_depth=max_tree_depth)

    nuts_with_adaptive_step_size = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=nuts,
        num_adaptation_steps=num_adaptation_steps)

    #@@@@ BEGIN
    #TODO: Fix the bijector (only apply to 2nd arg)
    #self.sampler = tfp.mcmc.TransformedTransitionKernel(
    #    inner_kernel=hmc_sampler_with_adaptive_step_size,
    #    bijector=unconstraining_bijector)
    #@@@@ END

    key, subkey = jax.random.split(key)
    (init_model_states, sampled_actions), accepted_matrix = tfp.mcmc.sample_chain(
        num_results=1,
        current_state=(init_model_states, sampler_init_state),
        num_burnin_steps=num_burnin_steps,
        kernel=nuts_with_adaptive_step_size,
        trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
        seed=subkey)

        #@@@@ BEGIN
        #TODO: Revert to previous version below once bijector fixed
        #trace_fn=lambda _, pkr: pkr.inner_results.inner_results.is_accepted)
        #@@@@ END

    return key, accepted_matrix, (init_model_states[0], sampled_actions[0])



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
            key, theta, self.target_log_prob_fn, init_model_states, sampler_step_size, sampler_init_state,
            num_leapfrog_steps, num_burnin_steps, num_adaptation_steps)

        self.stats['step_size'] = sampler_step_size
        self.stats['acceptance_rate'] = jnp.mean(accepted_matrix)

        # for HMC, the `non-accepted` samples are nevertheless used for computing the integral estimate
        # so, return the all-ones vector as the last argument
        return key, (init_model_states, sampled_actions), jnp.ones(accepted_matrix[0].shape, dtype=jnp.bool_)

    def print_report(self, it):
        print(f'\tHMC :: Batch={self.B} :: Init={self.config["init_strategy"]["type"]} :: Reinit={self.config["reinit_strategy"]["type"]}')
        print(f'\t       Step size={self.stats["step_size"]} :: Burnin={self.config["burnin_per_chain"]} :: Num.leapfrog={self.config["num_leapfrog_steps"]} :: Acceptance rate={self.stats["acceptance_rate"]:.3f}')


class NoUTurnSampler(HMCSampler):

    def sample(self, key, theta, init_model_states, sampler_step_size, sampler_init_state):
        max_tree_depth = self.config['max_tree_depth']
        num_burnin_steps = self.config['burnin_per_chain']
        num_adaptation_steps = self.config.get('num_adaptation_steps')

        if num_adaptation_steps is None:
            num_adaptation_steps = int(num_burnin_iters_per_chain * 0.8)

        key, accepted_matrix, (init_model_states, sampled_actions) = sample_nuts(
            key, theta, self.target_log_prob_fn, init_model_states, sampler_step_size, sampler_init_state,
            max_tree_depth, num_burnin_steps, num_adaptation_steps)

        self.stats['step_size'] = sampler_step_size
        self.stats['acceptance_rate'] = jnp.mean(accepted_matrix)

        # for NUTS, the `non-accepted` samples are nevertheless used for computing the integral estimate
        # so, return the all-ones vector as the last argument
        return key, (init_model_states, sampled_actions), jnp.ones(accepted_matrix[0].shape, dtype=jnp.bool_)

    def print_report(self, it):
        print(f'\tNUTS :: Batch={self.B} :: Init={self.config["init_strategy"]["type"]} :: Reinit={self.config["reinit_strategy"]["type"]}')
        print(f'\t        Step size={self.stats["step_size"]} :: Burnin={self.config["burnin_per_chain"]} :: Max.tree depth={self.config["max_tree_depth"]} :: Acceptance rate={self.stats["acceptance_rate"]:.3f}')
