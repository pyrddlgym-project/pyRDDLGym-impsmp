import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
from pyRDDLGym_impsmp.samplers.rejection_sampler import FixedNumTrialsRejectionSampler

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
    'random_prev_chain_elt',
    'uniform',
    'normal',
    'rollout_cur_policy',
    'rejection_sampler',
)

def compute_next_sample_corr(D):
    sample_size = len(D)
    D = D - jnp.mean(D)
    K = jnp.correlate(D, D, 'full')
    K = K / K[sample_size-1]
    return K[sample_size]


class HMCSampler:
    def __init__(self,
                 n_iters,
                 batch_size,
                 num_chains,
                 state_dim,
                 action_dim,
                 model,
                 policy,
                 config):

        # Shorthands for common parameters
        # B: Sample batch size
        # C: Number of parallel HMC chains
        # P: Number of policy parameters
        # T: Environment horizon
        # S: State space dimension
        # A: Action space dimension
        self.B = batch_size
        self.C = num_chains
        self.P = policy.n_params
        self.T = model.horizon
        self.S = state_dim
        self.A = action_dim
        self.model = model
        self.policy = policy
        self.config = config

        self.chain_len = config['chain_len']

        if self.config['init_strategy']['type'] == 'normal':
            self.config['init_strategy']['params']['std'] = jnp.sqrt(self.config['init_strategy']['params']['var'])

        self.reinit_strategy = self.config['reinit_strategy']

        assert self.config['step_size_distribution']['type'] in VALID_STEP_SIZE_DISTR_TYPES
        assert self.config['init_strategy']['type'] in VALID_INIT_STRATEGY_TYPES
        assert self.config['reinit_strategy']['type'] in VALID_REINIT_STRATEGY_TYPES




        self.divergence_threshold = self.config.get('divergence_threshold', 10.)
        self.track_next_sample_correlation = self.config.get('track_next_sample_correlation', False)

        self.stats = {
            'step_size': jnp.empty(shape=(n_iters,)),
            'acceptance_rate': jnp.empty(shape=(n_iters,)),
            'num_divergent_chains': jnp.empty(shape=(n_iters,)),
        }

        if self.track_next_sample_correlation:
            self.stats.update({
                'next_sample_correlation_per_chain': jnp.empty(shape=(n_iters, self.config['num_chains'], action_dim, 2, action_dim)),
                'next_sample_correlation_min': jnp.empty(shape=(n_iters,)),
                'next_sample_correlation_mean': jnp.empty(shape=(n_iters,)),
                'next_sample_correlation_max': jnp.empty(shape=(n_iters,)),
            })

        # @@@@@ BEGIN
        #self.rej_subsampler = FixedNumTrialsRejectionSampler(
        #    n_iters,
        #    1,
        #    action_dim,
        #    policy,
        #    config={
        #        'proposal_pdf_type': 'cur_policy',
        #        'sample_shape_type': 'one_sample_per_parameter',
        #        'rejection_rate_schedule': {
        #            'type': 'constant',
        #            'params': {
        #                'value': 250
        #            }
        #        }
        #    })
         #@@@@@ END

    def generate_step_size(self, key):
        """The HMC step size may follow a schedule or may be drawn from a
        random distribution to avoid getting stuck in a periodic pattern.
        """
        key, subkey = jax.random.split(key)

        config = self.config['step_size_distribution']
        type = config['type']
        params = config['params']
        if type == 'constant':
            step_size = params['value']
        elif type == 'uniform':
            step_size = jax.random.uniform(
                subkey,
                minval=params['min'],
                maxval=params['max'])
        elif type == 'discrete_uniform':
            index = jax.random.randint(
                subkey,
                shape=(),
                minval=0, maxval=len(params['values']))
            step_size = params['values'][index]
        elif type == 'log_uniform':
            log_step_size = jax.random.uniform(
                subkey,
                shape=(),
                minval=jnp.log(params['min']),
                maxval=jnp.log(params['max']))
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

        if type == 'random_prev_chain_elt':
            #index_range = self.B * self.C * self.P
            #sampled_indices = jax.random.choice(
            #    subkey,
            #    a=index_range,
            #    shape=(self.C * self.P,),
            #    replace=False)
            #prev_sample = prev_sample.reshape(index_range, self.T, self.A) # Flatten the front
            #init_state = prev_sample[sampled_indices]
            #init_state = init_state.reshape(self.C, self.P, self.T, self.A) # Unflatten
            init_state = prev_sample.reshape(self.B, self.C, self.P, self.T, self.A)

        elif type == 'uniform':
            shape = (self.B, self.C, self.P, self.T, self.A)
            init_state = jax.random.uniform(
                subkey,
                shape=shape,
                minval=params['min'],
                maxval=params['max'])

        elif type == 'normal':
            init_state = jax.random.normal(subkey, shape=shape)
            init_state = params['mean'] + init_state * params['std']

        elif type == 'rollout_cur_policy':
            total_rollouts = self.B * self.C * self.P
            parallel_rollout_keys = jnp.asarray(jax.random.split(subkey, num=total_rollouts))
            flat_init_model_state = init_model_state.reshape(total_rollouts, self.S)
            parallel_rollout = jax.vmap(self.model.rollout_parametrized_policy, (0, 0, None), (0, 0, 0, 0))
            _, _, sampled_actions, _ = parallel_rollout(parallel_rollout_keys, flat_init_model_state, self.policy.theta)
            init_state = sampled_actions.reshape(self.B, self.C, self.P, self.T, self.A)

        elif type == 'rejection_sampler':
            key, init_state, _ = self.rej_subsampler.sample(key, self.policy.theta)

        return key, init_state

    def prep(self,
             key,
             it,
             target_log_prob_fn,
             unconstraining_bijector,
             step_size):
        """Initialize the HMC sampler for the current iteration."""

        num_leapfrog_steps = self.config['num_leapfrog_steps']
        num_burnin_steps = self.config['burnin_per_chain']
        num_adaptation_steps = self.config.get('num_adaptation_steps')

        if num_adaptation_steps is None:
            num_adaptation_steps = int(num_burnin_steps * 0.8)

        parallel_log_density_B = jax.vmap(target_log_prob_fn, (0, 0), 0)
        parallel_log_density_BC = jax.vmap(parallel_log_density_B, (0, 0), 0)

        hmc_sampler = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=parallel_log_density_BC,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps)

        hmc_sampler_with_adaptive_step_size = tfp.mcmc.SimpleStepSizeAdaptation(
            inner_kernel=hmc_sampler,
            num_adaptation_steps=num_adaptation_steps)

        self.sampler = hmc_sampler_with_adaptive_step_size

        #@@@@ BEGIN
        #TODO: Fix the bijector (only apply to 2nd arg)
        #self.sampler = tfp.mcmc.TransformedTransitionKernel(
        #    inner_kernel=hmc_sampler_with_adaptive_step_size,
        #    bijector=unconstraining_bijector)

        #key = self.rej_subsampler.prep(key, it, target_log_prob_fn, unconstraining_bijector)
        #@@@@ END
        return key

    def sample(self, key, theta, init_model_states, sampler_step_size, sampler_init_state):
        key, subkey = jax.random.split(key)
        return key, *tfp.mcmc.sample_chain(
            seed=subkey,
            num_results=1,
            num_burnin_steps=self.config['burnin_per_chain'],
            current_state=(init_model_states, sampler_init_state),
            kernel=self.sampler,
            trace_fn=lambda _, pkr: pkr.inner_results.is_accepted)
            #@@@@ BEGIN
            #TODO: Revert to previous version below once bijector fixed
            #trace_fn=lambda _, pkr: pkr.inner_results.inner_results.is_accepted)
            #@@@@ END

    def update_stats(self, it, samples, is_accepted, step_size):
        self.stats['acceptance_rate'] = self.stats['acceptance_rate'].at[it].set(jnp.mean(is_accepted))
        self.stats['step_size'] = self.stats['step_size'].at[it].set(step_size)

        num_chains = self.config['num_chains']
        num_samples_per_chain = self.config['num_iters_per_chain']

        samples = samples.reshape(num_chains, num_samples_per_chain, self.action_dim, 2, self.action_dim)
        num_divergent_samples_per_chain = jnp.sum(jnp.abs(samples) > self.divergence_threshold, axis=1)
        num_divergent_chains = jnp.sum(num_divergent_samples_per_chain > 0)
        self.stats['num_divergent_chains'] = self.stats['num_divergent_chains'].at[it].set(num_divergent_chains)

        if self.track_next_sample_correlation:
            next_sample_correlation = jnp.apply_along_axis(
                compute_next_sample_corr,
                axis=1,
                arr=samples)
            self.stats['next_sample_correlation_per_chain'] = self.stats['next_sample_correlation_per_chain'].at[it].set(next_sample_correlation)
            self.stats['next_sample_correlation_min'] = self.stats['next_sample_correlation_min'].at[it].set(next_sample_correlation)
            self.stats['next_sample_correlation_mean'] = self.stats['next_sample_correlation_mean'].at[it].set(next_sample_correlation)
            self.stats['next_sample_correlation_max'] = self.stats['next_sample_correlation_max'].at[it].set(next_sample_correlation)

    def print_report(self, it):
        print(f'HMC :: Batch={self.batch_size} :: Chains={self.config["num_chains"]} :: Init.distr={self.config["init_distribution"]["type"]}')
        print(f'       Burnin={self.config["num_burnin_iters_per_chain"]} :: Step size={self.stats["step_size"][it]} :: Num.leapfrog={self.config["num_leapfrog_steps"]}')
        print(f'       Acceptance rate={self.stats["acceptance_rate"][it]} :: Num.div.chains={self.stats["num_divergent_chains"][it]}')
        if self.track_next_sample_correlation:
            print(f'Next sample corr.: {self.stats["next_sample_correlation_min"][it]} <= (Mean) {self.stats["next_sample_correlation_mean"][it]} <= {self.stats["next_sample_correlation_max"][it]}')


class NoUTurnSampler(HMCSampler):
    def prep(self,
             key,
             it,
             target_log_prob_fn,
             unconstraining_bijector,
             step_size):

        num_burnin_iters_per_chain = self.config['burnin_per_chain']
        num_adaptation_steps = self.config.get('num_adaptation_steps')
        max_tree_depth = self.config['max_tree_depth']

        if num_adaptation_steps is None:
            num_adaptation_steps = int(num_burnin_iters_per_chain * 0.8)

        parallel_log_density_B = jax.vmap(target_log_prob_fn, (0, 0), 0)
        parallel_log_density_BC = jax.vmap(parallel_log_density_B, (0, 0), 0)

        nuts = tfp.mcmc.NoUTurnSampler(
            target_log_prob_fn=parallel_log_density_BC,
            step_size=step_size,
            max_tree_depth=max_tree_depth)

        nuts_with_adaptive_step_size = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=nuts,
            num_adaptation_steps=num_adaptation_steps)

        self.sampler = nuts_with_adaptive_step_size
        #@@@@ BEGIN
        #self.sampler = tfp.mcmc.TransformedTransitionKernel(
        #    inner_kernel=nuts_with_adaptive_step_size,
        #    bijector=unconstraining_bijector)

        #key = self.rej_subsampler.prep(key, it, target_log_prob_fn, unconstraining_bijector)
        #@@@@ END
        return key

    def print_report(self, it):
        print(f'NUTS :: Batch={self.batch_size} :: Chains={self.config["num_chains"]} :: Init.distr={self.config["init_distribution"]["type"]}')
        print(f'        Burnin={self.config["num_burnin_iters_per_chain"]} :: Step size={self.stats["step_size"][it]} :: Max.tree depth={self.config["max_tree_depth"]}')
        print(f'        Acceptance rate={self.stats["acceptance_rate"][it]} :: Num.div.chains={self.stats["num_divergent_chains"][it]}')
