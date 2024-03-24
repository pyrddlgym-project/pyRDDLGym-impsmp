import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

VALID_PROPOSAL_PDF_TYPES = (
    'cur_policy',
    'uniform',
)
VALID_SHAPE_TYPES = (
    'one_sample_per_parameter',
    'one_sample_per_dJ_summand',
)
VALID_REJECTION_RATE_TYPES = (
    'constant',
    'linear_ramp',
)

def rejection_rate_schedule(it, type_, params):
    if type_ == 'constant':
        return params['value']
    elif type_ == 'linear_ramp':
        return params['from'] + it * params['delta']

class BaseRejectionSampler:
    def __init__(self,
                 n_iters,
                 batch_size,
                 action_dim,
                 policy,
                 config):
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.policy = policy
        self.config = config

        self.proposal_pdf_type = self.config['proposal_pdf_type']
        assert self.proposal_pdf_type in VALID_PROPOSAL_PDF_TYPES

        self.shape_type = self.config['sample_shape_type']
        assert self.shape_type in VALID_SHAPE_TYPES

        self.rejection_rate_type = self.config['rejection_rate_schedule']['type']
        assert self.rejection_rate_type in VALID_REJECTION_RATE_TYPES
        self.rejection_rate_params = self.config['rejection_rate_schedule']['params']
        if self.rejection_rate_type == 'linear_ramp':
            self.rejection_rate_params['delta'] = (self.rejection_rate_params['to'] - self.rejection_rate_params['from']) / n_iters
        self.rejection_rate = None

    def prep(self,
             key,
             it,
             target_log_prob_fn,
             unconstraining_bijector):
        self.target_log_prob_fn = target_log_prob_fn
        self.unconstraining_bijector = unconstraining_bijector
        self.rejection_rate = rejection_rate_schedule(it, self.rejection_rate_type, self.rejection_rate_params)
        return key

    def generate_initial_state(self, key, it, samples):
        """Included to have a consistent interface with that of HMC"""
        return key

    def generate_step_size(self, key):
        """Included to have a consistent interface with that of HMC"""
        return key

    def sample(self, key, theta):
        raise NotImplementedError

    def update_stats(self, it, samples, is_accepted):
        raise NotImplementedError

    def print_report(self, it):
        raise NotImplementedError



class FixedNumTrialsRejectionSampler(BaseRejectionSampler):
    def __init__(self,
                 n_iters,
                 batch_size,
                 action_dim,
                 policy,
                 config):
        super().__init__(n_iters, batch_size, action_dim, policy, config)

        if self.shape_type == 'one_sample_per_parameter':
            self.shape = (self.action_dim, 2, 1)
        elif self.shape_type == 'one_sample_per_dJ_summand':
            self.shape = (1,)

    def sample(self, key, theta, states):
        def _accept_reject(carry, x):
            key = carry
            key, *subkeys = jax.random.split(key, num=4)

            if self.proposal_pdf_type == 'cur_policy':
                proposed_sample = self.policy.sample_batch(subkeys[0], theta, states, self.shape)
                proposal_density_val = self.policy.pdf(subkeys[1], theta, states, proposed_sample)[..., 0]
            elif self.proposal_pdf_type == 'uniform':
                minval, maxval = -8.0, 8.0
                proposed_sample = jax.random.uniform(subkeys[1], shape=self.shape, minval=minval, maxval=maxval)
                proposal_density_val = (1/(maxval - minval))**(self.action_dim) * jnp.ones(shape=self.shape)

            instrumental_density_val = jnp.exp(self.target_log_prob_fn(proposed_sample))

            u = jax.random.uniform(subkeys[2], shape=self.shape[:-1])

            accepted_matrix = u < (instrumental_density_val / (self.rejection_rate * proposal_density_val))

            return key, (proposed_sample, accepted_matrix)

        key, (proposed_samples, accepted_matrix) = jax.lax.scan(_accept_reject, init=key, xs=None, length=self.batch_size)

        return key, proposed_samples, accepted_matrix



class FixedNumAcceptedRejectionSampler(BaseRejectionSampler):
    def __init__(self,
                 n_iters,
                 batch_size,
                 action_dim,
                 policy,
                 config):
        super().__init__(n_iters, batch_size, action_dim, policy, config)

        self.n_distinct_samples_used_buffer = []
        self.stats = {
            'n_distinct_samples_used': []
        }

    def cond_fn(self, val):
        _, _, _, _, _, _, is_sampled = val
        res = jnp.logical_not(jnp.all(is_sampled))
        return res

    def body_fn(self, val):
        key, M, theta, states, samples, n, is_sampled = val
        key, *subkeys = jax.random.split(key, num=4)

        if self.proposal_pdf_type == 'cur_policy':
            proposed_action = self.policy.sample(subkeys[0], theta, states)
            proposal_density_val = self.policy.pdf(subkeys[1], theta, states, proposed_action)[..., 0]
        elif self.proposal_pdf_type == 'uniform':
            minval, maxval = -8.0, 8.0
            proposed_action = jax.random.uniform(subkeys[1], shape=(self.action_dim,), minval=minval, maxval=maxval)
            proposal_density_val = (1/(maxval - minval))**(self.action_dim) * jnp.ones(shape=(1,))

        instrumental_density_val = jnp.exp(self.target_log_prob_fn(states, proposed_action))

        # sample independent uniform variables to test the acceptance criterion
        u = jax.random.uniform(subkeys[2], shape=instrumental_density_val.shape)
        acceptance_criterion = u < (instrumental_density_val / (M * proposal_density_val))

        # accept if meet criterion and not previously accepted
        acceptance_criterion = jnp.logical_and(acceptance_criterion,
                                               jnp.logical_not(is_sampled))

        # update used sample count
        n = n + jnp.any(acceptance_criterion)

        # mark newly accepted actions
        is_sampled = jnp.logical_or(is_sampled, acceptance_criterion)

        if self.shape_type == 'one_sample_per_parameter':
            acceptance_criterion = jnp.broadcast_to(
                acceptance_criterion[:, jnp.newaxis, jnp.newaxis],
                shape=samples.shape)

        samples = jnp.where(acceptance_criterion, proposed_action, samples)

        return (key, M, theta, states, samples, n, is_sampled)

    def sample(self, key, theta, n_params, states):

        def _accept_reject(key, states, theta, n_params, M):
            """Runs Accept/Reject sampling for a single element of the sample batch"""
            if self.shape_type == 'one_sample_per_parameter':
                # add a dummy dimension 1 to be consistent with the HMC return shape
                samples = jnp.empty((n_params, 1, self.action_dim))
                is_sampled = jnp.zeros(n_params).astype(bool)
                states = jnp.repeat(states[jnp.newaxis, ...], n_params, axis=0)
                states = jnp.repeat(states[:, jnp.newaxis, ...], 1, axis=1)
            elif self.shape_type == 'one_sample_per_dJ_summand':
                samples = jnp.empty(shape=(1, self.action_dim))
                is_sampled = jnp.zeros(shape=(1,)).astype(bool)

            init_val = (key, M, theta, states, samples, 0, is_sampled)
            _, _, _, _, samples, n_distinct, _ = jax.lax.while_loop(self.cond_fn, self.body_fn, init_val)
            return samples, n_distinct

        # run rejection sampling over a batch
        key, *batch_subkeys = jax.random.split(key, num=self.batch_size+1)
        batch_subkeys = jnp.asarray(batch_subkeys)
        accept_reject_parallel_over_batch = jax.vmap(_accept_reject, (0, 0, None, None, None), 0)

        samples, n_distinct = accept_reject_parallel_over_batch(batch_subkeys, states, theta, n_params, self.rejection_rate)

        # keep in buffer until statistics for the current iteration are updated
        self.n_distinct_samples_used_buffer.append(n_distinct[0])

        return key, samples, None

    def update_stats(self, it, samples, is_accepted):
        self.stats['n_distinct_samples_used'].extend(self.n_distinct_samples_used_buffer)
        self.n_distinct_samples_used_buffer.clear()

    def print_report(self, it):
        print(f'Rejection Sampler'
              f' :: Batch={self.batch_size}'
              f' :: Rej.rate cur.val={self.rejection_rate},'
              f' sched.type={self.rejection_rate_type}'
              f' :: Proposal pdf={self.config["proposal_pdf_type"]}'
              f' :: # Distinct samples={self.stats["n_distinct_samples_used"][-1]}')
