import jax
import optax
import jax.numpy as jnp
import numpy as np
import haiku as hk
from tensorflow_probability.substrates import jax as tfp


class MultivarNormalHKParametrization:
    """Interface for the policy pi_theta parametrized as a multivariable
    normal distribution N(m, Sigma) with vector of means m and diagonal
    covariance matrix Sigma.

    Constraints can be enforced using a smooth bijection called a bijector
    (see also bijectors in tensorflow_probability). If there are no constraints,
    the identity map can be used as the bijector.

    Different parameterizations are possible (linear, MLP, ...). The different
    parametrizations below are implemented using the dm-haiku library.
    The dm-haiku library stores the parameters theta structured as a
    dictionary, with the weights of each layer and the biases of each layer
    getting a separate dictionary key. To work with theta, it is often
    necessary to make use of the jax.tree_util module.
    """
    def __init__(self, key, state_dim, action_dim, bijector):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.bijector = bijector

    def apply(self, key, theta, state):
        """Apply the policy to get the means vector and the covariance matrix"""
        raise NotImplementedError

    def sample_batch(self, key, theta, state, n):
        mean, cov = self.apply(key, theta, state)
        action_sample = jax.random.multivariate_normal(
            key, mean, cov,
            shape=n)
        action_sample = self.bijector.forward(action_sample)
        return action_sample

    def sample(self, key, theta, state):
        return self.sample_batch(key, theta, state, n=None)

    def pdf(self, key, theta, state, actions):
        mean, cov = self.apply(key, theta, state)
        unconstrained_actions = self.bijector.inverse(actions)
        normal_pdf = jax.scipy.stats.multivariate_normal.pdf(
            unconstrained_actions,
            mean=mean,
            cov=cov)
        density_correction = jnp.apply_along_axis(self.bijector._inverse_det_jacobian, axis=-1, arr=actions)
        return normal_pdf * density_correction

    def diagonal_of_jacobian(self, key, theta, s, a):
        """A computation of the diagonal of the Jacobian matrix that uses JAX
        primitives, and works with any policy parametrization, but
        computes the entire Jacobian before taking the diagonal. Therefore,
        the computation time scales rather poorly with increasing dimension.
        There is apparently no natural way in JAX of computing the diagonal
        only without also computing all of the off-diagonal terms.

        See also:
            https://stackoverflow.com/questions/70956578/jacobian-diagonal-computation-in-jax
        """
        raise NotImplementedError

    def analytic_diagonal_of_jacobian(self, key, theta, s, a):
        """When it is possible to compute the diagonal of the Jacobian terms
        analytically (as it is for the linear parametrization of a normal distribution,
        for example), substituting the analytic computation in place of the general
        auto-differentiation as in the diagonal_of_jacobian method can significantly
        improve scaling of computation time with respect to dimension"""
        raise NotImplementedError

    def clip_theta(self, theta):
        """Clips the covariance parameters of the policy from below to the configured
        value, accounting for the softplus transform"""
        raise NotImplementedError



class MultivarNormalLinearParametrization(MultivarNormalHKParametrization):
    """Linear normal policy parametrization for horizon=1 problems.

    In this case, there is no state-dependence, and no need for bias terms,
    and therefore the number of parameters in the parameterization is equal
    to 2 * action_dim, where action_dim denotes the dimension of the action
    space.

    Explicitly the parameters are: mean and covariance values, one-per-dimension.

    NOTE: The state_dim parameter is ignored for this class. If state-dependence
    is desired, please see the MultivarNormalLinearStateDependentParametriaztion
    below, which encodes state-dependence at the cost of possibly more parameters
    """
    def __init__(self, key, state_dim, action_dim, bijector, cov_lower_cap):
        super().__init__(
            key=key,
            state_dim=-1, # See also MultivarNormalLinearStateDependentParametrization below
            action_dim=action_dim,
            bijector=bijector)

        def pi(input):
            linear = hk.Linear(2, with_bias=False)
            output = linear(input)
            mean, cov = output.T
            cov = jax.nn.softplus(cov)
            cov = jnp.diag(cov)
            return mean, cov

        self.one_hot_inputs = jnp.eye(action_dim)
        self.pi = hk.transform(pi)
        self.theta = self.pi.init(key, self.one_hot_inputs)
        self.n_params = sum(leaf.flatten().shape[0] for leaf in jax.tree_util.tree_leaves(self.theta))

        # record the lower cap value to be applied to the covariance terms,
        # correcting for usage of softplus
        if cov_lower_cap > 0.0:
            self.cov_lower_cap = np.log(np.exp(cov_lower_cap) - 1.0)
        else:
            self.cov_lower_cap = -np.inf

    def apply(self, key, theta, state):
        return self.pi.apply(theta, key, self.one_hot_inputs)

    def diagonal_of_jacobian(self, key, theta, s, a):
        dpi = jax.jacrev(self.pdf, argnums=1)(key, theta, s, a)
        dpi = jax.tree_util.tree_map(lambda x: jnp.diagonal(x, axis1=0, axis2=3), dpi)
        dpi = jax.tree_util.tree_map(lambda x: jnp.diagonal(x, axis1=0, axis2=2), dpi)
        dpi = jax.tree_util.tree_map(lambda x: x[0], dpi)
        return dpi

    def analytic_diagonal_of_jacobian(self, key, theta, s, a):
        """The following computes the diagonal of the Jacobian analytically.
        It valid ONLY when the policy is parametrized by a normal distribution
        with parameters

            mu_i
            sigma_i^2 = softplus(u_i)

        In this case, it is possible to compute the partials in closed form,
        and avoid computing the off-diagonal terms in the Jacobian.

        The scaling of computation time with increasing dimension seems much
        improved.

        TODO: Update to include state-dep
        """
        pi_val = self.pdf(key, theta, a)[..., 0]

        theta = theta['linear']['w']
        mu = theta[:, 0]
        u = theta[:, 1]
        sigsq = jax.nn.softplus(u)

        # the softplus correction comes from the chain rule
        softplus_correction = 1 - (1/(1 + jnp.exp(u)))

        mu_mult = (jnp.diag(a[:,0,0,:]) - mu) / sigsq
        N = (jnp.diag(a[:,1,0,:]) - mu)
        sigsq_mult = 0.5 * softplus_correction * (((N * N) / sigsq) - 1) / sigsq

        partials = jnp.stack([mu_mult, sigsq_mult], axis=1) * pi_val
        return partials

    def clip_theta(self, theta):
        """Clips the covariance parameters of the policy from below to the configured
        value, accounting for the softplus transform"""
        return jax.tree_util.tree_map(
            lambda term: term.at[:,1].set(jnp.maximum(term[:,1], self.cov_lower_cap)),
            self.theta)


class MultivarNormalLinearStateDependentParametrization(MultivarNormalHKParametrization):
    """Linear normal policy parametrization for sequential (horizon > 1) problems.

    In this case, the parametrized policy has an explicit state-dependence,
    and bias terms are required to ensure that the actions do not vanish at the
    all-zeros state.

    The number of parameters is equal to
        (state_dim + 1) * (2 * action_dim)
        |--input+bias--|  |--- output ---|

    NOTE: If no state-dependence is required (for example, the problem is not
    sequential, i.e. has horizon=1), then MultivarNormalLinearParametrization
    can be used, which requires fewer parameters.
    """
    def __init__(self, key, state_dim, action_dim, bijector, cov_lower_cap):
        super().__init__(
            key=key,
            state_dim=state_dim, #See also MultivarNormalLinearParametrization above
            action_dim=action_dim,
            bijector=bijector)

        # record the lower cap value to be applied to the covariance terms,
        # correcting for usage of softplus
        #if cov_lower_cap > 0.0:
        #    self.cov_lower_cap = np.log(np.exp(cov_lower_cap) - 1.0)
        #else:
        #    self.cov_lower_cap = -np.inf
        self.cov_lower_cap = cov_lower_cap

        def pi(input):
            linear = hk.Linear(2 * action_dim, with_bias=True)
            output = linear(input)
            mean, cov = jnp.split(output, action_dim, axis=-1)
            cov = jax.nn.softplus(cov)
            cov = jnp.maximum(cov, self.cov_lower_cap)
            # convert cov to a diagonal matrix along the last two indices
            cov_diag = jnp.zeros(shape=(input.shape[:-1] + (action_dim, action_dim)))
            cov_diag = cov_diag.at[..., jnp.arange(action_dim), jnp.arange(action_dim)].set(cov[..., :])
            return mean, cov_diag

        dummy_state_input = jnp.ones(shape=(state_dim,)) # used for initializing parameters theta
        self.pi = hk.transform(pi)
        self.theta = self.pi.init(key, dummy_state_input)
        self.n_params = sum(leaf.flatten().shape[0] for leaf in jax.tree_util.tree_leaves(self.theta))

    def apply(self, key, theta, state):
        return self.pi.apply(theta, key, state)

    def diagonal_of_jacobian(self, key, theta, s, a):
        def flat_reduce(accumulator, item):
            item = item.reshape(self.n_params, 1, -1)
            accumulator = jnp.concatenate([accumulator, item], axis=2)
            return accumulator

        dpi = jax.jacrev(self.pdf, argnums=1)(key, theta, s, a)
        dpi_matrix = jnp.squeeze(jax.tree_util.tree_reduce(flat_reduce, dpi))
        return jnp.diagonal(dpi_matrix)

    def clip_theta(self, theta):
        """Clips the covariance parameters of the policy from below to the configured
        value, accounting for the softplus transform"""
        return jax.tree_util.tree_map(
            lambda term: term.at[:,1].set(jnp.maximum(term[:,1], self.cov_lower_cap)),
            theta)





class MultivarNormalMLPParametrization(MultivarNormalHKParametrization):
    """ MLP with 32x32 hidden nodes parametrization for horizon=1 problems."""
    def __init__(self, key, state_dim, action_dim, bijector):
        super().__init__(
            key=key,
            state_dim=-1,
            action_dim=action_dim,
            bijector=bijector)

        def pi(input):
            mlp = hk.Sequential([
                hk.Linear(32), jax.nn.relu,
                hk.Linear(32), jax.nn.relu,
                hk.Linear(2)
            ])
            output = mlp(input)
            mean, cov = output.T
            cov = jax.nn.softplus(cov)
            cov = jnp.diag(cov)
            return mean, cov



if __name__ == '__main__':
    # define tests
    def linear_param_test_analytic_derivative_matches_autograd_derivative(action_dim, test_sample_size, cpu):
        """Test that the analytic derivative and the autograd derivative yield consistent results"""
        import pyRDDLGym_impsmp.bijectors
        if cpu:
            jax.config.update('jax_platform_name', 'cpu')

        bij_identity = pyRDDLGym_impsmp.bijectors.identity.Identity(action_dim)
        key = jax.random.PRNGKey(42)
        test_policy = MultivarNormalLinearParametrization(
            key=key,
            action_dim=action_dim,
            bijector=bij_identity,
            cov_lower_cap=0.0)

        key, *subkeys = jax.random.split(key, num=4)
        a = test_policy.sample(subkeys[0], test_policy.theta, (test_sample_size, action_dim, 2, 1))

        autograd_dpi = jax.vmap(test_policy.diagonal_of_jacobian, (None, None, 0), 0)(subkeys[1], test_policy.theta, a)['linear']['w']
        analytic_dpi = jax.vmap(test_policy.analytic_diagonal_of_jacobian, (None, None, 0), 0)(subkeys[2], test_policy.theta, a)

        print('[linear_param_test_analytic_derivative_matches_autograd_derivative] First three samples of diag(Jacobian) computed using autograd:')
        print(autograd_dpi[:3])
        print('[linear_param_test_analytic_derivative_matches_autograd_derivative] First three samples of diag(Jacobian) computed using analytic formula:')
        print(analytic_dpi[:3])

        test_result = jnp.all(jnp.isclose(autograd_dpi, analytic_dpi))
        print(f'[linear_param_test_analytic_derivative_matches_autograd_derivative] All samples, all coordinates close: {test_result}')
        return test_result

    # run tests
    assert linear_param_test_analytic_derivative_matches_autograd_derivative(action_dim=8, test_sample_size=1000, cpu=True)
