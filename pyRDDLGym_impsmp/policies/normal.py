import math
import jax
import optax
import jax.numpy as jnp
import numpy as np
import haiku as hk
from tensorflow_probability.substrates import jax as tfp


class MultivarNormalHKParametrization:
    """Interface for the policy pi_theta parametrized as a
    multivariable normal distribution N(m, Sigma) with vector
    of means m and diagonal covariance matrix Sigma. Here

               m = m(s) and Sigma = Sigma(s)

    are functions of the environment state.

    Different functions can be used for implementing the
    state-dependence of the policy (e.g. linear, MLP, ...).
    The functions below are implemented using the dm-haiku library.
    The dm-haiku library stores the parameters theta structured as a
    nested dictionary, with each layer getting a separate key.
    For linear layers, the weights and biases are themselves stored
    in a dictionary with separate keys for the weights and the biases.
    This nested dictionary structure is a special case of a PyTree.
    To work with theta, it is often necessary to make use of the
    jax.tree_util module.

    Constraints can be enforced using a smooth bijection called
    a bijector (see also bijectors in tensorflow_probability).
    If there are no constraints, the identity map can be used
    as the bijector.
    """
    def __init__(self,
                 key,
                 state_dim,
                 action_dim,
                 bijector,
                 compute_jacobians_analytically):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.bijector = bijector

        if compute_jacobians_analytically:
            self.diagonal_of_jacobian      = self.analytic_diagonal_of_jacobian
            self.diagonal_of_jacobian_traj = self.analytic_diagonal_of_jacobian_traj
        else:
            self.diagonal_of_jacobian      = self.autodiff_diagonal_of_jacobian
            self.diagonal_of_jacobian_traj = self.autodiff_diagonal_of_jacobian_traj

    def set_weights(self, theta):
        """Set the policy weights theta to the provided values.
        The provided theta must have the same PyTree shape as
        the policy theta"""
        #TODO: This gets confused by theta formatted as {'key': [[a], [b]]}
        self.theta = jax.tree_util.tree_map(jnp.array, theta)

    def apply(self, key, theta, state):
        """Apply the policy to get the means vector and the covariance matrix"""
        raise NotImplementedError

    def sample_batch(self, key, theta, state, n):
        mean, cov = self.apply(key, theta, state)
        action_sample = jax.random.multivariate_normal(key, mean, cov, shape=n)
        action_sample = self.bijector.forward(action_sample)
        return action_sample

    def sample(self, key, theta, state):
        return self.sample_batch(key, theta, state, n=None)

    def pdf(self, key, theta, state, action):
        mean, cov = self.apply(key, theta, state)
        unconstrained_action = self.bijector.inverse(action)
        normal_pdf = jax.scipy.stats.multivariate_normal.pdf(
            unconstrained_action,
            mean=mean,
            cov=cov)
        density_correction = jnp.apply_along_axis(self.bijector._inverse_det_jacobian, axis=-1, arr=action)
        return normal_pdf * density_correction

    def pdf_traj_vector(self, key, theta, states, actions):
        """Calculates the vector of densities

                pi(a_t | s_t), t = 0, 1, 2, ..., T-1

           over a batch of state-action trajectories.
        Args:
            key: jax.random.PRNGKey
                Random generator state key
            theta: PyTree (generated by hk)
                Dictionary of policy parameters
            states: jnp.array shape=(..., T, state_dim)
            actions: jnp.array shape=(..., T, action_dim)
                The trajectory or a batch of trajectories.
                The batch may have an arbitrary shape.

        Returns:
            jnp.array shape=(..., T)
                The densities
        """
        assert states.shape[-2] == actions.shape[-2]
        T = states.shape[-2]
        batch_shape = states.shape[:-2]
        batch_size = np.prod(batch_shape)

        key, *subkeys = jax.random.split(key, num=batch_size+1)
        subkeys = jnp.asarray(subkeys).reshape(batch_shape + (2,))
        pdfs = jax.vmap(self.pdf, (0, None, 0, 0), 0)(subkeys, theta, states, actions)
        pdfs = pdfs.reshape(states.shape[:-1])
        return pdfs

    def pdf_traj(self, key, theta, states, actions):
        """Calculates the product

                \Pi_{t=0}^{T-1} pi(a_t | s_t)

        Args:
            key: jax.random.PRNGKey
                Random generator state key
            theta: PyTree (generated by hk)
                Dictionary of policy parameters
            states: jnp.array shape=(..., T, state_dim)
            actions: jnp.array shape=(..., T, action_dim)
                The trajectory or a batch of trajectories.
                The batch may have an arbitrary shape.

        Returns:
            Float or jnp.array
                Product of densities, or the batched version
        """
        pdfs = self.pdf_traj_vector(key, theta, states, actions)
        return jnp.prod(pdfs, axis=-1)

    def autodiff_diagonal_of_jacobian(self, key, theta, s, a):
        """The Importance Sampling augmentation of the REINFORCE algorithm
        introduces a family of instrumental densities rho_i, one per parameter
        theta_i.

        When sampling from this family of densities, we frequently need to
        evaluate terms of the form

                          \partial pi
                        ---------------- (a_i | s_i)    (i=1, 2, ..., n)
                        \partial theta_i

        that is, each parameter theta_i has a dedicated state-action sample
        (a_i, s_i). Treating pi as a vector-valued function (mapping into R^n),
        we see that we are looking for the diagonal of an n x n Jacobian matrix.

        The `autodiff` implementation of the computation uses JAX primitives.
        The disadvantage of this approach is that the entire Jacobian needs to
        be computed before taking the diagonal. Therefore, the computation time
        scales rather poorly with increasing dimension. There is apparently no
        natural way in JAX of computing the diagonal only without also computing
        all of the off-diagonal terms.

        See also:
            https://stackoverflow.com/questions/70956578/jacobian-diagonal-computation-in-jax

        The `_traj` version of the problem is similar, but computes the partial
        derivatives of the probabilities of a sample of full state-action trajectories
        under the policy pi.

        The `analytic` versions of these computations perform the partial
        derivative computations by-hand. This way, we can only perform the
        computation of the diagonal terms, without computing the entire
        Jacobian matrix. This can significantly improve the scaling of
        computation time with respect to dimension.
        """
        raise NotImplementedError

    def autodiff_diagonal_of_jacobian_traj(self, key, theta, states, actions):
        """Please see the docstring for `autodiff_diagonal_of_jacobian`"""
        raise NotImplementedError

    def analytic_diagonal_of_jacobian(self, key, theta, s, a):
        """Please see the docstring for `autodiff_diagonal_of_jacobian`"""
        raise NotImplementedError

    def analytic_diagonal_of_jacobian_traj(self, key, theta, states, actions):
        """Please see the docstring for `autodiff_diagonal_of_jacobian`"""
        raise NotImplementedError

    def print_report(self, it):
        """Prints out policy information"""
        raise NotImplementedError



class MultivarNormalLinearParametrization(MultivarNormalHKParametrization):
    """Normal policy parametrization, where the means vector m and covariance
    matrix diagonal diag(Sigma) are linear functions of the state vector.

    Bias terms are required to ensure that the actions do not vanish at the
    all-zeros state.

    The number of parameters is equal to

                    (state_dim + 1) * (2 * action_dim)
                    |--input+bias--|  |--- output ---|
    """
    def __init__(self,
                 key,
                 state_dim,
                 action_dim,
                 bijector,
                 cov_lower_cap,
                 compute_jacobians_analytically=False):

        super().__init__(
            key=key,
            state_dim=state_dim,
            action_dim=action_dim,
            bijector=bijector,
            compute_jacobians_analytically=compute_jacobians_analytically)

        self.cov_lower_cap = cov_lower_cap

        def pi(input):
            linear = hk.Linear(2 * action_dim, with_bias=True)
            output = linear(input)
            mean, cov = jnp.split(output, 2, axis=-1)
            cov = jax.nn.softplus(cov)
            cov = jnp.maximum(cov, self.cov_lower_cap)
            cov_diag = jnp.apply_along_axis(jnp.diag, axis=-1, arr=cov)
            return mean, cov_diag

        dummy_state_input = jnp.ones(shape=(state_dim,)) # used for initializing parameters theta
        self.pi = hk.transform(pi)
        self.theta = self.pi.init(key, dummy_state_input)
        self.n_params = sum(leaf.flatten().shape[0] for leaf in jax.tree_util.tree_leaves(self.theta))

    def apply(self, key, theta, state):
        return self.pi.apply(theta, key, state)

    def autodiff_diagonal_of_jacobian(self, key, theta, s, a):
        """Please see the base class.

        The input states and actions should have shape

            (n_params, state_dim)
            (n_params, action_dim)

        respectively. For other shapes, vmap over the left indices (e.g. batch index).
        """
        dpi = jax.jacrev(self.pdf, argnums=1)(key, theta, s, a)
        dpi_w = dpi['linear']['w'].reshape(-1, 2 * self.state_dim * self.action_dim)
        dpi_b = dpi['linear']['b'].reshape(-1, 2 * self.action_dim)
        dpi_matrix = jnp.concatenate([dpi_w, dpi_b], axis=-1)
        return jnp.diagonal(dpi_matrix)

    def autodiff_diagonal_of_jacobian_traj(self, key, theta, states, actions):
        """Please see the base class.

        The input state and action trajectories should have shape

            (n_params, horizon, state_dim)
            (n_params, horizon, action_dim)

        respectively. For other shapes, vmap over the left indices (e.g. batch index)
        """
        dpi = jax.jacrev(self.pdf_traj, argnums=1)(key, theta, states, actions)
        dpi_w = dpi['linear']['w'].reshape(-1, 2 * self.state_dim * self.action_dim)
        dpi_b = dpi['linear']['b'].reshape(-1, 2 * self.action_dim)
        dpi_matrix = jnp.concatenate([dpi_w, dpi_b], axis=-1)
        return jnp.diagonal(dpi_matrix)

    def analytic_diagonal_of_jacobian(self, key, theta, s, a):
        """Please see the base class and the docstring for `autodiff_diagonal_of_jacobian`"""
        pi_val = self.pdf(key, theta, s, a)[:, jnp.newaxis]
        mu, cov = self.apply(key, theta, s)
        var = jnp.diagonal(cov, axis1=-2, axis2=-1)
        softplus_crctn = 1 - jnp.exp(-var)

        C = ((a - mu) / var)
        dpidmu = C * pi_val
        dpidu = 0.5 * softplus_crctn * (C * C - (1/var)) * pi_val

        D = jnp.concatenate([dpidmu, dpidu], axis=1)
        aug_s = jnp.pad(s, ((0, 0), (0, 1)), mode='constant', constant_values=1.0)

        n_out = 2 * self.action_dim
        D_indices = tuple(np.array([(j, j % n_out) for j in range(self.n_params)]).T)
        aug_s_indices = tuple(np.array([(j, math.floor(j / n_out)) for j in range(self.n_params)]).T)

        D = D[D_indices]
        aug_s = aug_s[aug_s_indices]
        jac_diagonal = D * aug_s
        return jac_diagonal

    def analytic_diagonal_of_jacobian_traj(self, key, theta, states, actions):
        """Please see the base class and the docstring for `autodiff_diagonal_of_jacobian`.
        Applies `analytic_diagonal_of_jacobian` and the product rule."""
        T = states.shape[-2]
        key, subkey = jax.random.split(key)

        pi_vals = self.pdf_traj_vector(subkey, theta, states, actions)
        key, *subkeys = jax.random.split(key, num=T+1)
        subkeys = jnp.asarray(subkeys)
        dpi_vals = jax.vmap(self.analytic_diagonal_of_jacobian, (0, None, 1, 1), 1)(subkeys, theta, states, actions)

        ratios = dpi_vals / (pi_vals + 1e-12)
        jac_diagonal = jnp.prod(pi_vals, axis=-1) * jnp.sum(ratios, axis=-1)

        return jac_diagonal

    def flatten_dJ(self, dJ):
        """Converts a dictionary representation of a derivative of J with respect
        to the policy parameters theta into a flattened representation"""
        dJ_w = dJ['linear']['w'].ravel()
        dJ_b = dJ['linear']['b'].ravel()
        return jnp.concatenate([dJ_w, dJ_b])

    def unflatten_dJ(self, dJ):
        """Converts a flattened dJ back into a PyTree of same shape as the policy
        parameters object theta, (which is the required format for updating theta)
        """
        break_idx = 2 * self.state_dim * self.action_dim
        dJ_w = dJ[..., :break_idx].reshape(self.state_dim, 2 * self.action_dim)
        dJ_b = dJ[..., break_idx:].reshape(2 * self.action_dim)
        return {'linear': {
            'w': dJ_w,
            'b': dJ_b
        }}

    def print_report(self, it):
        """Prints out policy information"""
        print(f'\tPolicy :: Normal (Linear with bias) n_params={self.n_params}')



class MultivarNormalMLPParametrization(MultivarNormalHKParametrization):
    """Normal policy parametrization, where the means vector m and covariance
    matrix diagonal diag(Sigma) are parametrized by an MLP as a function of
    the state vector.
    """
    def __init__(self,
                 key,
                 state_dim,
                 action_dim,
                 bijector,
                 cov_lower_cap,
                 compute_jacobians_analytically=False):

        super().__init__(
            key=key,
            state_dim=state_dim,
            action_dim=action_dim,
            bijector=bijector,
            compute_jacobians_analytically=compute_jacobians_analytically)

        self.cov_lower_cap = cov_lower_cap

        def pi(input):
            mlp = hk.Sequential([
                hk.Linear(32), jax.nn.relu,
                hk.Linear(32), jax.nn.relu,
                hk.Linear(2 * action_dim)
            ])
            output = mlp(input)
            mean, cov = jnp.split(output, 2, axis=-1)
            cov = jax.nn.softplus(cov)
            cov = jnp.maximum(cov, self.cov_lower_cap)
            cov_diag = jnp.apply_along_axis(jnp.diag, axis=-1, arr=cov)
            return mean, cov_diag

        dummy_state_input = jnp.ones(shape=(state_dim,)) # used for initializing parameters theta
        self.pi = hk.transform(pi)
        self.theta = self.pi.init(key, dummy_state_input)
        self.n_params = sum(leaf.flatten().shape[0] for leaf in jax.tree_util.tree_leaves(self.theta))

    def apply(self, key, theta, state):
        return self.pi.apply(theta, key, state)

    def autodiff_diagonal_of_jacobian(self, key, theta, s, a):
        """Please see the base class.

        The input states and actions should have shape

            (n_params, state_dim)
            (n_params, action_dim)

        respectively. For other shapes, vmap over the left indices (e.g. batch index).
        """
        raise NotImplementedError
        dpi = jax.jacrev(self.pdf, argnums=1)(key, theta, s, a)
        dpi_w = dpi['linear']['w'].reshape(-1, 2 * self.state_dim * self.action_dim)
        dpi_b = dpi['linear']['b'].reshape(-1, 2 * self.action_dim)
        dpi_matrix = jnp.concatenate([dpi_w, dpi_b], axis=-1)
        return jnp.diagonal(dpi_matrix)

    def autodiff_diagonal_of_jacobian_traj(self, key, theta, states, actions):
        """Please see the base class.

        The input state and action trajectories should have shape

            (n_params, horizon, state_dim)
            (n_params, horizon, action_dim)

        respectively. For other shapes, vmap over the left indices (e.g. batch index)
        """
        raise NotImplementedError
        dpi = jax.jacrev(self.pdf_traj, argnums=1)(key, theta, states, actions)
        dpi_w = dpi['linear']['w'].reshape(-1, 2 * self.state_dim * self.action_dim)
        dpi_b = dpi['linear']['b'].reshape(-1, 2 * self.action_dim)
        dpi_matrix = jnp.concatenate([dpi_w, dpi_b], axis=-1)
        return jnp.diagonal(dpi_matrix)

    def analytic_diagonal_of_jacobian(self, key, theta, s, a):
        """Please see the base class and the docstring for `autodiff_diagonal_of_jacobian`"""
        raise NotImplementedError

    def analytic_diagonal_of_jacobian_traj(self, key, theta, states, actions):
        """Please see the base class and the docstring for `autodiff_diagonal_of_jacobian`.
        Applies `analytic_diagonal_of_jacobian` and the product rule."""
        raise NotImplementedError

    def unflatten_dJ(self, dJ):
        """Converts a flattened dJ back into a PyTree of same shape as the policy
        parameters object theta, (which is the required format for updating theta)
        """
        raise NotImplementedError

    def print_report(self, it):
        """Prints out policy information"""
        print(f'\tPolicy :: Normal (MLP [32, 32]) n_params={self.n_params}')



if __name__ == '__main__':
    # define tests
    def linear_param_test_analytic_derivative_matches_autograd_derivative(state_dim, action_dim, horizon, test_sample_size, cpu):
        """Test that the analytic derivative and the autograd derivative yield consistent results"""
        import pyRDDLGym_impsmp.bijectors
        if cpu:
            jax.config.update('jax_platform_name', 'cpu')

        bij_identity = pyRDDLGym_impsmp.bijectors.identity.Identity(action_dim)
        key = jax.random.PRNGKey(42)
        test_policy = MultivarNormalLinearParametrization(
            key=key,
            state_dim=state_dim,
            action_dim=action_dim,
            bijector=bij_identity,
            cov_lower_cap=0.0)

        key, state_gen_key = jax.random.split(key)

        B = test_sample_size
        P = test_policy.n_params
        T = horizon
        S = state_dim

        state_mean = jnp.zeros((B, P, T, S))
        state_cov = jnp.diag(jnp.ones(S))
        state_cov = jnp.stack([state_cov] * T, axis=0)
        state_cov = jnp.stack([state_cov] * P, axis=0)
        state_cov = jnp.stack([state_cov] * B, axis=0)
        states = jax.random.multivariate_normal(state_gen_key, state_mean, state_cov, shape=(B, P, T))

        sample_B = jax.vmap(test_policy.sample, (0, None, 0), 0)
        sample_BP = jax.vmap(sample_B, (0, None, 0), 0)
        sample_BPT = jax.vmap(sample_BP, (0, None, 0), 0)

        key, *action_sample_subkeys = jax.random.split(key, num=1+B*P*T)
        action_sample_subkeys = jnp.asarray(action_sample_subkeys).reshape(B, P, T, 2)
        actions = sample_BPT(action_sample_subkeys, test_policy.theta, states)

        key, *subkeys = jax.random.split(key, num=1+4*B)
        autodiff_keys = jnp.asarray(subkeys[:B])
        analytic_keys = jnp.asarray(subkeys[B:2*B])
        autodiff_traj_keys = jnp.asarray(subkeys[2*B:3*B])
        analytic_traj_keys = jnp.asarray(subkeys[3*B:4*B])

        # single-sample test
        autodiff_dpi = jax.vmap(test_policy.autodiff_diagonal_of_jacobian, (0, None, 0, 0), 0)(autodiff_keys, test_policy.theta, states[:, :, 0], actions[:, :, 0])
        analytic_dpi = jax.vmap(test_policy.analytic_diagonal_of_jacobian, (0, None, 0, 0), 0)(analytic_keys, test_policy.theta, states[:, :, 0], actions[:, :, 0])

        print('[linear_param_test_analytic_derivative_matches_autograd_derivative] First three samples of diag(Jacobian) computed using autodiff (JAX primitives):')
        print(autodiff_dpi[:3])
        print('[linear_param_test_analytic_derivative_matches_autograd_derivative] First three samples of diag(Jacobian) computed using analytic formula:')
        print(analytic_dpi[:3])

        test_result_single = jnp.all(jnp.isclose(autodiff_dpi, analytic_dpi))
        # done

        # trajectory test
        autodiff_traj_dpi = jax.vmap(test_policy.autodiff_diagonal_of_jacobian_traj, (0, None, 0, 0), 0)(autodiff_keys, test_policy.theta, states, actions)
        analytic_traj_dpi = jax.vmap(test_policy.analytic_diagonal_of_jacobian_traj, (0, None, 0, 0), 0)(analytic_keys, test_policy.theta, states, actions)

        print('[linear_param_test_analytic_derivative_matches_autograd_derivative] First three samples of diag(JacobianTraj) computed using autodiff (JAX primitives):')
        print(autodiff_traj_dpi[:3])
        print('[linear_param_test_analytic_derivative_matches_autograd_derivative] First three samples of diag(JacobianTraj) computed using analytic formula:')
        print(analytic_traj_dpi[:3])

        test_result_traj = jnp.all(jnp.isclose(autodiff_traj_dpi, analytic_traj_dpi))
        # done

        print(f'[linear_param_test_analytic_derivative_matches_autograd_derivative] Single-sample: All coordinates close: {test_result_single}')
        print(f'[linear_param_test_analytic_derivative_matches_autograd_derivative] Trajectory:    All coordinates close: {test_result_traj}')
        return test_result_single and test_result_traj

    # run tests
    assert linear_param_test_analytic_derivative_matches_autograd_derivative(state_dim=2, action_dim=8, horizon=10, test_sample_size=1000, cpu=True)
