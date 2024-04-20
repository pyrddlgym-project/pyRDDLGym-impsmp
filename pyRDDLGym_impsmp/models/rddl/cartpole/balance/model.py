"""Interface to the RDDL CartPole models"""
import jax.numpy as jnp
import jax.random
import os
import copy

import pyRDDLGym

import pyRDDLGym_jax.core.compiler
import pyRDDLGym_jax.core.logic
import pyRDDLGym_jax.core.planner

from pyRDDLGym_impsmp.models.base import BaseDeterministicModel

this_dir = os.path.dirname(os.path.abspath(__file__))

VALID_INITIALIZATION_STRATEGIES = (
    'constant',
    'normal',
    'uniform',
)

VALID_SOLVERS = (
    'euler',
    'adams_bashforth_2step',
    'adams_bashforth_3step',
    'adams_bashforth_4step',
    'adams_bashforth_5step',
)

def generate_initial_states(key, config, batch_shape):
    """Initializes a batch of policy rollouts"""
    if config['type'] == 'constant':
        val = config['params']['value']
        init_states = jnp.ones(shape=batch_shape) * val
    elif config['type'] == 'normal':
        mean = config['params']['mean']
        scale = config['params']['scale']
        init_states = mean + jax.random.normal(key, shape=batch_shape) * scale
    elif config['type'] == 'uniform':
        min = config['params']['min']
        max = config['params']['max']
        init_states = jax.random.uniform(key, shape=batch_shape, minval=min, maxval=max)
    return init_states


class RDDLCartpoleBalanceModel(BaseDeterministicModel):
    """A wrapper around the RDDL Sum-of-Half-Spaces environment.

    In the Sum-of-Half-Spaces environment, the reward function is given by the
    expression

            R(x) = sum_{i=1}^N sgn( sum_{d=1}^|S| W_{id} * (x_d - B_{id} )

    where N is the number of summands parameter, W_{id}, B_{id} are fixed constants
    that define the problem, and x = (x_1, ..., x_|S|) is the state vector.
    Please refer to the domain.rddl file for the domain definition in the RDDL
    language.

    The actions in the environment are translations of the state vector.

    The purpose of the SumOfHalfSpaces wrapper is to provide methods for JIT compiling
    batched RDDL rollouts. The rollouts can use either the relaxed RDDL model or the
    non-relaxed model. The rollouts can sample actions from a given policy with respect
    to the current parameters theta, or the rollouts can use a predetermined sequence
    of actions.

    The RDDL instance files for the SumOfHalfSpaces environment are pre-generated.
    They are indexed by the dimension of the ambient space (action_dim), the number
    of summands in the objective function (n_summands), and the instance index (instance_idx).
    (Several instances are generated for each (action_dim, n_summands) pair in
    order to make statistically meaningful comparisons.)

    For example, when action_dim=81, n_summands=10, instance_idx=3, the relevant
    RDDL instance file is stored in the directory
        models/sum_of_half_spaces/instances/dim81_sum10/instance3.rddl

    Note: As required, new instances may be generated using the script
        models/sum_of_half_spaces/instances/generator.py

    Constructor parameters:
        key: jax.random.PRNGKey
            Key for the current random generator state
        action_dim: Int
            Dimension of the state and action space
        n_summands: Int
            Number of summands in the reward function definition
        instance_idx: Int
            Index of the RDDL instance.
            There are several instances for each (action_dim, n_summands) pair
        is_relaxed: Bool
            Whether or not to apply RDDL relaxations
        initial_state_config: Dict
            Parameters that define the initial state distribution. Please
            also see the generate_initial_states function above.
        reward_shift: Float
            Fixed value to add to all (immediate) reward calculations.
            Can be 0.0
        compiler_kwargs: Dict
            Keyword arguments for the JIT compiler. For example, this
            can specify the batch dimension, or properties of the RDDL
            relaxation.
    """
    def __init__(self,
                 key,
                 state_dim,
                 action_dim,
                 is_relaxed,
                 initial_state_config,
                 reward_shift,
                 dense_reward,
                 solver,
                 compiler_kwargs):


        # initialize the RDDL environment
        if dense_reward:
            reward_type_dir = 'dense'
        else:
            reward_type_dir = 'sparse'
            raise NotImplementedError('RDDL CartPole Balance sparse reward not yet implemented')

        if solver == 'euler':
            solver_type_dir = 'euler'
        elif solver == 'adams_bashforth_2step':
            solver_type_dir = 'ab2'
        elif solver == 'adams_bashforth_3step':
            solver_type_dir = 'ab3'
        elif solver == 'adams_bashforth_4step':
            solver_type_dir = 'ab4'
        elif solver == 'adams_bashforth_5step':
            solver_type_dir = 'ab5'

        self.domain_def_file_path = os.path.join(this_dir, reward_type_dir, solver_type_dir, 'domain.rddl')
        self.instance_def_file_path = os.path.join(this_dir, reward_type_dir, solver_type_dir, 'instance0.rddl')
        self.rddl_env = pyRDDLGym.make(domain=self.domain_def_file_path,
                                       instance=self.instance_def_file_path)

        self.model = self.rddl_env.model
        self.horizon = self.rddl_env.horizon

        self.state_dim = state_dim
        self.action_dim = action_dim
        assert self.action_dim == self.rddl_env.max_allowed_actions

        # compile rollouts
        if is_relaxed:
            self.compile_relaxed(compiler_kwargs)
        else:
            self.compile(compiler_kwargs)

        assert initial_state_config['type'] in VALID_INITIALIZATION_STRATEGIES
        self.initial_state_config = initial_state_config
        self.reward_shift_val = reward_shift

        print(self.rddl_env.observation_space)
        print(self.rddl_env.action_space)

        exit()
        #@@@ edited up to here


    def compile(self, compiler_kwargs):
        """Compiles rollouts in the non-relaxed RDDL model.
        Each rollout takes 'horizon' many steps.
        """
        policy_sample_fn = compiler_kwargs['policy_sample_fn']
        use64bit = compiler_kwargs.get('use64bit', True)

        self.compiler = pyRDDLGym_jax.core.compiler.JaxRDDLCompiler(
            rddl=self.model,
            use64bit=use64bit)

        init_state_subs = self.rddl_env.sampler.subs
        rollout_horizon = self.rddl_env.horizon

        # JIT - compiling various utility methods
        self.compiler.compile()

        # compiling rollouts of the parametrized policy (the actions
        # are sampled using the policy; the policy parameters are
        # passed to the compiled rollouts generator at run-time)
        def parametrized_stochastic_policy(key, policy_params, hyperparams, step, states):
            states = states['s']
            return {'a': policy_sample_fn(key, policy_params['theta'], states)}

        self.parametrized_policy_rollout_sampler = self.compiler.compile_rollouts(
            policy=parametrized_stochastic_policy,
            n_steps=rollout_horizon,
            n_batch=1)

        # compiling rollouts that are used to evaluate a sequence of actions.
        # the actions a0, a1, ..., aT are passed in the hyperparams dictionary.
        # the policy simply takes action a_t at step t
        def deterministic_action_traj_policy(key, policy_params, hyperparams, step, states):
            return {'a': policy_params['a'][step]}

        self.action_traj_evaluator = self.compiler.compile_rollouts(
            policy=deterministic_action_traj_policy,
            n_steps=rollout_horizon,
            n_batch=1)

        # add dummy lead-axis to fit expected semantics
        subs = {}
        for (name, value) in init_state_subs.items():
            value = jnp.array(value)[jnp.newaxis, ...]
            subs[name] = value
        for (state, next_state) in self.model.next_state.items():
            subs[next_state] = subs[state]
        self.subs = subs


    def compile_relaxed(self, compiler_kwargs):
        """Compiles batched rollouts in the relaxed RDDL model.
        Each rollout takes 'horizon' many steps.
        """
        weight = compiler_kwargs.get('weight', 15)
        policy_sample_fn = compiler_kwargs['policy_sample_fn']
        use64bit = compiler_kwargs.get('use64bit', True)

        self.weight = weight

        self.compiler = pyRDDLGym_jax.core.planner.JaxRDDLCompilerWithGrad(
            rddl=self.model,
            logic=pyRDDLGym_jax.core.logic.FuzzyLogic(weight=weight),
            use64bit=use64bit)

        init_state_subs = self.rddl_env.sampler.subs
        rollout_horizon = self.rddl_env.horizon

        # JIT - compiling various utility methods
        self.compiler.compile()

        # compiling rollouts of the parametrized policy (the actions
        # are sampled using the policy; the policy parameters are
        # passed to the compiled rollouts generator at run-time)
        def parametrized_stochastic_policy(key, policy_params, hyperparams, step, states):
            states = states['s']
            return {'a': policy_sample_fn(key, policy_params['theta'], states)}

        self.parametrized_policy_rollout_sampler = self.compiler.compile_rollouts(
            policy=parametrized_stochastic_policy,
            n_steps=rollout_horizon,
            n_batch=1)

        # compiling rollouts that are used to evaluate a sequence of actions.
        # the actions a0, a1, ..., aT are passed in the hyperparams dictionary.
        # the policy simply takes action a_t at step t
        def deterministic_action_traj_policy(key, policy_params, hyperparams, step, states):
            return {'a': policy_params['a'][step]}

        self.action_traj_evaluator = self.compiler.compile_rollouts(
            policy=deterministic_action_traj_policy,
            n_steps=rollout_horizon,
            n_batch=1)

        # cast subs to real numbers, and add dummy lead-axis
        subs = {}
        for (name, value) in init_state_subs.items():
            value = value.astype(self.compiler.REAL)
            value = jnp.array(value)[jnp.newaxis, ...]
            subs[name] = value
        for (state, next_state) in self.model.next_state.items():
            subs[next_state] = subs[state]
        self.subs = subs

    def batch_generate_initial_state(self, key, batch_shape):
        """Generates the initial states over a batch of generic shape.
        The batch does not have to have shape (n_rollouts, state_dim).
        For example, it could have shape (n_rollouts, n_params, state_dim).
        """
        key, subkey = jax.random.split(key)
        init_states = generate_initial_states(subkey, self.initial_state_config, batch_shape)
        return key, init_states

    def rollout_parametrized_policy(self, key, init_states, theta, shift_reward=False):
        """Rolls out the policy with parameters theta.

        Args:
            key: jax.random.PRNGKey
                Random key
            init_states: jnp.array shape=(state_dim,)
                Initial state
            theta: PyTree
                Current policy parameters
            shift_reward: Boolean
                Whether or not to apply a reward shift to the
                rollout rewards

        Returns:
            key: jax.random.PRNGKey
                Mutated key
            states: jnp.array shape=(horizon, state_dim)
            actions: jnp.array shape=(horizon, action_dim)
            rewards: jnp.array shape=(horizon,)
                Sampled trajectory
        """
        self.subs['s'] = init_states[jnp.newaxis, :]

        key, subkey = jax.random.split(key)
        rollouts = self.parametrized_policy_rollout_sampler(
            subkey,
            policy_params={'theta': theta},
            hyperparams=None,
            subs=self.subs,
            model_params=self.compiler.model_params)

        # add the initial state and remove the final state
        # from the trajectory of states generated during the rollout
        truncated_state_traj = rollouts['pvar']['s'][0, :-1]
        states = jnp.concatenate([init_states[jnp.newaxis, :], truncated_state_traj], axis=0)
        actions = rollouts['action']['a'][0]
        # shift rewards if required
        rewards = rollouts['reward'][0] + shift_reward * self.reward_shift_val

        return key, states, actions, rewards

    def rollout_parametrized_policy_batched(self, key, batch_init_states, theta, shift_reward=False):
        """Runs `rollout_parametrized_policy` vectorized over a batch axis.

        Args:
            key: jax.random.PRNGKey
                Random generator state key
            batch_init_states: jnp.array shape=(batch_size, state_dim)
                Batch of initial states
            theta: PyTree
                Current policy parameters
            shift_reward: Boolean
                Whether or not to apply a reward shift to the
                rollout rewards

        Returns:
            key: jax.random.PRNGKey
                Mutated key
            states: jnp.array shape=(horizon, state_dim)
            actions: jnp.array shape=(horizon, action_dim)
            rewards: jnp.array shape=(horizon,)
                Sampled trajectory
        """
        B = batch_init_states.shape[0]
        key, *subkeys = jax.random.split(key, num=B+1)
        subkeys = jnp.asarray(subkeys)
        _, batch_states, batch_actions, batch_rewards = jax.vmap(
            self.rollout_parametrized_policy, (0, 0, None, None), (0, 0, 0, 0))(
                subkeys, batch_init_states, theta, shift_reward)
        return key, batch_states, batch_actions, batch_rewards

    def evaluate_action_trajectory(self, key, init_state, actions, shift_reward=False):
        """Evaluates an action trajectory. The input actions should have shape

                             (horizon, action_dim)

        Args:
            key: jax.random.PRNGKey
                The random generator state key
            init_state: jnp.array shape=(state_dim,)
                The initial state for the trajectory
            actions: jnp.array shape=(horizon, action_dim)
                The action trajectory to follow
            shift_reward: Bool, optional
                Whether or not to add the reward shift value to all
                immediate rewards

        Returns:
            key: jax.random.PRNGKey
                The mutated key
            states: jnp.array shape=(horizon, state_dim)
            actions: jnp.array shape=(horizon, action_dim)
            rewards: jnp.array shape=(horizon,)
                The trajectory
        """
        self.subs['s'] = init_state[jnp.newaxis, :]
        key, subkey = jax.random.split(key)
        rollouts = self.action_traj_evaluator(
            subkey,
            policy_params={'a': actions},
            hyperparams=None,
            subs=self.subs,
            model_params=self.compiler.model_params)

        # add the initial state and remove the final state
        # from the trajectory of states generated during the rollout
        states = jnp.concatenate((init_state[jnp.newaxis, :], rollouts['pvar']['s'][0, :-1]), axis=0)
        # shift rewards if required
        rewards = rollouts['reward'][0] + shift_reward * self.reward_shift_val

        return key, states, actions, rewards

    def evaluate_action_trajectory_batched(self, key, batch_init_states, batch_actions, shift_reward=False):
        """Evaluates a batch of action trajectories. The input actions should have shape

                             (batch_size, horizon, action_dim)

        Args:
            key: jax.random.PRNGKey
                The random generator state key
            init_state: jnp.array shape=(batch_size, state_dim)
                The initial state for the trajectory
            actions: jnp.array shape=(batch_size, horizon, action_dim)
                The action trajectory to follow
            shift_reward: Bool, optional
                Whether or not to add the reward shift value to all
                immediate rewards

        Returns:
            key: jax.random.PRNGKey
                The mutated key
            states: jnp.array shape=(batch_size, horizon, state_dim)
            actions: jnp.array shape=(batch_size, horizon, action_dim)
            rewards: jnp.array shape=(batch_size, horizon)
                Batch of trajectories
        """
        assert batch_init_states.shape[0] == batch_actions.shape[0]
        B = batch_init_states.shape[0]
        key, *subkeys = jax.random.split(key, num=B+1)
        subkeys = jnp.asarray(subkeys)
        _, batch_states, batch_actions, batch_rewards = jax.vmap(
            self.evaluate_action_trajectory, (0, 0, 0, None), (0, 0, 0, 0))(
                subkeys, batch_init_states, batch_actions, shift_reward)
        return key, batch_states, batch_actions, batch_rewards
