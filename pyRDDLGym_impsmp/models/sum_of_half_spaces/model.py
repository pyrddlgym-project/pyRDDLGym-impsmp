"""RDDL SumOfHalfSpaces models for quick iteration"""
import jax.numpy as jnp
import os

import pyRDDLGym

import pyRDDLGym_jax.core.compiler
import pyRDDLGym_jax.core.logic
import pyRDDLGym_jax.core.planner


this_dir = os.path.dirname(os.path.abspath(__file__))


class SumOfHalfSpacesModel:
    def __init__(self,
                 key,
                 action_dim,
                 n_summands,
                 instance_idx,
                 is_relaxed,
                 reward_shift,
                 compiler_kwargs):
        """A wrapper around the RDDL Sum-of-Half-Spaces environment. The purpose
        of the wrapper is to provide methods for compiling batched RDDL rollouts
        (in either the relaxed model or the non-relaxed model), and a method for
        fetching the rollout loss.

        The RDDL instance files are pre-generated. They are indexed by the dimension
        of the ambient space (action_dim), the number of summands in the objective
        function (n_summands), and the instance index (instance_idx) (several
        instances are generated for each (action_dim, n_summands) pair).

        For example, when action_dim=81, n_summands=10, instance_idx=3, the relevant
        RDDL instance file is stored in the directory
            models/sum_of_half_spaces/instances/dim81_sum10/instance3.rddl

        If necessary, new instances may be generated using the script
            models/sum_of_half_spaces/instances/generator.py
        """
        # check that the requested directory and file exist
        domain_def_file_path = os.path.join(this_dir, 'domain.rddl')
        instance_dir_path = os.path.join(this_dir, 'instances', f'dim{action_dim}_sum{n_summands}')
        if not os.path.isdir(instance_dir_path):
            raise RuntimeError(
                f'[SumOfHalfSpacesModel] Please check that the instance '
                f'directory parameters dim={action_dim}, sum={n_summands} '
                f'has been generated to models/sum_of_half_spaces/instances')
        instance_file_path = os.path.join(instance_dir_path, f'instance{instance_idx}.rddl')
        if not os.path.isfile(instance_file_path):
            raise RuntimeError(
                f'[SumOfHalfSpacesModel] Please check that the instance '
                f'file instance{instance_idx}.rddl has been generated in '
                f'models/sum_of_half_spaces/instances/dim{action_dim}_sum{n_summands}/')

        self.domain_def_file_path = domain_def_file_path
        self.instance_file_path = instance_file_path


        # initialize the RDDL environment
        self.rddl_env = pyRDDLGym.RDDLEnv(domain=self.domain_def_file_path,
                                          instance=self.instance_file_path)

        self.model = self.rddl_env.model
        self.action_dim = action_dim
        self.n_summands = n_summands
        assert self.action_dim == self.rddl_env.max_allowed_actions

        # compile rollouts
        if is_relaxed:
            self.compile_relaxed(compiler_kwargs)
        else:
            self.compile(compiler_kwargs)

        self.reward_shift_val = reward_shift


    def compile(self, compiler_kwargs):
        """Compiles batched rollouts in the non-relaxed RDDL model.
        The batch size is given by n_rollouts.
        """
        n_rollouts = compiler_kwargs['n_rollouts']
        policy_sample_fn = compiler_kwargs['policy_sample_fn']
        use64bit = compiler_kwargs.get('use64bit', True)

        self.n_rollouts = n_rollouts

        self.compiler = pyRDDLGym_jax.core.compiler.JaxRDDLCompiler(
            rddl=self.model,
            use64bit=use64bit)

        init_state_subs = self.rddl_env.sampler.subs
        rollout_horizon = self.rddl_env.horizon

        def policy(key, policy_params, hyperparams, step, states):
            return {'a': policy_sample_fn(key, policy_params['theta'])}

        self.compiler.compile()
        self.sampler = self.compiler.compile_rollouts(
            policy=policy,
            n_steps=rollout_horizon,
            n_batch=n_rollouts)

        # repeat subs over the batch
        subs = {}
        for (name, value) in init_state_subs.items():
            value = jnp.array(value)[jnp.newaxis, ...]
            value_repeated = jnp.repeat(value, repeats=n_rollouts, axis=0)
            subs[name] = value_repeated
        for (state, next_state) in self.model.next_state.items():
            subs[next_state] = subs[state]
        self.subs = subs

    def compile_relaxed(self, compiler_kwargs):
        """Compiles batched rollouts in the relaxed RDDL model.
        The batch size is given by n_rollouts
        """
        n_rollouts = compiler_kwargs['n_rollouts']
        weight = compiler_kwargs.get('weight', 15)
        policy_sample_fn = compiler_kwargs['policy_sample_fn']
        use64bit = compiler_kwargs.get('use64bit', True)

        self.n_rollouts = n_rollouts
        self.weight = weight

        self.compiler = pyRDDLGym_jax.core.planner.JaxRDDLCompilerWithGrad(
            rddl=self.model,
            logic=pyRDDLGym_jax.core.logic.FuzzyLogic(weight=weight),
            use64bit=use64bit)

        init_state_subs = self.rddl_env.sampler.subs
        rollout_horizon = self.rddl_env.horizon

        def policy(key, policy_params, hyperparams, step, states):
            return {'a': policy_sample_fn(key, policy_params['theta'])}

        self.compiler.compile()
        self.sampler = self.compiler.compile_rollouts(
            policy=policy,
            n_steps=rollout_horizon,
            n_batch=n_rollouts)

        # repeat subs over the batch and convert to real
        subs = {}
        for (name, value) in init_state_subs.items():
            value = value.astype(self.compiler.REAL)
            value = jnp.array(value)[jnp.newaxis, ...]
            value_repeated = jnp.repeat(value, repeats=n_rollouts, axis=0)
            subs[name] = value_repeated
        for (state, next_state) in self.model.next_state.items():
            subs[next_state] = subs[state]
        self.subs = subs

    def rollout(self, key, theta, shift_reward=False):
        """Rolls out the policy with parameters theta over a batch

        Args:
            key: jax.random.PRNGKey
                Random key
            theta: pyTree
                Current policy parameters
            shift_reward: Boolean
                Whether or not to apply a reward shift to the
                rollout rewards

        Returns:
            actions: jnp.array
                Array of actions. Shape (n_rollouts, action_dim)
            rewards: jnp.array
                Array of rewards. Shape (n_rollouts,)
        """
        rollouts = self.sampler(
            key,
            policy_params={'theta': theta},
            hyperparams=None,
            subs=self.subs,
            model_params=self.compiler.model_params)

        # squeeze redundant action axis
        actions = rollouts['action']['a'][:, 0, :]
        # shift reward if required
        rewards = rollouts['reward'][..., 0] + shift_reward + self.reward_shift_val

        return actions, rewards
