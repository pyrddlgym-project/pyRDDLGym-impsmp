import optax
from tensorflow_probability.substrates import jax as tfp
import pyRDDLGym_impsmp.advantage_estimators
import pyRDDLGym_impsmp.algorithms
import pyRDDLGym_impsmp.bijectors
import pyRDDLGym_impsmp.models
import pyRDDLGym_impsmp.policies
import pyRDDLGym_impsmp.samplers
#import pyRDDLGym.Examples.Traffic.Calibration.instances.inflow_calibration_models

advantage_estimator_lookup_table = {
    'total_traj_reward': pyRDDLGym_impsmp.advantage_estimators.reinforce_estimators.TotalTrajRewardAdvEstimator,
    'future_traj_reward': pyRDDLGym_impsmp.advantage_estimators.reinforce_estimators.FutureTrajRewardAdvEstimator,
    'future_traj_reward_w_constant_baseline': pyRDDLGym_impsmp.advantage_estimators.reinforce_estimators.FutureTrajRewardWConstantBaselineAdvEstimator,
    'future_traj_reward_w_running_average_baseline': pyRDDLGym_impsmp.advantage_estimators.reinforce_estimators.FutureTrajRewardWRunningAvgBaselineAdvEstimator,
    'V_function': pyRDDLGym_impsmp.advantage_estimators.reinforce_estimators.FutureTrajRewardWLearnedBaselineAdvEstimator,
    'Q_function': pyRDDLGym_impsmp.advantage_estimators.reinforce_estimators.QFunctionAdvEstimator,
    'A_function': pyRDDLGym_impsmp.advantage_estimators.reinforce_estimators.AFunctionAdvEstimator,
    'TD_residual': pyRDDLGym_impsmp.advantage_estimators.reinforce_estimators.TDResidualAdvEstimator,
    'sampling_model_Q_function': pyRDDLGym_impsmp.advantage_estimators.impsamp_estimators.SamplingModelQFunctionAdvEstimator,
}

algorithm_lookup_table = {
    'reinforce': pyRDDLGym_impsmp.algorithms.reinforce2.reinforce,
    'impsamp': pyRDDLGym_impsmp.algorithms.impsamp.impsamp,
    'impsamp_split_by_sign': pyRDDLGym_impsmp.algorithms.impsamp_split_by_sign.impsamp,
}

bijector_lookup_table = {
    'identity': pyRDDLGym_impsmp.bijectors.identity.Identity,
    'simplex': pyRDDLGym_impsmp.bijectors.simplex.SimplexBijector,
}

model_lookup_table = {
    'rddl_sum_of_half_spaces': pyRDDLGym_impsmp.models.rddl.sum_of_half_spaces.model.RDDLSumOfHalfSpacesModel,
    'rddl_cartpole_balance': pyRDDLGym_impsmp.models.rddl.cartpole.balance.model.RDDLCartpoleBalanceModel,
    'mujoco_cartpole_balance': pyRDDLGym_impsmp.models.mujoco.cartpole.balance.model.MuJoCoCartpoleBalanceModel,
    'purejax_sum_of_half_spaces': pyRDDLGym_impsmp.models.purejax.sum_of_half_spaces.model.JAXSumOfHalfSpacesModel,
#    'rddl_inflow_calibration': pyRDDLGym.Examples.Traffic.Calibration.instances.inflow_calibration_models.InflowCalibrationModel,
}

optimizer_lookup_table = {
    'adabelief': optax.adabelief,
    'adafactor': optax.adafactor,
    'adagrad': optax.adagrad,
    'adam': optax.adam,
    'adamw': optax.adamw,
    'adamax': optax.adamax,
    'adamaxw': optax.adamaxw,
    'amsgrad': optax.amsgrad,
    'fromage': optax.fromage,
    'lamb': optax.lamb,
    'lars': optax.lars,
    'lion': optax.lion,
    'noisy_sgd': optax.noisy_sgd,
    'novograd': optax.novograd,
    'optimistic_gradient_descent': optax.optimistic_gradient_descent,
    'dpsgd': optax.dpsgd,
    'radam': optax.radam,
    'rmsprop': optax.rmsprop,
    'sgd': optax.sgd,
    'sgd_with_momentum': optax.sgd,
    'sm3': optax.sm3,
    'yogi': optax.yogi,
}

policy_lookup_table = {
    'multivar_normal_with_linear_parametrization': pyRDDLGym_impsmp.policies.normal.MultivarNormalLinearParametrization,
    'multivar_normal_with_mlp_parametrization': pyRDDLGym_impsmp.policies.normal.MultivarNormalMLPParametrization,
}

sampler_lookup_table = {
    'hmc': pyRDDLGym_impsmp.samplers.hmc.HMCSampler,       # tensorflow_probability
    'nuts': pyRDDLGym_impsmp.samplers.hmc.NoUTurnSampler,  # tensorflow_probability
    'hmc_blackjax': pyRDDLGym_impsmp.samplers.hmc_blackjax.HMCSampler,
    'fixed_num_proposed_rejection_sampler': pyRDDLGym_impsmp.samplers.rejection_sampler.FixedNumProposedRejectionSampler,
    'fixed_num_sampled_rejection_sampler': pyRDDLGym_impsmp.samplers.rejection_sampler.FixedNumSampledRejectionSampler,
}
