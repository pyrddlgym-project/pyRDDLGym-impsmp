import optax
from tensorflow_probability.substrates import jax as tfp
import pyRDDLGym_impsmp.algorithms
import pyRDDLGym_impsmp.bijectors
import pyRDDLGym_impsmp.models
import pyRDDLGym_impsmp.policies
import pyRDDLGym_impsmp.samplers
#import pyRDDLGym.Examples.Traffic.Calibration.instances.inflow_calibration_models
#import pyRDDLGym.Examples.SumOfHalfSpaces.instances.model

algorithm_lookup_table = {
    'reinforce': pyRDDLGym_impsmp.algorithms.reinforce.reinforce,
    'impsmp': pyRDDLGym_impsmp.algorithms.impsmp.impsmp,
    'impsmp_per_parameter': pyRDDLGym_impsmp.algorithms.impsmp_per_parameter.impsmp_per_parameter,
    'impsmp_per_parameter_signed': pyRDDLGym_impsmp.algorithms.impsmp_per_parameter_signed.impsmp_per_parameter_signed,
}

bijector_lookup_table = {
    'identity': pyRDDLGym_impsmp.bijectors.identity.Identity,
    'simplex': pyRDDLGym_impsmp.bijectors.simplex.SimplexBijector,
}

model_lookup_table = {
#    'inflow_calibration': pyRDDLGym.Examples.Traffic.Calibration.instances.inflow_calibration_models.InflowCalibrationModel,
#    'inflow_calibration_grid_2x2': pyRDDLGym.Examples.Traffic.Calibration.instances.inflow_calibration_models.InflowCalibration2x2GridModel,
#    'inflow_calibration_grid_3x3': pyRDDLGym.Examples.Traffic.Calibration.instances.inflow_calibration_models.InflowCalibration3x3GridModel,
#    'inflow_calibration_grid_4x4': pyRDDLGym.Examples.Traffic.Calibration.instances.inflow_calibration_models.InflowCalibration4x4GridModel,
#    'inflow_calibration_grid_6x6': pyRDDLGym.Examples.Traffic.Calibration.instances.inflow_calibration_models.InflowCalibration6x6GridModel,
    'sum_of_half_spaces': pyRDDLGym_impsmp.models.sum_of_half_spaces.model.SumOfHalfSpacesModel,
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
    'hmc': pyRDDLGym_impsmp.samplers.hmc.HMCSampler,
    'nuts': pyRDDLGym_impsmp.samplers.hmc.NoUTurnSampler,
    'fixed_num_trials_rejection_sampler': pyRDDLGym_impsmp.samplers.rejection_sampler.FixedNumTrialsRejectionSampler,
    'fixed_num_accepted_rejection_sampler': pyRDDLGym_impsmp.samplers.rejection_sampler.FixedNumAcceptedRejectionSampler,
}
