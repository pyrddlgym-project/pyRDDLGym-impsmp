import os.path
import argparse
from datetime import datetime
from copy import deepcopy
from time import sleep
import numpy as np
import json
import jax
import jax.numpy as jnp

import pyRDDLGym_impsmp.registry.registry as registry

class SimpleNumpyToJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, jnp.integer)): return int(obj)
        if isinstance(obj, (np.floating, jnp.floating)): return float(obj)
        if isinstance(obj, (np.ndarray, jnp.ndarray)): return obj.tolist()
        return super().default(obj)



def main(config):
    # initialize the save-file
    saved_dict = {
        'configuration_file': deepcopy(config)
    }

    action_dim = config['action_dim']
    n_iters = config['n_iters']
    batch_size = config['algorithm']['params']['batch_size']

    # configure JAX
    useGPU = config['useGPU']
    platform_name = 'gpu' if useGPU else 'cpu'
    use64bit = config.get('use64bit', True)
    debug_nans = config.get('debug_nans', True)
    enable_jit = config.get('enable_jit', True)

    jax.config.update('jax_platform_name', platform_name)
    jax.config.update('jax_enable_x64', use64bit)
    jax.config.update('jax_debug_nans', debug_nans)

    jnp.set_printoptions(
        linewidth=9999,
        formatter={'float': lambda x: "{0:0.3f}".format(x)})

    # initialize master random generator key
    seed = config.get('seed', 3264)
    key = jax.random.PRNGKey(seed)

    # configure the bijector
    bijector_config = config['bijector']
    bijector_cls = registry.bijector_lookup_table[bijector_config['id']]
    bijector_params = bijector_config['params']
    bijector = bijector_cls(
        action_dim=action_dim,
        **bijector_params)

    # configure the policy
    policy_config = config['policy']
    policy_cls = registry.policy_lookup_table[policy_config['id']]
    policy_params = policy_config['params']
    key, subkey = jax.random.split(key)
    policy = policy_cls(
        key=subkey,
        action_dim=action_dim,
        bijector=bijector,
        **policy_params)

    # configure the model(s)
    # (note: frequently, the training and evaluation models are configured separately,
    # even if both are not relaxed, because just-in-time compiled (jitted) functions
    # require the shapes of the arguments to be immutable)
    model_config = config['models']
    model_cls = registry.model_lookup_table[model_config['id']]
    model_params = model_config['params']
    model_specs = model_params.pop('specs')
    models = {}
    for model_key, spec in model_specs.items():
        key, subkey = jax.random.split(key)
        spec['compiler_kwargs']['use64bit'] = use64bit
        spec['compiler_kwargs']['policy_sample_fn'] = policy.sample
        spec.update(model_params)
        models[model_key] = model_cls(
            key=subkey,
            action_dim=action_dim,
            **spec)

    # configure the optimizer
    optimizer_config = config['optimizer']
    optimizer_cls = registry.optimizer_lookup_table[optimizer_config['id']]
    optimizer_params = optimizer_config['params']
    optimizer = optimizer_cls(**optimizer_params)

    # configure the algorithm
    algorithm_config = config['algorithm']
    algorithm_fn = registry.algorithm_lookup_table[algorithm_config['id']]
    algorithm_params = algorithm_config['params']

    # configure the sampler (if used)
    if 'sampler' in algorithm_params:
        sampler_config = algorithm_params['sampler']
        sampler_cls = registry.sampler_lookup_table[sampler_config['type']]
        sampler_params = sampler_config['params']
        sampler = sampler_cls(
            n_iters=n_iters,
            batch_size=batch_size,
            action_dim=action_dim,
            policy=policy,
            config=sampler_params)
    else:
        sampler = None

    # run
    with jax.disable_jit(disable=not enable_jit):
        key, algo_stats = algorithm_fn(
            key=key,
            n_iters=n_iters,
            config=algorithm_params,
            bijector=bijector,
            policy=policy,
            sampler=sampler,
            optimizer=optimizer,
            models=models)

    # save stats dump
    save_to = config.get('save_to')
    if save_to is not None:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f'{timestamp}_{algorithm_config["id"]}_{model_config["id"]}_a{action_dim}_iters{n_iters}'

        path = os.path.join(save_to, f'{filename}.json')
        # possible race-condition on file write
        if os.path.isfile(path):
            disambiguator = 1
            while os.path.isfile(path):
                sleep(1)
                path = os.path.join(save_to, f'{filename}-{disambiguator}.json')
                disambiguator = disambiguator + 1

        saved_dict.update(algo_stats)
        with open(path, 'w') as file:
            json.dump(saved_dict, file, cls=SimpleNumpyToJSONEncoder)
        print('Saved results to', path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launch a training run for one of the implemented Policy Gradient algorithms.')
    parser.add_argument('config_path', type=str, help='Path to the configuration file (JSON format, please see the "configs" subdirectory for examples).')
    parser.add_argument('-s', '--save-to', type=str, help='Path where to save the stats results. Optional, defaults to /tmp')

    parser.add_argument('-d', '--dimension', type=int, help='Override the dimension setting.')
    parser.add_argument('-i', '--instance-index', type=int, help='Override the instance index setting.')
    parser.add_argument('-l', '--learning-rate', type=float, help='Override the learning rate setting.')
    parser.add_argument('-b', '--batch-size', type=int, help='Override the batch size setting.')
    parser.add_argument('--num-iters', type=int, help='Override the number of training iterations setting.')

    parser.add_argument('--verbose', type=int, help='Override the verbose printout setting.')
    args = parser.parse_args()

    with open(args.config_path, 'r') as jsonfile:
        config = json.load(jsonfile)

    if args.num_iters is not None:
        config['n_iters'] = args.num_iters
    if args.dimension is not None:
        config['action_dim'] = args.dimension
    if args.instance_index is not None:
        config['models']['params']['instance_idx'] = args.instance_index
    if args.learning_rate is not None:
        config['optimizer']['params']['learning_rate'] = args.learning_rate
    if args.batch_size is not None:
        config['algorithm']['params']['batch_size'] = args.batch_size
    if args.save_to is not None:
        config['save_to'] = args.save_to
    if args.verbose is not None:
        config['algorithm']['params']['verbose'] = args.verbose

    main(config)
