import json
import pyRDDLGym.PolicyGradient.main
from sys import argv
from copy import deepcopy

base_config_path = argv[1]
save_to = argv[2]
dim = int(argv[3])

with open(base_config_path, 'r') as jsonfile:
    base_config = json.load(jsonfile)

base_config['action_dim'] = dim
base_config['algorithm']['params']['sampler']['params']['rejection_rate_schedule']['type'] = 'constant'

base_config['algorithm']['params']['verbose'] = False
base_config['save_to'] = save_to

cov_clip_vals = (0.005, 0.01, 0.05)
rej_rate_vals = (250, 1000, 5000)
learning_rate_vals = (0.01, 0.03, 0.1, 0.3, 0.7, 1.1, 1.3, 1.7, 2.1, 2.3, 2.7)


print('DIM', dim)
for rej_rate in rej_rate_vals:
    for cov_clip in cov_clip_vals:
        for learning_rate in learning_rate_vals:
            config = deepcopy(base_config)

            config['policy']['params']['cov_lower_cap'] = cov_clip
            config['algorithm']['params']['sampler']['params']['rejection_rate_schedule']['params']['value'] = rej_rate
            config['optimizer']['params']['learning_rate'] = learning_rate

            print(f'cov_clip={cov_clip}, rej_rate={rej_rate}, learning_rate={learning_rate}')
            pyRDDLGym.PolicyGradient.main.main(config)

