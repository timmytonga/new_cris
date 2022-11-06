import wandb
from default_sweep_config import get_default_sweep_config
from sweep_args import get_args

PROJECT_NAME = 'ReducedVal'
dataset_name = 'multinli'

args = get_args()

# Example sweep configuration
sweep_configuration = get_default_sweep_config(dataset_name, PROJECT_NAME, args)

sweep_configuration['name'] = f'Seed {args.seed} p={args.p} more lr and wd'
sweep_configuration['parameters'] = {
    'part2_batch_size':
        {'values': [ 32]},
    'part2_lr':
        {'values': ['5e-3' ,'5e-4', '5e-5', '1e-6']},
    'part2_wd':
        {'values': [1, '1e-1', 0]},
    'reduce_val_fraction':
        {'value': 0.05}
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project=f'{PROJECT_NAME}_{dataset_name}')
