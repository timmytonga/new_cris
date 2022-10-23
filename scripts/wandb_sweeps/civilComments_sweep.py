import wandb
from default_sweep_config import get_default_sweep_config
from sweep_args import get_args

PROJECT_NAME = 'ReducedVal'
dataset_name = 'civilComments'

args = get_args()

# Example sweep configuration
sweep_configuration = get_default_sweep_config(dataset_name, PROJECT_NAME, args)
sweep_configuration['parameters'] = {
    'jigsaw_use_group':
        {'values': ['any_identity']},
    'part2_batch_size':
        {'values': [24, 16]},
    'part2_lr':
        {'values': ['1e-2', '1e-3', '1e-4', '1e-5']},
    'part2_wd':
        {'values': ['1e-1', '1e-4', 0]},
    'reduce_val_fraction':
        {'value': 0.05}
}
sweep_configuration['name'] = f'Seed {args.seed} p={args.p} new fixed val'

sweep_id = wandb.sweep(sweep=sweep_configuration, project=f'{PROJECT_NAME}_{dataset_name}')
