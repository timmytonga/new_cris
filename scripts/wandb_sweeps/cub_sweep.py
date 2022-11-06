import wandb
from default_sweep_config import get_default_sweep_config
from sweep_args import get_args

PROJECT_NAME = 'ReducedVal'
dataset_name = 'cub'

args = get_args()

# Example sweep configuration
sweep_configuration = get_default_sweep_config(dataset_name, PROJECT_NAME, args)
sweep_configuration['parameters']['reduce_val_fraction']['value'] = args.reduce_val_fraction

sweep_configuration['parameters']['part2_batch_size']['values'] = [8, 16]
sweep_configuration['parameters']['part2_wd']['values'] = [1e-1, 1, 0]
sweep_configuration['parameters']['part2_lr']['values'] = [1e-5, 1e-4, 1e-3]
sweep_configuration['parameters']['part2_n_epochs'] = {'value': 201}

sweep_configuration['name'] = f"RVF={args.reduce_val_fraction} Seed {args.seed} p={args.p}"

sweep_id = wandb.sweep(sweep=sweep_configuration, project=f'{PROJECT_NAME}_{dataset_name}')
