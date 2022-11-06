import wandb
from default_sweep_config import get_default_sweep_config
from sweep_args import get_args

PROJECT_NAME = 'ReducedVal'
dataset_name = 'cub'

args = get_args()

# Example sweep configuration
sweep_configuration = get_default_sweep_config(dataset_name, PROJECT_NAME, args)

sweep_configuration['parameters']['part2_batch_size']['values'] = [4, 8]
sweep_configuration['parameters']['part2_wd']['values'] = [1e-1, 1, 10]
sweep_configuration['parameters']['part2_lr']['values'] = [1e-5, 5e-5, 1e-4 ,5e-4]
sweep_configuration['parameters']['part2_n_epochs'] = {'value': 101}
sweep_configuration['name'] = f'Seed {args.seed} p={args.p} high wd and early stopping'

sweep_id = wandb.sweep(sweep=sweep_configuration, project=f'{PROJECT_NAME}_{dataset_name}')
