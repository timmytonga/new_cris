import wandb
from default_sweep_config import get_default_sweep_config
from sweep_args import get_args

PROJECT_NAME = 'ReducedVal'
dataset_name = 'cub'

args = get_args()

# Example sweep configuration
sweep_configuration = get_default_sweep_config(dataset_name, PROJECT_NAME, args)

sweep_configuration['parameters']['part2_batch_size']['values'] = [4, 8, 16]
sweep_id = wandb.sweep(sweep=sweep_configuration, project=f'{PROJECT_NAME}_{dataset_name}')
