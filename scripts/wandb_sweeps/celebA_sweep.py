import wandb
import argparse
from default_sweep_config import get_default_sweep_config


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)

args = parser.parse_args()

PROJECT_NAME = 'ReducedVal'
dataset_name = 'celebA'

# Example sweep configuration
sweep_configuration = get_default_sweep_config(dataset_name, args.seed, PROJECT_NAME)

sweep_id = wandb.sweep(sweep=sweep_configuration, project=f'{PROJECT_NAME}_{dataset_name}')
