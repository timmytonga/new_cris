import wandb
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
args = parser.parse_args()

PROJECT_NAME = 'ReducedVal'
dataset_name = 'civilComments'


# Example sweep configuration
sweep_configuration = {'program': f'scripts/{dataset_name}_sweep.py',
                       'name': f'Seed {args.seed}',
                       'project': f'{PROJECT_NAME}_{dataset_name}',
                       'description': 'Grid search with prechosen values',
                       'method': 'grid',
                       'metric': {'goal': 'maximize', 'name': 'test/wg_acc'},
                       'command': ['${env}',
                                   '${interpreter}',
                                   '${program}',
                                   '${args}',
                                   '--val_split',
                                   '--part1_save_best',
                                   '--run_test',
                                   '--part2_reweight',
                                   '--part1_model_epochs=-1',
                                   '-p=1',
                                   f'--seed={args.seed}',
                                   '--part2_loss_type=group_dro',
                                   '--part2_save_best'],
                       'parameters': {
                           'jigsaw_use_group':
                               {'values': ['black', 'any_identity']},
                           'part2_batch_size':
                               {'values': [8, 16, 32]},
                           'part2_lr':
                               {'values': ['1e-2', '1e-3', '1e-4', '1e-5']},
                           'part2_wd':
                               {'values': ['1e-1', '1e-2', '1e-3', 0]},
                           'reduce_val_fraction':
                               {'value': 0.05}
                       }
                       }
sweep_id = wandb.sweep(sweep=sweep_configuration, project=f'{PROJECT_NAME}_{dataset_name}')
