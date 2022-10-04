def get_default_sweep_config(dataset_name, seed, project_name):
    return {'program': f'scripts/{dataset_name}_sweep.py',
            'name': f'Seed {seed}',
            'project': f'{project_name}_{dataset_name}',
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
                        f'--seed={seed}',
                        '--part2_loss_type=group_dro',
                        '--part2_save_best'],
            'parameters': {
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
