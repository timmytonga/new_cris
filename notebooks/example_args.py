import os


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class MyCUBArgs:
    def __init__(self, n_epochs=51, wd=1e-4, lr=1e-3,
                 upweight=10, run_name='waterbird_newrun',
                 only_last_layer=True, seed=0, wandb=True, show_progress=True):
        self.upweight = upweight
        self.only_last_layer = only_last_layer
        self.root_log = f"myresults/CUB/{run_name}"
        self.ROOT_LOG =  self.root_log + f"/ERM_upweight_0_epochs_{n_epochs}_lr_{lr}_weight_decay_{wd}"
        self.args_naive_ERM_CUB = DotDict({
            "q": 0.7,
            "lr": lr,
            "btl": False,
            "fold": None,
            "seed": seed,
            "alpha": 0.2,
            "gamma": 0.1,
            "hinge": False,
            "model": "resnet50",
            "wandb": wandb,
            "resume": False,
            "aug_col": "None",
            "dataset": "CUB",
            "log_dir": f"{self.ROOT_LOG}/model_outputs",
            "fraction": 1,
            "n_epochs": n_epochs,
            "root_dir": "./cub",
            "log_every": 50,
            "loss_type": "erm",
            "save_best": False,
            "save_last": False,
            "save_step": 10,
            "scheduler": False,
            "up_weight": 0,
            "batch_size": 64,
            "num_sweeps": 4,
            "shift_type": "confounder",
            "target_name": "waterbird_complete95",
            "augment_data": False,
            "project_name": "spurious",
            "val_fraction": 0.1,
            "weight_decay": wd,
            "metadata_path": "myresults/CUB/waterbird_firstrun/metadata_aug.csv",
            "show_progress": show_progress,
            "imbalance_ratio": None,
            "joint_dro_alpha": 1,
            "reweight_groups": False,
            "use_bert_params": 0,
            "confounder_names": ["forest2water2"],
            "robust_step_size": 0.01,
            "metadata_csv_name": "metadata.csv",
            "minority_fraction": None,
            "train_from_scratch": False,
            "num_folds_per_sweep": 5,
            "use_normalized_loss": False,
            "automatic_adjustment": False,
            "generalization_adjustment": "0.0",
            "minimum_variational_weight": 0
        })

        self.args_retrain_CUB = DotDict({
            "q": 0.7,
            "lr": lr,
            "btl": False,
            "fold": None,
            "seed": seed,
            "alpha": 0.2,
            "gamma": 0.1,
            "hinge": False,
            "model": "resnet50",
            "wandb": wandb,
            "resume": False,
            "aug_col": "None",
            "dataset": "CUB",
            "log_dir": f"{self.ROOT_LOG}/{'retrain' if self.only_last_layer else 'last_layer'}_part2_upweight{self.upweight}",
            "fraction": 1,
            "n_epochs": n_epochs,
            "root_dir": "./cub",
            "log_every": 50,
            "loss_type": "erm",
            "save_best": False,
            "save_last": False,
            "save_step": 10,
            "scheduler": False,
            "up_weight": self.upweight,
            "batch_size": 64,
            "num_sweeps": 4,
            "shift_type": "confounder",
            "target_name": "waterbird_complete95",
            "augment_data": False,
            "only_last_layer": self.only_last_layer,
            "project_name": "spurious",
            "val_fraction": 0.1,
            "weight_decay": 0.0001,
            "metadata_path": f"myresults/CUB/{run_name}/metadata_aug.csv",
            "show_progress": show_progress,
            "imbalance_ratio": None,
            "joint_dro_alpha": 1,
            "reweight_groups": False,
            "use_bert_params": 0,
            "confounder_names": ["forest2water2"],
            "robust_step_size": 0.01,
            "metadata_csv_name": "metadata.csv",
            "minority_fraction": None,
            "train_from_scratch": False,
            "num_folds_per_sweep": 5,
            "use_normalized_loss": False,
            "automatic_adjustment": False,
            "generalization_adjustment": "0.0",
            "minimum_variational_weight": 0
        })

    def set_param_both(self, param, value):
        self.args_retrain_CUB[param] = value
        self.args_naive_ERM_CUB[param] = value
        # need to update log dir
        if param in ['n_epochs', 'weight_decay', 'lr']:
            N_EPOCHS = self.args_naive_ERM_CUB['n_epochs']
            LR = self.args_naive_ERM_CUB['lr']
            WEIGHT_DECAY = self.args_naive_ERM_CUB['weight_decay']
            log_root = os.path.join(self.root_log, f"ERM_upweight_0_epochs_{N_EPOCHS}_lr_{LR}_weight_decay_{WEIGHT_DECAY}")
            self.args_naive_ERM_CUB['log_dir'] = os.path.join(self.root_log, "model_outputs")
            self.args_retrain_CUB['log_dir'] = os.path.join(log_root,
                                                            f"{'retrain' if self.only_last_layer else 'last_layer'}_part2_upweight{self.upweight}")

    def set_param_retrain(self, param, value):
        self.args_retrain_CUB[param] = value

    def set_param_naive(self, param, value):
        self.args_naive_ERM_CUB[param] = value
        # need to update log dir
        if param in ['n_epochs', 'weight_decay', 'lr']:
            N_EPOCHS = self.args_naive_ERM_CUB['n_epochs']
            LR = self.args_naive_ERM_CUB['lr']
            WEIGHT_DECAY = self.args_naive_ERM_CUB['weight_decay']
            log_root = os.path.join(self.root_log,
                                    f"ERM_upweight_0_epochs_{N_EPOCHS}_lr_{LR}_weight_decay_{WEIGHT_DECAY}")
            self.args_naive_ERM_CUB['log_dir'] = os.path.join(self.root_log, "model_outputs")
            self.args_retrain_CUB['log_dir'] = os.path.join(log_root,
                                                            f"{'retrain' if self.only_last_layer else 'last_layer'}_part2_upweight{self.upweight}")


test_args = MyCUBArgs()
args_naive_ERM_CUB = test_args.args_naive_ERM_CUB
args_retrain_CUB = test_args.args_retrain_CUB
