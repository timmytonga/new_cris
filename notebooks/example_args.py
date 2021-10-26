import os


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class MyCelebaArgs:
    def __init__(self, n_epochs=51, wd=1e-5, lr=1e-5,
                 upweight=0, run_name='celebA_run', project_name='splitpgl',
                 only_last_layer=True, seed=0, wandb=True, show_progress=True,
                 split_proportion=0.5, gpu=0, part1_save_every=10):
        self.upweight = upweight
        self.only_last_layer = only_last_layer
        self.root_log = f"/home/thiennguyen/research/pseudogroups/CelebA/splitpgl_sweep_logs"
        self.ROOT_LOG = os.path.join(self.root_log, f"/SPGL_proportion{split_proportion}_epochs{n_epochs}_lr{lr}_weightdecay{wd}")
        BATCH_SIZE = 64
        self.part1_args = DotDict({
            "q": 0.7,
            "lr": lr,
            "btl": False,
            "gpu": gpu,
            "fold": None,
            "seed": seed,
            "alpha": 0.2,
            "gamma": 0.1,
            "hinge": False,
            "model": "resnet50",
            "wandb": wandb,
            "resume": False,
            "aug_col": "None",
            "dataset": "CelebA",
            "log_dir": f"{self.ROOT_LOG}/part1",
            "fraction": 1,
            "n_epochs": n_epochs,
            "root_dir": "/home/thiennguyen/research/datasets/",
            "log_every": 50,
            "loss_type": "erm",
            "save_best": False,
            "save_last": False,
            "save_step": part1_save_every,
            "scheduler": False,
            # "up_weight": 0,
            "batch_size": BATCH_SIZE,
            "num_sweeps": 4,
            "shift_type": "confounder",
            "target_name": "Blond_Hair",
            "augment_data": False,
            "project_name": project_name,
            "val_fraction": 0.1,
            "weight_decay": wd,
            "subsample_minority": False,
            "metadata_path": f"myresults/celebA/{run_name}/metadata_aug.csv",
            "show_progress": show_progress,
            "imbalance_ratio": None,
            "joint_dro_alpha": 1,
            "reweight_groups": False,
            "use_bert_params": 0,
            "confounder_names": ["Male"],
            "robust_step_size": 0.01,
            "metadata_csv_name": "list_attr_celeba.csv",
            "minority_fraction": None,
            "train_from_scratch": False,
            "num_folds_per_sweep": 5,
            "use_normalized_loss": False,
            "automatic_adjustment": False,
            "generalization_adjustment": "0.0",
            "minimum_variational_weight": 0,
            "part": 1,
            "part1_split_proportion": split_proportion,
            "part1_model_epoch": 10,
            "part2_only_last_layer": False,
            "part2_use_old_model": False,
            "upweight": 0
        })

        self.part2_args = DotDict({
            "q": 0.7,
            "lr": lr,
            "btl": False,
            "gpu": gpu,
            "fold": None,
            "seed": seed,
            "alpha": 0.2,
            "gamma": 0.1,
            "hinge": False,
            "model": "resnet50",
            "wandb": wandb,
            "resume": False,
            "aug_col": "None",
            "dataset": "CelebA",
            "log_dir": f"{self.ROOT_LOG}/part1",
            "fraction": 1,
            "n_epochs": n_epochs,
            "root_dir": "/home/thiennguyen/research/datasets/",
            "log_every": 50,
            "loss_type": "erm",
            "save_best": False,
            "save_last": False,
            "save_step": 10,
            "scheduler": False,
            # "up_weight": 0,
            "batch_size": BATCH_SIZE,
            "num_sweeps": 4,
            "shift_type": "confounder",
            "target_name": "Blond_Hair",
            "augment_data": False,
            "project_name": project_name,
            "val_fraction": 0.1,
            "weight_decay": wd,
            "subsample_minority": False,
            "metadata_path": f"myresults/celebA/{run_name}/metadata_aug.csv",
            "show_progress": show_progress,
            "imbalance_ratio": None,
            "joint_dro_alpha": 1,
            "reweight_groups": False,
            "use_bert_params": 0,
            "confounder_names": ["Male"],
            "robust_step_size": 0.01,
            "metadata_csv_name": "list_attr_celeba.csv",
            "minority_fraction": None,
            "train_from_scratch": False,
            "num_folds_per_sweep": 5,
            "use_normalized_loss": False,
            "automatic_adjustment": False,
            "generalization_adjustment": "0.0",
            "minimum_variational_weight": 0,
            "part": 2,
            "part1_split_proportion": split_proportion,
            "part1_model_epoch": 50,
            "part2_only_last_layer": True,
            "part2_use_old_model": True,
            "upweight": 0
        })

class MyCUBArgs:
    def __init__(self, n_epochs=51, wd=1e-4, lr=1e-3,
                 upweight=0, run_name='waterbird_newrun', project_name='splitpgl',
                 only_last_layer=True, seed=0, wandb=True, show_progress=True,
                 split_proportion=0.5):
        self.upweight = upweight
        self.only_last_layer = only_last_layer
        self.root_log = f"/home/thiennguyen/research/pseudogroups/CUB/splitpgl_sweep_logs"
        self.ROOT_LOG = os.path.join(self.root_log, f"/SPGL_proportion{split_proportion}_epochs{n_epochs}_lr{lr}_weightdecay{wd}")
        BATCH_SIZE = 64
        self.part1_args = DotDict({
            "q": 0.7,
            "lr": lr,
            "btl": False,
            "gpu": 0,
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
            "log_dir": f"{self.ROOT_LOG}/part1",
            "fraction": 1,
            "n_epochs": n_epochs,
            "root_dir": "./cub",
            "log_every": 50,
            "loss_type": "erm",
            "save_best": False,
            "save_last": False,
            "save_step": 10,
            "scheduler": False,
            # "up_weight": 0,
            "batch_size": BATCH_SIZE,
            "num_sweeps": 4,
            "shift_type": "confounder",
            "target_name": "waterbird_complete95",
            "augment_data": False,
            "project_name": project_name,
            "val_fraction": 0.1,
            "weight_decay": wd,
            "subsample_minority": False,
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
            "minimum_variational_weight": 0,
            "part": 1,
            "part1_split_proportion": split_proportion,
            "part1_model_epoch": 10,
            "part2_only_last_layer": False,
            "part2_use_old_model": False,
            "upweight": 0
        })

        self.part2_args = DotDict({
            "q": 0.7,
            "lr": lr,
            "btl": False,
            "gpu": 0,
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
            "log_dir": f"{self.ROOT_LOG}/part2",
            "fraction": 1,
            "n_epochs": n_epochs,
            "root_dir": "./cub",
            "log_every": 50,
            "loss_type": "erm",
            "save_best": False,
            "save_last": False,
            "save_step": 10,
            "scheduler": False,
            # "up_weight": self.upweight,
            "batch_size": BATCH_SIZE,
            "num_sweeps": 4,
            "shift_type": "confounder",
            "target_name": "waterbird_complete95",
            "augment_data": False,
            "only_last_layer": self.only_last_layer,
            "project_name": project_name,
            "val_fraction": 0.1,
            "weight_decay": wd,
            "subsample_minority": False,
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
            "minimum_variational_weight": 0,
            "part": 2,
            "part1_split_proportion": split_proportion,
            "part1_model_epoch": 50,
            "part2_only_last_layer": True,
            "part2_use_old_model": True,
            "upweight": 0
        })

    def set_param_both(self, param, value):
        self.part2_args[param] = value
        self.part1_args[param] = value
        # need to update log dir
        if param in ['n_epochs', 'weight_decay', 'lr']:
            N_EPOCHS = self.part1_args['n_epochs']
            LR = self.part1_args['lr']
            WEIGHT_DECAY = self.part1_args['weight_decay']
            log_root = os.path.join(self.root_log, f"ERM_upweight_0_epochs_{N_EPOCHS}_lr_{LR}_weight_decay_{WEIGHT_DECAY}")
            self.part1_args['log_dir'] = os.path.join(self.root_log, "model_outputs")
            self.part2_args['log_dir'] = os.path.join(log_root,
                                                            f"{'retrain' if self.only_last_layer else 'last_layer'}_part2_upweight{self.upweight}")

    def set_param_retrain(self, param, value):
        self.part2_args[param] = value

    def set_param_naive(self, param, value):
        self.part1_args[param] = value
        # need to update log dir
        if param in ['n_epochs', 'weight_decay', 'lr']:
            N_EPOCHS = self.part1_args['n_epochs']
            LR = self.part1_args['lr']
            WEIGHT_DECAY = self.part1_args['weight_decay']
            log_root = os.path.join(self.root_log,
                                    f"ERM_upweight_0_epochs_{N_EPOCHS}_lr_{LR}_weight_decay_{WEIGHT_DECAY}")
            self.part1_args['log_dir'] = os.path.join(self.root_log, "model_outputs")
            self.part2_args['log_dir'] = os.path.join(log_root,
                                                            f"{'retrain' if self.only_last_layer else 'last_layer'}_part2_upweight{self.upweight}")


test_args = MyCUBArgs()
part1_args = test_args.part1_args
part2_args = test_args.part2_args
