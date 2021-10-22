"""
    Here we check the effects of different propotion and parameters
"""
from notebooks.example_args import MyCUBArgs
import my_run_expt
import os
from copy import deepcopy

WANDB = True
SEED = 1
SHOW_PROGRESS = True
n_epochs_p1, n_epochs_p2 = 301, 301
project_name = "splitpgl2"

# part1 args
RUN_PART1 = False  # this means we have already trained part1 and we want to reuse this part1
RUN_TAU_NORM = True

part1_LR, part1_WD = 1e-4, 1e-4

# part2 args
part2_LR, part2_WD = 1e-4, 1e-5
PART2_ONLY_LAST_LAYER = True
PART2_LOSS_TYPE = "erm"  # choices ["erm", "group_dro", "join_dro"]
USE_REAL_GROUP_LABELS = True
PART2_REWEIGHT = True  # this only works with use_real_group_labels

# initialize args
mainargs = MyCUBArgs(wandb=WANDB,
                     seed=SEED,
                     show_progress=SHOW_PROGRESS,
                     project_name=project_name)  # default gpu = 0
part1_args = mainargs.part1_args
part2_args = mainargs.part2_args

# set some specific params
part1_args.lr, part1_args.weight_decay = part1_LR, part1_WD
part2_args.lr, part2_args.weight_decay = part2_LR, part2_WD
part1_args.n_epochs, part2_args.n_epochs = n_epochs_p1, n_epochs_p2

# part 2 args
part2_args.part1_model_epoch = part1_args.n_epochs-1  # last model from part1
part2_args.part2_only_last_layer = PART2_ONLY_LAST_LAYER
part2_args.loss_type = PART2_LOSS_TYPE
part2_args.use_real_group_labels = USE_REAL_GROUP_LABELS
part2_args.reweight_groups = PART2_REWEIGHT

# tau norm args
tau_norm_args = deepcopy(part2_args)
tau_norm_args.model_epoch = part2_args.n_epochs - 1
tau_norm_args.min_tau, tau_norm_args.max_tau, tau_norm_args.step = 0.0, 5.0, 51
tau_norm_args.run_test = True

for p in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
# for p in [1]:
    part1_args.part1_split_proportion = p
    part2_args.part1_split_proportion = p
    root_log = os.path.join(mainargs.root_log, f"p{p}_wd{part1_WD}_lr{part1_LR}_s{SEED}")
    part1_args.log_dir = os.path.join(root_log, f"part1")
    part2_args.log_dir = os.path.join(root_log, f"part2_{PART2_LOSS_TYPE}_p{p}_wd{part2_WD}_lr{part2_LR}")
    tau_norm_args.log_dir = os.path.join(part2_args.log_dir, "tau_norm")
    # # run part1
    if RUN_PART1:  # ensure we have already run part 1 if this is set to False
        my_run_expt.main(part1_args)
    # now run part2
    my_run_expt.main(part2_args)
