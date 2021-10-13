"""
    Here we check the effects of different propotion and parameters
"""
from notebooks.example_args import MyCUBArgs
import my_run_expt
import os

WANDB = True
SEED = 0
SHOW_PROGRESS = True
part1_LR, part1_WD = 1e-4, 1e-4
part2_LR, part2_WD = 1e-4, 1e-4
n_epochs_p1, n_epochs_p2 = 301, 51
PART2_ONLY_LAST_LAYER = True
PART2_LOSS_TYPE = "group_dro"  # choices ["erm", "group_dro", "join_dro"]
project_name = "splitpgl2"

RUN_PART1 = False

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
part2_args.part1_model_epoch = part1_args.n_epochs-1  # last model from part1
part2_args.part2_only_last_layer = PART2_ONLY_LAST_LAYER
part2_args.loss_type = PART2_LOSS_TYPE

for p in [0.5, 0.75, 0.85]:
    part1_args.part1_split_proportion = p
    part2_args.part1_split_proportion = p
    root_log = os.path.join(mainargs.root_log, f"p{p}_wd{part1_WD}_lr{part1_LR}")
    part1_args.log_dir = os.path.join(root_log, f"part1")
    part2_args.log_dir = os.path.join(root_log, f"part2_{PART2_LOSS_TYPE}_p{p}_wd{part2_WD}_lr{part2_LR}")
    # # run part1
    if RUN_PART1:  # ensure we have already run part 1 if this is set to False
        my_run_expt.main(part1_args)
    # now run part2
    my_run_expt.main(part2_args)
