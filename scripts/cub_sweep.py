"""
    Here we check the effects of different propotion and parameters
"""
from notebooks.example_args import MyCUBArgs, set_args_and_run_sweep, set_two_parts_args

args = set_two_parts_args(seed=0,
                          p=[0.3, 0.5, 0.7],
                          gpu=0,
                          part1_wd=1e-4,
                          part1_lr=1e-4,
                          part1_n_epochs=301,
                          part2_lr=1e-4,
                          part2_n_epochs=301)

######### SET ARGS HERE ########
# misc args
WANDB = not args.no_wandb
SEED = args.seed
SHOW_PROGRESS = False

# part1 args
RUN_PART1 = args.run_part1
part1_LR, part1_WD = args.part1_lr, args.part1_wd
n_epochs_p1 = args.part1_n_epochs

# part2 args
RUN_PART2 = not args.no_part2
n_epochs_p2 = args.part2_n_epochs
DEFAULT_PART1_MODEL_EPOCHS = [n_epochs_p1 - 1]
part2_LR, part2_WD = args.part2_lr, 1e-4
PART2_ONLY_LAST_LAYER = True
USE_REAL_GROUP_LABELS = True
PART2_REWEIGHT = args.part2_reweight  # this only works with use_real_group_labels

# TAU_NORM_ARGS
RUN_TAU_NORM = False
MIN_TAU, MAX_TAU, TAU_STEP = 1.0, 10.0, 101

project_name = "Rgl" if USE_REAL_GROUP_LABELS else "Pgl"
################################

# initialize args
mainargs = MyCUBArgs(wandb=WANDB,
                     seed=SEED,
                     show_progress=SHOW_PROGRESS,
                     project_name=project_name,
                     gpu=args.gpu,
                     part1_save_every=args.part1_save_every,
                     part1_use_all_data=args.part1_use_all_data)

# run with args
set_args_and_run_sweep(mainargs, args,
                       part1_LR, part1_WD,
                       n_epochs_p1, n_epochs_p2,
                       part2_LR, part2_WD,
                       PART2_ONLY_LAST_LAYER, USE_REAL_GROUP_LABELS,
                       PART2_REWEIGHT, MIN_TAU, MAX_TAU, TAU_STEP,
                       DEFAULT_PART1_MODEL_EPOCHS, RUN_PART1,
                       RUN_PART2, RUN_TAU_NORM, SEED)
