"""
    Here we check the effects of different propotion and parameters
"""
from notebooks.example_args import MyCelebaArgs, set_args_and_run_sweep
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("-p", nargs="+", type=float, default=[0.3, 0.5, 0.7])
parser.add_argument("--no_wandb", action="store_true", default=False)
parser.add_argument("--gpu", type=int, default=1)

# part1 args
parser.add_argument("--part1_wd", type=float, default=1e-4)
parser.add_argument("--part1_loss_type", default="erm",
                    choices=["erm", "group_dro", "joint_dro"])
parser.add_argument("--part1_reweight", action="store_true", default=False)
parser.add_argument("--run_part1", action="store_true", default=False)
parser.add_argument("--part1_use_all_data", action="store_true", default=False)

# part2 args
parser.add_argument("--part1_model_epochs", nargs="+", type=int, default=None)
parser.add_argument("--part2_loss_type", default="erm",
                    choices=["erm", "group_dro", "joint_dro"])
parser.add_argument("--part2_subsample", action="store_true", default=False)
parser.add_argument("--part2_reweight", action="store_true", default=False)
parser.add_argument("--part2_lr", type=float, default=1e-4)
parser.add_argument("--part2_n_epochs", type=int, default=51)
parser.add_argument("--no_part2", action="store_true", default=False)

parser.add_argument("--tau_norm_after_part2", action="store_true", default=False)

args = parser.parse_args()

######### SET ARGS HERE ########
# misc args
WANDB = not args.no_wandb
SEED = args.seed
SHOW_PROGRESS = True

project_name = "splitpgl2"

# part1 args
RUN_PART1 = args.run_part1
part1_LR, part1_WD = 1e-4, args.part1_wd
n_epochs_p1 = 51
PART1_SAVE_EVERY = 1

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

################################

# initialize args
mainargs = MyCelebaArgs(wandb=WANDB,
                        seed=SEED,
                        show_progress=SHOW_PROGRESS,
                        project_name=project_name,
                        part1_save_every=PART1_SAVE_EVERY,
                        gpu=args.gpu,
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
