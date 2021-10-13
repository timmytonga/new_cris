"""
    Here we check the effects of different propotion and parameters
"""
from notebooks.example_args import MyCelebaArgs
import my_run_expt
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
# parser.add_argument("--wandb", action='store_false', default=0)
parser.add_argument("--gpu", type=int, default=1)

args = parser.parse_args()
######### SET ARGS HERE ########
WANDB = True
SEED = args.seed
SHOW_PROGRESS = True
part1_LR, part1_WD = 1e-5, 1e-5
part2_LR, part2_WD = 1e-5, 1e-5
n_epochs_p1, n_epochs_p2 = 51, 51
PART2_ONLY_LAST_LAYER = False
GPU = args.gpu
################################

# initialize args
mainargs = MyCelebaArgs(wandb=WANDB,
                        seed=SEED,
                        show_progress=SHOW_PROGRESS,
                        gpu=GPU)
part1_args = mainargs.part1_args
part2_args = mainargs.part2_args

# set some specific params
part1_args.lr, part1_args.weight_decay = part1_LR, part1_WD
part2_args.lr, part2_args.weight_decay = part2_LR, part2_WD
part1_args.n_epochs, part2_args.n_epochs = n_epochs_p1, n_epochs_p2
part2_args.part1_model_epoch = part1_args.n_epochs-1  # last model from part1
part2_args.part2_only_last_layer = PART2_ONLY_LAST_LAYER

for p in [0.5, 0.75]:
    part1_args.part1_split_proportion = p
    part2_args.part1_split_proportion = p
    root_log = os.path.join(mainargs.root_log, f"p{p}_wd{part1_WD}_lr{part1_LR}")
    part1_args.log_dir = os.path.join(root_log, f"part1")
    part2_args.log_dir = os.path.join(root_log, f"part2_p{p}_wd{part2_WD}_lr{part2_LR}")
    # # run part1
    my_run_expt.main(part1_args)
    # now run part2
    my_run_expt.main(part2_args)
