"""
    Here we check the effects of different propotion and parameters
"""
from notebooks.example_args import MyCUBArgs, DotDict
import my_run_expt
import os
import argparse
import tau_norm


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("-p", nargs="+", type=float, default=[0.3, 0.5, 0.7])
parser.add_argument("--no_wandb", action="store_true", default=False)

# part1 args
parser.add_argument("--part1_wd", type=float, default=1e-4)
parser.add_argument("--part1_loss_type", default="erm",
                    choices=["erm", "group_dro", "joint_dro"])
parser.add_argument("--run_part1", action="store_true", default=False)

# part2 args
parser.add_argument("--part1_model_epochs", nargs="+", type=int, default=None)
parser.add_argument("--part2_loss_type", default="erm",
                    choices=["erm", "group_dro", "joint_dro"])
parser.add_argument("--part2_subsample", action="store_true", default=False)
parser.add_argument("--part2_reweight", action="store_true", default=False)

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
n_epochs_p1 = 301

# part2 args
RUN_PART2 = True
n_epochs_p2 = 301
DEFAULT_PART1_MODEL_EPOCHS = [n_epochs_p1-1]
part2_LR, part2_WD = 1e-4, 1e-4
PART2_ONLY_LAST_LAYER = True
USE_REAL_GROUP_LABELS = True
PART2_REWEIGHT = args.part2_reweight  # this only works with use_real_group_labels

# TAU_NORM_ARGS
RUN_TAU_NORM = False
MIN_TAU, MAX_TAU, TAU_STEP = 1.0, 10.0, 101

################################

# initialize args
mainargs = MyCUBArgs(wandb=WANDB,
                     seed=SEED,
                     show_progress=SHOW_PROGRESS,
                     project_name=project_name)  # default gpu = 0
part1_args = mainargs.part1_args
part2_args = mainargs.part2_args

# set some specific params
part1_args.loss_type = args.part1_loss_type
part1_args.lr, part1_args.weight_decay = part1_LR, part1_WD
part2_args.lr, part2_args.weight_decay = part2_LR, part2_WD
part1_args.n_epochs, part2_args.n_epochs = n_epochs_p1, n_epochs_p2

# part 2 args
part2_args.part2_only_last_layer = PART2_ONLY_LAST_LAYER
part2_args.use_real_group_labels = USE_REAL_GROUP_LABELS
part2_args.loss_type = args.part2_loss_type
part2_args.reweight_groups = False if args.part2_loss_type == "group_dro" else PART2_REWEIGHT
part2_args.subsample_minority = args.part2_subsample
# some log dir
oll = "oll" if PART2_ONLY_LAST_LAYER else "full"

# tau norm args
tau_norm_args = DotDict(part2_args.copy())
tau_norm_args['model_epoch'] = part2_args.n_epochs - 1
tau_norm_args['min_tau'], tau_norm_args['max_tau'], tau_norm_args['step'] = MIN_TAU, MAX_TAU, TAU_STEP
tau_norm_args['run_test'] = True

if args.part1_model_epochs is None:
    p1me = DEFAULT_PART1_MODEL_EPOCHS
    p1me.append(part1_args.n_epochs-1)
else:
    p1me = args.part1_model_epochs

print(f"Run with p1me {p1me} and p {args.p}")
for part1_model_epoch in p1me:
    part2_args.part1_model_epoch = part1_model_epoch
    for p in args.p:
        print(f"Running p={p} and p1me={part1_model_epoch}")
        part1_args.part1_split_proportion = p
        part2_args.part1_split_proportion = p
        tau_norm_args.part1_split_proportion = p
        root_log = os.path.join(mainargs.root_log, f"p{p}_wd{part1_WD}_lr{part1_LR}_s{SEED}")
        part1_args.log_dir = os.path.join(root_log, f"part1")
        part2_args.log_dir = os.path.join(root_log, f"part2_{oll}{part1_model_epoch}_"
                                                    f"{args.part2_loss_type}_p{p}_wd{part2_WD}_lr{part2_LR}")
        tau_norm_args.log_dir = os.path.join(part2_args.log_dir, "tau_norm")
        # # run part1
        if RUN_PART1:  # ensure we have already run part 1 if this is set to False
            my_run_expt.main(part1_args)
        # now run part2
        if RUN_PART2:
            my_run_expt.main(part2_args)
        if RUN_TAU_NORM:
            tau_norm.main(tau_norm_args)
