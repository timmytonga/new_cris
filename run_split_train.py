import os, csv
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
import numpy as np
import wandb
from copy import deepcopy

from models import model_attributes
from data.data import dataset_attributes, shift_types, prepare_data, log_data
from data import dro_dataset
from data import folds
from utils import set_seed, Logger, CSVBatchLogger, log_args, get_model, hinge_loss, my_split_data, check_args
from train import train, run_epoch
from loss import LossComputer
import pandas as pd
from data.folds import ConcatDataset


RUN_TEST = False  # do we want to run test (turn this on later when developing method is finished)
ROOT_LOG_DIR = "."

"""
    Steps for running split train data:
    - Use the same log dir for both programs --> they are nested in the initial model's training param. 
    - The initial model training depends on the part1 and part2 split. Save the splits in the log dir.
    - Write run_expt so that it trains using a specific path for either part1 or part2. 
        - Can use this to retrain and upweight the model for part2. 
"""


def main(args):
    ############################################################
    #       First partition the data into two parts            #
    ############################################################
    set_seed(args.seed)
    ## Function given dataset and fraction f (is this implemented already?)
    # first load the train set
    train_data, val_data, test_data = prepare_data(args, train=True)
    # then split it into part1 containing f*n examples of trainset and part2 containing the rest
    part1, part2 = my_split_data(train_data.dataset, part1_split_fraction=0.5, seed=args.seed)
    part1_data = dro_dataset.DRODataset(
        part1,
        process_item_fn=None,
        n_groups=train_data.n_groups,
        n_classes=train_data.n_classes,
        group_str_fn=train_data.group_str)

    part2_data = dro_dataset.DRODataset(
        part2,
        process_item_fn=None,
        n_groups=train_data.n_groups,
        n_classes=train_data.n_classes,
        group_str_fn=train_data.group_str)

    ############################################################
    #       Now train the initial ERM model on part 1          #
    ############################################################
    if args.wandb:
        run = wandb.init(project=f"{args.project_name}_{args.dataset}")
        wandb.config.update(args)

    ## Initialize logs
    if os.path.exists(args.log_dir) and args.resume:
        resume = True
        mode = "a"
    else:
        resume = False
        mode = "w"

    logger = Logger(os.path.join(args.log_dir, "log.txt"), 'w')
    # Record args
    log_args(args, logger)
    # set_seed(args.seed)

    # get loader and run train ERM
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": 4,
        "pin_memory": True,
    }
    part1_loader = dro_dataset.get_loader(part1_data,
                                          train=True,
                                          reweight_groups=None,
                                          **loader_kwargs)
    val_loader = dro_dataset.get_loader(val_data,
                                        train=False,
                                        reweight_groups=None,
                                        **loader_kwargs)

    data = {}
    data["train_data"] = part1_data
    data["train_loader"] = part1_loader  # we train using  part1
    data["val_data"] = val_data
    data["val_loader"] = val_loader

    data["test_data"] = test_data
    data["test_loader"] = None

    train_csv_logger = CSVBatchLogger(os.path.join(args.log_dir, f"train.csv"),
                                      train_data.n_groups,
                                      mode=mode)
    val_csv_logger = CSVBatchLogger(os.path.join(args.log_dir, f"val.csv"),
                                    val_data.n_groups,
                                    mode=mode)
    test_csv_logger = None

    if RUN_TEST:
        test_loader = dro_dataset.get_loader(test_data,
                                             train=False,
                                             reweight_groups=None,
                                             **loader_kwargs)
        data["test_loader"] = test_loader
        test_csv_logger = CSVBatchLogger(os.path.join(args.log_dir, f"test.csv"),
                                         test_data.n_groups,
                                         mode=mode)

    n_classes = train_data.n_classes

    log_data(data, logger)
    logger.flush()

    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    ### INITIALIZE MODEL AND TRAIN ###
    resume = False  # not resuming yet.
    ## Initialize model
    model = get_model(
        model=args.model,
        pretrained=not args.train_from_scratch,
        resume=resume,
        n_classes=train_data.n_classes,
        dataset=args.dataset,
        log_dir=args.log_dir,
    )
    if args.wandb:
        wandb.watch(model)

    epoch_offset = 0

    train(
        model,
        criterion,
        data,
        logger,
        train_csv_logger,
        val_csv_logger,
        test_csv_logger,
        args,
        epoch_offset=epoch_offset,
        csv_name=args.fold,
        wandb=wandb if args.wandb else None,
    )

    train_csv_logger.close()
    val_csv_logger.close()
    if RUN_TEST:
        test_csv_logger.close()

    ############################################################
    #       Now evaluate the initial ERM model on part2        #
    ############################################################
    part2_log_dir = args.part2_log_dir
    if not os.path.exists(part2_log_dir):
        os.makedirs(part2_log_dir)
    part2_logger = Logger(os.path.join(part2_log_dir, "log.txt"), 'w')
    part2_logger.flush()

    # first load the previous model from args.log_dir i.e. log_dir for part1.
    model_path = os.path.join(args.log_dir, f"{args.n_epochs - 1}_model.pth")  # this should be from part1_log_dir
    modeleval = torch.load(model_path)
    # initialize logger and loader for part2
    part2eval_csv_logger = CSVBatchLogger(os.path.join(part2_log_dir, f"part2_eval.csv"),
                                          test_data.n_groups,
                                          mode=mode)
    part2_loader = dro_dataset.get_loader(part2_data,
                                          train=False,
                                          reweight_groups=None,
                                          **loader_kwargs)
    adjustments = np.array([float(c) for c in args.generalization_adjustment.split(",")] * 4)
    part2_loss_computer = LossComputer(
        criterion,
        loss_type=args.loss_type,
        dataset=train_data,
        alpha=args.alpha,
        gamma=args.gamma,
        adj=adjustments,
        step_size=args.robust_step_size,
        normalize_loss=args.use_normalized_loss,
        btl=args.btl,
        min_var_weight=args.minimum_variational_weight,
        joint_dro_alpha=args.joint_dro_alpha,
    )

    # then run an epoch on part2 and during that run, generate a csv containing the status of each example
    run_epoch(
        epoch=0,
        model=modeleval,
        optimizer=None,
        loader=part2_loader,
        loss_computer=part2_loss_computer,
        logger=part2_logger,
        csv_logger=part2eval_csv_logger,
        args=args,
        is_training=False,
        show_progress=True,
        log_every=50,
        scheduler=None,
        csv_name="pseudogroup_eval",
        wandb_group="part2_eval",
        wandb=wandb,
    )
    ############################################################
    #       Now we generate the pseudogroup labels             #
    ############################################################
    part2_df = pd.read_csv(os.path.join(part2_log_dir, 'output_part2_eval_epoch_0.csv'))
    misclassified = part2_df['y_pred_pseudogroup_eval_epoch_0_val'] != part2_df['y_true_pseudogroup_eval_epoch_0_val']
    aug_indices = part2_df['indices_pseudogroup_eval_epoch_0_val'][misclassified]
    upweight_factor = len(part2_df) // len(aug_indices) if args.upweight != -1 else args.upweight
    combined_indices = list(aug_indices) * upweight_factor + list(part2_df['indices_pseudogroup_eval_epoch_0_val'])
    upsampled_part2 = torch.utils.data.Subset(train_data.dataset.dataset, combined_indices)
    ############################################################
    #     Now we train a new model using pseudogroup labels    #
    ############################################################


if __name__ == "__main__":
    """
        TODO: rename args so that it reflects 2 parts of the pipeline
    """
    parser = argparse.ArgumentParser()
    # Settings
    parser.add_argument("-d",
                        "--dataset",
                        choices=dataset_attributes.keys(),
                        required=True)
    parser.add_argument("-s",
                        "--shift_type",
                        choices=shift_types,
                        required=True)

    parser.add_argument("--wandb", action="store_true", default=True)
    parser.add_argument("--project_name", type=str, default="spurious", help="wandb project name")
    # Confounders
    parser.add_argument("-t", "--target_name")
    parser.add_argument("-c", "--confounder_names", nargs="+")
    parser.add_argument("--up_weight", type=int, default=-1)
    # Resume?
    parser.add_argument("--resume", default=False, action="store_true")
    # Label shifts
    parser.add_argument("--minority_fraction", type=float)
    parser.add_argument("--imbalance_ratio", type=float)
    # Data
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--root_dir", default=None)
    parser.add_argument("--reweight_groups", action="store_true",
                        default=False,
                        help="set to True if loss_type is group DRO")
    parser.add_argument("--augment_data", action="store_true", default=False)
    parser.add_argument("--val_fraction", type=float, default=0.1)

    # Objective
    parser.add_argument("--loss_type", default="erm",
                        choices=["erm", "group_dro", "joint_dro"])
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--generalization_adjustment", default="0.0")
    parser.add_argument("--automatic_adjustment",
                        default=False,
                        action="store_true")
    parser.add_argument("--robust_step_size", default=0.01, type=float)
    parser.add_argument("--joint_dro_alpha", default=1, type=float,
                        help=("Size param for CVaR joint DRO."
                              " Only used if loss_type is joint_dro"))
    parser.add_argument("--use_normalized_loss",
                        default=False,
                        action="store_true")
    parser.add_argument("--btl", default=False, action="store_true")
    parser.add_argument("--hinge", default=False, action="store_true")
    # Model
    parser.add_argument("--model",
                        choices=model_attributes.keys(),
                        default="resnet50")
    parser.add_argument("--train_from_scratch",
                        action="store_true",
                        default=False)
    # Optimization
    parser.add_argument("--n_epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--scheduler", action="store_true", default=False)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--minimum_variational_weight", type=float, default=0)
    # Misc
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show_progress", default=False, action="store_true")
    parser.add_argument("--log_dir", default="./logs")
    parser.add_argument("--log_every", default=50, type=int)
    parser.add_argument("--save_step", type=int, default=10)
    parser.add_argument("--save_best", action="store_true", default=False)
    parser.add_argument("--save_last", action="store_true", default=False)
    parser.add_argument("--use_bert_params", type=int, default=1)
    parser.add_argument("--num_folds_per_sweep", type=int, default=5)
    parser.add_argument("--num_sweeps", type=int, default=4)
    parser.add_argument("--q", type=float, default=0.7)
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument(
        "--metadata_csv_name",
        type=str,
        default="metadata.csv",
        help="name of the csv data file (dataset csv has to be placed in dataset folder).",
    )
    parser.add_argument("--fold", default=None)
    # Our groups (upweighting/dro_ours)
    parser.add_argument(
        "--metadata_path",
        default=None,
        help="path to metadata csv",
    )
    parser.add_argument("--aug_col", default=None)

    ############################################################
    #       Arguments for part2 of the data                    #
    ############################################################
    parser.add_argument("--part2_log_dir", default="./part2_logs")
    args = parser.parse_args()

    if args.model.startswith("bert"):  # and args.model != "bert":
        if args.use_bert_params:
            print("\n" * 5, f"Using bert params", "\n" * 5)
        else:
            print("\n" * 5, f"WARNING, Using {args.model} without using BERT HYPER-PARAMS", "\n" * 5)

    check_args(args)
    if args.metadata_csv_name != "metadata.csv":
        print("\n" * 2
              + f"WARNING: You are using '{args.metadata_csv_name}' instead of the default 'metadata.csv'."
              + "\n" * 2)
    # auto log dir

    if args.log_dir == "AUTO":
        logroot = f"{ROOT_LOG_DIR}/{args.dataset}/autolog_{args.project_name}"
        run_specific = f"{args.loss_type}_upweight{args.up_weight}_epochs{args.n_epochs}_lr{args.lr}_wd{args.weight_decay}"
        args.log_dir = f"{logroot}/{run_specific}/"

    main(args)
