"""
New run_expt that can take into account the split (is this implemented already)?
"""
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
from utils import set_seed, Logger, CSVBatchLogger, log_args, get_model, hinge_loss, split_data, check_args
from train import train, run_epoch
from loss import LossComputer
import pandas as pd
from generate_pgl import generate_pgl
from torch.utils.data import DataLoader

ROOT_LOG_DIR = "/home/thiennguyen/research/pseudogroups/"
RUN_TEST = False  # make this an args somehow??


def main(args):
    # if part 1: split data --> save split data in log_dir --> train initial model
    # if part 2: load initial ERM model and part2 data
    #       --> generate PGL according to correct upweight factor --> train final model.
    set_seed(args.seed)

    if args.wandb:
        if args.part == 1:
            group_name = f"part{args.part}_n_epochs{args.n_epochs}_wd{args.weight_decay}_lr{args.lr}"
            tags = [f"part{args.part}"]
        else:  # part2
            only_last_layer = "oll" if args.part2_only_last_layer else "full"
            which_old_model = f"old-e{args.part1_model_epoch}" if args.part2_use_old_model else "new"
            group_name = f"part{args.part}_{args.loss_type}_{which_old_model}-{only_last_layer}_wd{args.weight_decay}_lr{args.lr}"
            tags = [only_last_layer, which_old_model, 'part2']
        run = wandb.init(project=f"{args.project_name}_{args.dataset}",
                         group=group_name,
                         tags=tags,
                         job_type=f"part{args.part}",
                         name=f"p{args.part1_split_proportion}")
        wandb.config.update(args)

    ## Initialize logs
    if os.path.exists(args.log_dir) and args.resume:
        resume = True
        mode = "a"
    else:
        resume = False
        mode = "w"
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
        print(f"******* Created dir {args.log_dir} **********")
    logger = Logger(os.path.join(args.log_dir, "log.txt"), 'w')
    # Record args
    log_args(args, logger)
    root_log_dir = os.path.dirname(args.log_dir)  # if log_dir is autogenerated, we save files where both stages use 1 level above
    # Some data prep for parts below
    loader_kwargs = {  # setting for args
        "batch_size": args.batch_size,
        "num_workers": 4,
        "pin_memory": True,
    }
    data = {}  # this will get set accordingly whether we use part1 or part2
    train_data, val_data, test_data = prepare_data(args, train=True)

    if args.part == 1:
        ######################################################
        #   First split the data into two parts and save     #
        ######################################################
        # then split it into part1 containing f*n examples of trainset and part2 containing the rest
        part1, part2 = split_data(train_data.dataset, part1_proportion=args.part1_split_proportion, seed=args.seed)
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
        part1and2_data = {"part1": part1_data, "part2": part2_data}
        torch.save(part1and2_data, os.path.join(args.log_dir, "part1and2_data"))

        # Now setup the data to train the initial model
        part1_loader = dro_dataset.get_loader(part1_data,
                                              train=True,
                                              reweight_groups=None,
                                              **loader_kwargs)
        data["train_data"] = part1_data
        data["train_loader"] = part1_loader  # we train using  part1
        n_classes = train_data.n_classes
        model = get_model(
            model=args.model,
            pretrained=not args.train_from_scratch,
            resume=resume,
            n_classes=n_classes,
            dataset=args.dataset,
            log_dir=args.log_dir,
        )
        if args.wandb:
            wandb.watch(model)

    elif args.part == 2:   # part2
        # we are given the same log_dir from part1 -- make sure we copy the right logdir somehow...
        # first we need to generate the pgls and get the loader with that
        part1_dir = os.path.join(root_log_dir, "part1")
        part2_data = torch.load(os.path.join(part1_dir, "part1and2_data"))["part2"]
        part1_model_path = os.path.join(part1_dir, f"{args.part1_model_epoch}_model.pth")
        # this should be a Subset that contains the right upweight for the right points
        upsampled_part2, eq_group_sampler = generate_pgl(part1_model_path, part2_data, train_data, args)
        if args.upweight == 0:
            part2_rw_loader = DataLoader(part2_data, shuffle=False, sampler=eq_group_sampler, **loader_kwargs)
        else:
            part2_rw_loader = dro_dataset.get_loader(upsampled_part2,
                                                 train=True,
                                                 reweight_groups=None,
                                                 **loader_kwargs)
        data["train_loader"] = part2_rw_loader  # we train using part2 rw now!
        data["train_data"] = part2_data
        ## Initialize model
        if not args.part2_use_old_model:
            model = get_model(
                model=args.model,
                pretrained=not args.train_from_scratch,
                resume=resume,
                n_classes=train_data.n_classes,
                dataset=args.dataset,
                log_dir=args.log_dir,
            )
        else:
            model = torch.load(part1_model_path)  # this model_path is from the cell above -- fix this in code

        if args.part2_only_last_layer:
            assert args.part2_use_old_model, "is this intentional? Retraining and only training last layer --> linear classifier on random features?"
            # freeze everything except the last layer
            for name, param in model.named_parameters():
                if name not in ['fc.weight', 'fc.bias']:
                    param.requires_grad = False
            # make sure freezing really worked
            parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
            assert len(parameters) == 2  # fc.weight, fc.bias
        if args.wandb:
            wandb.watch(model)
    else:
        raise NotImplementedError("Only part1 and part2... Should not be here at all.")

    # Train model given the setup above
    # get loader
    val_loader = dro_dataset.get_loader(val_data,
                                        train=False,
                                        reweight_groups=None,
                                        **loader_kwargs)
    data["val_data"] = val_data
    data["val_loader"] = val_loader

    data["test_data"] = None
    data["test_loader"] = None  # this will get set below if RUN_TEST is true

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
        data["test_data"] = test_data
        data["test_loader"] = test_loader
        test_csv_logger = CSVBatchLogger(os.path.join(args.log_dir, f"test.csv"),
                                         test_data.n_groups,
                                         mode=mode)

    log_data(data, logger)
    logger.flush()

    ## Define the objective
    if args.hinge:
        assert args.dataset in ["CelebA", "CUB"]  # Only supports binary
        criterion = hinge_loss
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction="none")

    if resume:
        raise NotImplementedError  # Check this implementation.
        # df = pd.read_csv(os.path.join(args.log_dir, "test.csv"))
        # epoch_offset = df.loc[len(df) - 1, "epoch"] + 1
        # logger.write(f"starting from epoch {epoch_offset}")
    else:
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

    if args.wandb:
        wandb.finish()


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

    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--project_name", type=str, default="splitpgl", help="wandb project name")
    # Confounders
    parser.add_argument("-t", "--target_name")
    parser.add_argument("-c", "--confounder_names", nargs="+")
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
    ###################
    #  SPLIT_PGL ARGS #
    ###################
    parser.add_argument("--part", choices=[1, 2], type=int, help="Specify whether we are training part1 or part2", required=True)
    # for use with part1
    parser.add_argument("--part1_split_proportion", type=float, default=0.5, help="Split proportion for part1")
    # for use with part2
    parser.add_argument("--part1_model_epoch", type=int, help="Specify which epoch to load the initial model1 from")
    parser.add_argument("--part2_only_last_layer", action="store_true", default=False, help="This is used to finetune model1 last layer")
    parser.add_argument("--part2_use_old_model", action="store_true", default=False, help="For part2: whether we want to resuse the model from part1")
    parser.add_argument("--upweight", type=int, default=-1, help="upweight factor for retraining. Set upweight=0 to get equal group sampling. Set to -1 for inverse group count sampling.")

    args = parser.parse_args()

    assert 1 >= args.part1_split_proportion >= 0, "split proportion must be in [0,1]."

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
        run_specific = f"{args.loss_type}_upweight{args.upweight}_epochs{args.n_epochs}_lr{args.lr}_wd{args.weight_decay}"
        args.log_dir = f"{logroot}/{run_specific}/part{args.part}"

    main(args)
