"""
New run_expt that can take into account the split (is this implemented already)?
"""
import os
import argparse
import torch

import utils
import wandb

from models import model_attributes
from data.data import dataset_attributes, shift_types, prepare_data, log_data, log_single_data
from data import dro_dataset
from utils import set_seed, Logger, CSVBatchLogger, log_args, get_model, hinge_loss, my_split_data, check_args, \
    get_subsampled_indices
from utils import ROOT_DIR_PATH
from train import train
from generate_pgl import generate_pgl
from data.folds import Subset

ROOT_LOG_DIR = os.path.join(ROOT_DIR_PATH, 'logs')  # "/home/thien/research/pseudogroups/"
WANDB_LOG_DIR = ROOT_LOG_DIR  # os.path.join(ROOT_LOG_DIR, "wandb")

BEST_MODEL_EPOCH = -1  # this is to indicate that we are using the best val_avg_acc from part1 to train part2
NUM_WORKERS = 4  # lower this if memory issue
GROUP_BALANCE_SAMPLING = True


# todo: check why memory increases after every epoch
# todo: refactor so that one split proportion and a knob to control splitting train or val set


def main(args):
    # if part 1: split data --> save split data in log_dir --> train initial model
    # if part 2: load initial ERM model and part2 data
    #       --> generate PGL according to correct upweight factor --> train final model.
    set_seed(args.seed)

    root_log_dir = os.path.dirname(
        args.log_dir)  # if log_dir is autogenerated, we save files where both stages use 1 level above

    # pname is a str for indicating the split propotion of either part1 OR validation set
    pname = f"valp{args.val_split_proportion}" if args.val_split_proportion > 0.0 else f"p{args.part1_split_proportion}"
    pname += f"g" if args.per_group_splitting else ""

    # setup wandb
    if args.wandb:
        group_name = os.path.basename(root_log_dir)
        if args.part == 1:
            job_type = f"part{args.part}{'_all' if args.part1_use_all_data else ''}_{args.loss_type}"
            # group_name = f"part{args.part}_n_epochs{args.n_epochs}_wd{args.weight_decay}_lr{args.lr}"
            tags = [f"part{args.part}", args.loss_type]
            run_name = f"{'all' if args.part1_use_all_data else pname}" + f"_seed{args.seed}"
        elif args.part == 2:  # part2
            job_type = f"part2{pname}{'_oll' if args.part2_only_last_layer else ''}" \
                       f"{args.part1_model_epoch}" \
                       f"_valfrac{args.reduce_val_fraction}" \
                       f"_{'rgl' if args.use_real_group_labels else f'_pgl{args.part1_pgl_model_epoch}'}" \
                       f"{'_rw' if args.reweight_groups else ''}" \
                       f"{'_ga' if args.generalization_adjustment != '0.0' else ''}"
            if args.subsample_minority:
                job_type += '_ss'
            if args.multi_subsample:
                job_type += '_mss'
            job_type += f"_{args.loss_type}_wd{args.weight_decay}lr{args.lr}"
            run_name = f"e{args.part1_model_epoch}_seed{args.seed}"
            run_name += f'_ga{args.generalization_adjustment}' if args.generalization_adjustment != '0.0' else ''
            run_name += f"_resume{args.resume_epoch}" if args.resume else ''
            run_name += "-".join(args.confounder_names) if args.dataset == "jigsaw" else ''
            tags = ['part2', args.loss_type]
        else:
            raise NotImplementedError
        run = wandb.init(project=f"{args.project_name}_{args.dataset}",
                         group=group_name,
                         tags=tags,
                         job_type=job_type,
                         name=run_name,
                         dir=WANDB_LOG_DIR,
                         settings=wandb.Settings(start_method="fork"))
        wandb.config.update(args)

    # Bert specific params
    if args.model.startswith("bert"):
        if args.use_bert_params:
            print(f"Using BERT param with BERT")
            args.max_grad_norm = 1.0
            args.adam_epsilon = 1e-8
            args.warmup_steps = 0
        else:
            print("Warning: Not using BERT params when training BERT")

    # Initialize logs
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
    logger.flush()

    # Some data prep for parts below
    loader_kwargs = {  # setting for args
        "batch_size": args.batch_size,  # can adjust the batchsize of sweep in example_args
        "num_workers": NUM_WORKERS,
        "pin_memory": True,
    }
    data = {}  # this will get set accordingly whether we use part1 or part2
    train_data, val_data, test_data = prepare_data(args, train=True)
    torch.cuda.set_device(args.gpu)
    resume_str = f"RESUMING FROM EPOCH {args.resume_epoch}" if args.resume else ''
    if args.part == 1:
        logger.write(f"****************** TRAINING PART 1 {resume_str}*******************\n")
        ######################################################
        #   First split the data into two parts and save     #
        ######################################################
        # then split it into part1 containing p*n examples of trainset and part2 containing the rest
        if args.resume:  # we should have split our data by now
            part1and2_data = torch.load(os.path.join(args.log_dir, f"part1and2_data_{pname}"))
            part1_data = part1and2_data['part1']  # part2_data = part1and2_data['part2']
            if os.path.exists(os.path.join(args.log_dir, f"new_val_data_{pname}")):
                val_data = torch.load(os.path.join(args.log_dir, f"new_val_data_{pname}"))
        else:  # we need to split data
            logger.write(f"*** PART1: SPLITTING {'TRAIN' if args.val_split_proportion == 0 else 'VAL'} DATA ***\n")
            if args.reduce_val_fraction != 1:
                assert 0 < args.reduce_val_fraction < 1, "reduce_val_fraction must be in (0,1)"
                logger.write(f"*** Using only {args.reduce_val_fraction} fraction of the validation set on part2 ***\n")
                part2_data, _ = make_data_split(val_data, args.reduce_val_fraction,
                                                args.seed, group_balanced=args.per_group_splitting)
                # for part2 we are validating on the same data... Must change the log dir now
                torch.save(part2_data, os.path.join(args.log_dir, f"part2_val_data_{pname}"))
            if args.val_split_proportion == 0:  # we are NOT splitting validation
                part1_data, part2_data = make_data_split(train_data, args.part1_split_proportion,
                                                         args.seed, group_balanced=args.per_group_splitting)
                part1and2_data = {"part1": part1_data, "part2": part2_data}
            else:  # we ARE splitting validation
                part2_data, new_val_data = make_data_split(val_data, args.val_split_proportion,
                                                           args.seed, group_balanced=args.per_group_splitting)
                val_data, old_val_data = new_val_data, val_data  # we are doing validation on this split only now!!!
                logger.write(f"*** Using the reduced validation for validation part1!!! ***\n")
                # if we are splitting validation, part1 is just train_data
                part1and2_data = {"part1": train_data, "part2": part2_data}
                torch.save(new_val_data, os.path.join(args.log_dir, f"new_val_data_{pname}"))
                torch.save(new_val_data, os.path.join(args.log_dir, f"part2_val_data_{pname}"))

            torch.save(part1and2_data, os.path.join(args.log_dir, f"part1and2_data_{pname}"))

        # Now setup the data to train the initial model
        if args.part1_use_all_data:
            logger.write("*** PART1: USING ALL DATA TO TRAIN ***\n")
            data["train_data"] = train_data
            data["train_loader"] = dro_dataset.get_loader(train_data,
                                                          train=True,
                                                          reweight_groups=args.reweight_groups,
                                                          **loader_kwargs)
        else:  # else we are using only the splitted part1
            part1_data = part1and2_data["part1"]  # this ensures we are using the correct split
            data["train_data"] = part1_data
            data["train_loader"] = dro_dataset.get_loader(part1_data,
                                                          train=True,
                                                          reweight_groups=args.reweight_groups,
                                                          **loader_kwargs)
        n_classes = train_data.n_classes
        if args.resume:
            model_path = os.path.join(args.log_dir, f"{args.resume_epoch}_model.pth")
            model = torch.load(model_path)
            logger.write(f"Loaded Resume: {args.resume_epoch}_model.pth\n")
        else:
            model = get_model(
                model=args.model,
                pretrained=not args.train_from_scratch,
                resume=resume,
                n_classes=n_classes,
                dataset=args.dataset,
                log_dir=args.log_dir,
            )

    elif args.part == 2:  # part2
        # first we need to get the correct data for part 2 from the saved dict in part1
        part1_dir = os.path.join(root_log_dir, f"part1_s{args.seed}")

        if args.part1_split_proportion == 1:
            part2_data = train_data
        else:  # we are loading data from the split obtain in part1
            data_path = os.path.join(part1_dir, f"part1and2_data_{pname}")
            if os.path.exists(data_path):
                part2_data = torch.load(data_path)["part2"]
            elif args.val_split_proportion > 0:
                logger.write(f"[Data path not found] Using val split. Creating Part2 data from validation set...\n ")
                if args.reduce_val_fraction != 1:  # need to reduce fraction and use that to train and val
                    assert 0 < args.reduce_val_fraction < 1, "reduce_val_fraction must be in (0,1)"
                    logger.write(
                        f"*** Using only {args.reduce_val_fraction} fraction of the validation set on part2 ***\n")
                    val_data, _ = make_data_split(val_data, args.reduce_val_fraction,
                                                  args.seed, group_balanced=args.per_group_splitting)

                # if args.val_split_proportion == 1, then part2_data is same as new_val_data
                # which is the same as val_data
                part2_data, new_val_data = make_data_split(val_data, args.val_split_proportion, args.seed
                                                           , group_balanced=args.per_group_splitting)
                part1and2_data = {"part1": train_data, "part2": part2_data}
                torch.save(new_val_data, os.path.join(part1_dir, f"new_val_data_{pname}"))
                torch.save(part1and2_data, data_path)
            else:  # this is mainly to avoid retraining for all data on part1
                logger.write(f"[WARNING?] DATA PATH NOT FOUND! CREATING NEW SPLITS INSTEAD! \n\t {data_path}\n")
                part1_data, part2_data = make_data_split(train_data, args.part1_split_proportion, args.seed
                                                         , group_balanced=args.per_group_splitting)
                part1and2_data = {"part1": part1_data, "part2": part2_data}
                torch.save(part1and2_data, data_path)

        if 0 < args.val_split_proportion < 1:  # furthermore, we need to load the correct validation data for part2
            val_data = torch.load(os.path.join(part1_dir, f"new_val_data_{pname}"))
        if args.val_split_proportion == 1:  # this means that train data and val data of part2 is the same
            # don't need to run val again
            val_data = None

        # Now we either create a loader that uses real group labels or generate pseudogroup labels
        if args.use_real_group_labels:
            logger.write(
                f"************* USING REAL GROUP LABELS TO TRAIN PART2 ON {'rw' if args.reweight_groups else ''}"
                f"{args.loss_type}***************\n")
            if args.subsample_minority:  # here we subsample the indices so each group has equal proportion
                logger.write("***** Subsampling to Minority *****\n")
                subsampled_idxs = get_subsampled_indices(part2_data)
                # we are playing with the subset indices here --> this should also affect the loader's indices
                part2_data = dro_dataset.DRODataset(
                    Subset(part2_data.dataset, subsampled_idxs),
                    process_item_fn=None,
                    n_groups=part2_data.n_groups,
                    n_classes=part2_data.n_classes,
                    group_str_fn=part2_data.group_str)

            part2_rw_loader = dro_dataset.get_loader(part2_data,
                                                     train=True,
                                                     reweight_groups=args.reweight_groups,
                                                     **loader_kwargs)
        else:
            assert args.part1_split_proportion < 1, "Part2 data must not be empty... " \
                                                    "Otherwise we haven't implemented this..."
            logger.write(f"*** Generating Pseudogroup Labels from part2_data (size = {len(part2_data)}) *** \n")
            part1_pgl_model_path = os.path.join(part1_dir, f"{args.part1_pgl_model_epoch}_model.pth")
            part2_pgl_data, upsampled_part2 = generate_pgl(part1_pgl_model_path, part2_data, train_data, args)
            logger.write(f"PGL_data group counts = {part2_pgl_data.group_counts()}")
            if args.upweight > 0:  # we are upweighting
                logger.write("*** Training with Upweighted data ***\n")
                part2_rw_loader = dro_dataset.get_loader(upsampled_part2,
                                                         train=True,
                                                         reweight_groups=None,
                                                         **loader_kwargs)
            else:  # else just use the pgl data
                logger.write("*** Training with PGL data.  Real data is: ***\n")
                log_single_data(part2_data, logger, data_name="Real Part2 Data...")
                part2_rw_loader = dro_dataset.get_loader(part2_pgl_data,
                                                         train=True,
                                                         reweight_groups=args.reweight_groups,
                                                         **loader_kwargs)
                part2_data = part2_pgl_data

        data["train_data"] = part2_data  # this is the pgl data if args.loss is group_dro
        data["train_loader"] = part2_rw_loader  # we train using part2 rw now!

        ### Get the main model ###
        # get the correct part1 model.
        if args.part1_model_epoch == BEST_MODEL_EPOCH:
            part1_model_path = os.path.join(part1_dir, f"best_model.pth")
            logger.write("Using best val_avg_acc model from part 1. \n")
        else:
            part1_model_path = os.path.join(part1_dir, f"{args.part1_model_epoch}_model.pth")
        # Initialize model
        if args.resume:  # if we are resuming, load the correct resumed epoch
            model_path = os.path.join(args.log_dir, f"{args.resume_epoch}_model.pth")
            model = torch.load(model_path)
        else:  # else we either load old model or get brand new model
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

        # only last layer
        if args.part2_only_last_layer:
            assert args.part2_use_old_model, \
                "is this intentional? Retraining and only training last layer --> linear classifier on random features?"
            # freeze everything except the last layer
            if args.model.startswith("bert"):
                for name, param in model.named_parameters():
                    if 'classifier' not in name:
                        param.requires_grad = False
                    else:
                        print(f"Not freezing {name}")
            else:
                for name, param in model.named_parameters():
                    if name not in ['fc.weight', 'fc.bias']:
                        param.requires_grad = False

            # make sure freezing really worked
            parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
            assert len(parameters) == 2, f"{len(parameters)} does not match 2!"  # fc.weight, fc.bias
    else:
        raise NotImplementedError("Only part1 and part2... Should not be here at all.")

    if args.wandb:
        wandb.watch(model)

    train_csv_logger = CSVBatchLogger(os.path.join(args.log_dir, f"train.csv"),
                                      train_data.n_groups,
                                      mode=mode)

    if val_data is not None:
        data["val_data"] = val_data
        data["val_loader"] = dro_dataset.get_loader(val_data,
                                                    train=False,
                                                    reweight_groups=None,
                                                    **loader_kwargs)
        val_csv_logger = CSVBatchLogger(os.path.join(args.log_dir, f"val.csv"),
                                        val_data.n_groups,
                                        mode=mode)
    else:
        # Train model given the setup above
        data["val_data"] = None
        data["val_loader"] = None
        val_csv_logger = None

    if args.run_test:
        test_loader = dro_dataset.get_loader(test_data,
                                             train=False,
                                             reweight_groups=None,
                                             **loader_kwargs)
        data["test_data"] = test_data
        data["test_loader"] = test_loader
        test_csv_logger = CSVBatchLogger(os.path.join(args.log_dir, f"test.csv"),
                                         test_data.n_groups,
                                         mode=mode)
    else:
        data["test_data"] = None
        data["test_loader"] = None
        test_csv_logger = None

    log_data(data, logger)
    logger.flush()

    # Define the objective
    if args.hinge:
        assert args.dataset in ["CelebA", "CUB"]  # Only supports binary
        criterion = hinge_loss
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction="none")

    if resume:
        # raise NotImplementedError  # Check this implementation.
        # # df = pd.read_csv(os.path.join(args.log_dir, "test.csv"))
        # # epoch_offset = df.loc[len(df) - 1, "epoch"] + 1
        # # logger.write(f"starting from epoch {epoch_offset}")
        epoch_offset = args.resume_epoch + 1
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
    if val_data is not None:
        val_csv_logger.close()
    if args.run_test:
        test_csv_logger.close()

    if args.wandb:
        wandb.finish()


def make_data_split(data, split_proportion, seed, group_balanced=GROUP_BALANCE_SAMPLING):
    """
    Set split_proportion = 1 to make copies
    """
    # then split it into part1 containing f*n examples of trainset and part2 containing the rest
    if split_proportion < 1:
        part1, part2 = my_split_data(data, part1_split_fraction=split_proportion,
                                     seed=seed, group_balanced=group_balanced)
        part1_data = dro_dataset.DRODataset(
            part1,
            process_item_fn=None,
            n_groups=data.n_groups,
            n_classes=data.n_classes,
            group_str_fn=data.group_str)
        part2_data = dro_dataset.DRODataset(
            part2,
            process_item_fn=None,
            n_groups=data.n_groups,
            n_classes=data.n_classes,
            group_str_fn=data.group_str)
    elif split_proportion == 1:  # this means we use the full dataset in both parts
        part1_data, part2_data = data, data
    else:
        raise NotImplementedError
    return part1_data, part2_data


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
    parser.add_argument("--run_test", action="store_true", default=False)
    # Confounders
    parser.add_argument("-t", "--target_name")
    parser.add_argument("-c", "--confounder_names", nargs="+")
    # Resume?
    parser.add_argument("--resume", default=False, action="store_true")
    parser.add_argument("--resume_epoch", default=None, type=int)
    # Label shifts
    parser.add_argument("--minority_fraction", type=float)
    parser.add_argument("--imbalance_ratio", type=float)
    # Data
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--root_dir", default=None)
    parser.add_argument("--reweight_groups", action="store_true",
                        default=False,
                        help="set to True if loss_type is group DRO")
    parser.add_argument("--subsample_minority", action="store_true", default=False,
                        help="Subsample so that all the groups have equal proportion")
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
    utils.parser_add_optimization_args(parser)
    # Misc args: seed, log_dir, gpu, save_best, save_last, etc.
    utils.parser_add_misc_args(parser)
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
    parser.add_argument("--part", choices=[1, 2], type=int, help="Specify whether we are training part1 or part2",
                        required=True)
    # new!
    parser.add_argument("--val_split_proportion", type=float, default=0,
                        help="Save this much proportion for part2 of the validation set."
                             "Don't set this (i.e. leave it at 0) if want to use part1_split_proportion..."
                             "Otherwise, this will override all part1_split_proportion settings.")
    parser.add_argument("--reduce_val_fraction", type=float, default=1,
                        help="Use only this fraction of the validation set.")

    # for use with part1
    parser.add_argument("--part1_split_proportion", type=float, default=0.5,
                        help="Split proportion for part1: how much data to use for part1 while the rest for part2")
    parser.add_argument("--part1_use_all_data", action="store_true", default=False)
    # for use with part2
    parser.add_argument("--part1_model_epoch", type=int, help="Specify which epoch from part1 to continue part2")
    parser.add_argument("--part1_pgl_model_epoch", type=int, help="Specify which epoch to load the initial model1 from"
                                                                  "to generate the pseudogroup labels")
    parser.add_argument("--part2_only_last_layer", action="store_true", default=False,
                        help="This is used to finetune model1 last layer")
    parser.add_argument("--part2_use_old_model", action="store_true", default=False,
                        help="For part2: whether we want to resuse the model from part1")
    parser.add_argument("--upweight", type=int, default=-1,
                        help="upweight factor for retraining. Set upweight=0 to get equal group sampling. "
                             "Set to -1 for inverse group count sampling.")
    parser.add_argument("--use_real_group_labels", action="store_true", default=False,
                        help="Use real group labels to retrain part2")
    parser.add_argument("--multi_subsample", action="store_true", default=False,
                        help="We re-subsample every epoch instead of just subsampling once")
    parser.add_argument("--per_group_splitting", action="store_true", default=False,
                        help="When splitting dataset, split each group individually instead of the whole dataset"
                             "to ensure that when one group is too small, one split doesn't contain all the members of"
                             "that group. ")

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
    pname = f"p{args.part1_split_proportion}" if args.val_split_propotion == 0 else f"valp{args.val_split_proportion}"

    if args.log_dir == "AUTO":
        logroot = f"{ROOT_LOG_DIR}/{args.dataset}/autolog_{args.project_name}"
        run_specific = f"part{args.part}_{pname}_wd{args.weight_decay}_lr{args.lr}_{args.loss_type}"
        if args.part == 2:
            oll_part2 = "oll" if not args.part2_train_full else "full"
            extra_part2 = f"{'rw' if args.reweight_groups else ''}" \
                          f"{'_subsample' if args.subsample_minority else ''}" \
                          f"{'_rgl' if args.use_real_group_labels else f'_pgl{args.part1_pgl_model_epoch}'}"
            run_specific += f"_{oll_part2}{args.part1_model_epoch}{extra_part2}_s{args.seed}"

        run_specific += f'_ga{args.generalization_adjustment}' if args.generalization_adjustment != '0.0' else ''
        args.log_dir = os.path.join(logroot, run_specific)

    main(args)
