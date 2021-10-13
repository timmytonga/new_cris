import os
from train import run_epoch
from loss import LossComputer
import numpy as np
import pandas as pd
import torch
from utils import Logger, CSVBatchLogger
from data import dro_dataset
from torch.utils.data.sampler import WeightedRandomSampler


def generate_pgl(part1_model_path, part2_data, train_data, args):
    n_classes = train_data.n_classes
    part2eval_log_dir = os.path.join(args.log_dir, "part2_eval")
    if not os.path.exists(part2eval_log_dir):
        os.makedirs(part2eval_log_dir)
    part2eval_logger = Logger(os.path.join(part2eval_log_dir, "log.txt"), 'w')
    part2eval_logger.flush()

    # first load the previous model
    modeleval = torch.load(part1_model_path)
    # initialize logger and loader for part2
    part2eval_csv_logger = CSVBatchLogger(os.path.join(part2eval_log_dir, f"part2_eval.csv"),
                                          part2_data.n_groups,
                                          mode='w')
    loader_kwargs = {  # setting for args
        "batch_size": 128,
        "num_workers": 4,
        "pin_memory": True,
    }

    part2_loader = dro_dataset.get_loader(part2_data,
                                          train=False,
                                          reweight_groups=None,
                                          **loader_kwargs)
    adjustments = np.array([0] * 4)
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    part2eval_loss_computer = LossComputer(
        criterion,
        loss_type=args.loss_type,
        dataset=part2_data,
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
        loss_computer=part2eval_loss_computer,
        logger=part2eval_logger,
        csv_logger=part2eval_csv_logger,
        args=args,
        is_training=False,
        show_progress=True,
        log_every=50,
        scheduler=None,
        csv_name="pseudogroup_eval",
        wandb_group="part2_eval",
        wandb=None,
    )
    # todo: make upweight factor so that it's equal sampling per group...
    part2_df = pd.read_csv(os.path.join(part2eval_log_dir, 'output_part2_eval_epoch_0.csv'))
    n_groups = n_classes*2
    true_y = part2_df['y_true_pseudogroup_eval_epoch_0_val']
    pred_y = part2_df['y_pred_pseudogroup_eval_epoch_0_val']
    misclassified = true_y != pred_y
    sampler, upsampled_part2 = None, None
    group_array = misclassified + true_y * 2  # times 2 to make group (true_y, status)
    group_counts = (torch.arange(n_groups).unsqueeze(1) == torch.tensor(group_array)).sum(1).float()
    if args.upweight == 0:  # this means we do equal sampling
        group_weights = len(part2_data) / group_counts
        weights = group_weights[group_array]
        sampler = WeightedRandomSampler(weights, len(part2_data), replacement=True)
    else:  # this means we upweight
        assert args.upweight == -1 or args.upweight > 0
        aug_indices = part2_df['indices_pseudogroup_eval_epoch_0_val'][misclassified]
        upweight_factor = len(part2_df) // len(aug_indices) if args.upweight != -1 else args.upweight
        print(f"UPWEIGHT FACTOR = {upweight_factor}")
        combined_indices = list(aug_indices) * upweight_factor + list(part2_df['indices_pseudogroup_eval_epoch_0_val'])
        upsampled_part2 = torch.utils.data.Subset(train_data.dataset.dataset, combined_indices)

    return upsampled_part2, sampler
