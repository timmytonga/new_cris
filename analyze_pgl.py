import os
from train import run_epoch
from loss import LossComputer
import numpy as np
import pandas as pd
import torch
from utils import Logger, CSVBatchLogger
from data import dro_dataset
from torch.utils.data.sampler import WeightedRandomSampler
from copy import deepcopy
import wandb
from tqdm import tqdm
import argparse

ROOT_LOG_DIR = "/home/thiennguyen/research/pseudogroups/"
WANDB_LOG_DIR = os.path.join(ROOT_LOG_DIR, "wandb")
device = 'cuda:1'


def main(args):
    if args.wandb:
        group_name = 'group_name'
        tags = ['tag']
        job_type = 'job_type'
        run_name = 'run_name'
        wandb.init(project=f"{args.project_name}_{args.dataset}",
                   group=group_name,
                   tags=tags,
                   job_type=job_type,
                   name=run_name,
                   dir=WANDB_LOG_DIR,
                   settings=wandb.Settings(start_method="fork"))
    model_data_root_dir = "/home/thiennguyen/research/pseudogroups/"
    model_data_root_dir += f"{args.dataset}/splitpgl_sweep_logs/p{args.p}_wd0.0001_lr0.0001_s{args.seed}/part1"

    for model_epoch in args.model_epochs:
        print('Model Epoch:', model_epoch)
        part1_model_path = f"{model_data_root_dir}/{model_epoch}_model.pth"
        data_path = f"{model_data_root_dir}/part1and2_data_p{args.p}"
        log_dir = f"{model_data_root_dir}/pgl_analysis_{model_epoch}"

        run_eval_data_on_model(part1_model_path, data_path, log_dir, device)

        csv_path = f"{log_dir}/output.csv"
        analyze_pgl(csv_path, model_epoch, wandb=wandb if args.wandb else None)


def run_eval_data_on_model(part1_model_path, part2_data_path, log_dir, device):
    """
        Run data on part1_model_path and save to log_dir/part2_eval.csv
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = Logger(os.path.join(log_dir, "log.txt"), 'w')
    logger.flush()
    model = torch.load(part1_model_path)
    model.to(device)
    model.eval()
    part2_data = torch.load(part2_data_path)["part2"]
    csv_logger = CSVBatchLogger(os.path.join(log_dir, f"part2_eval.csv"), part2_data.n_groups, mode='w')
    loader_kwargs = {  # setting for args
        "batch_size": 32,
        "num_workers": 4,
        "pin_memory": True,
    }
    part2_loader = dro_dataset.get_loader(part2_data,
                                          train=False,
                                          reweight_groups=None,
                                          **loader_kwargs)

    # then run an epoch on part2 and during that run, generate a csv containing the status of each example
    with torch.set_grad_enabled(False):
        for batch_idx, batch in enumerate(tqdm(part2_loader)):
            batch = tuple(t.to(device) for t in batch)
            x, y, g, data_idx = batch[0], batch[1], batch[2], batch[3]
            outputs = model(x)
            output_df = pd.DataFrame()

            # Calculate stats -- get the prediction and compare with groundtruth -- save to output df
            if batch_idx == 0:
                acc_y_pred = np.argmax(outputs.detach().cpu().numpy(), axis=1)
                acc_y_true = y.cpu().numpy()
                acc_g_true = g.cpu().numpy()
                indices = data_idx.cpu().numpy()

                probs = outputs.detach().cpu().numpy()
            else:  # concatenate
                acc_y_pred = np.concatenate([
                    acc_y_pred,
                    np.argmax(outputs.detach().cpu().numpy(), axis=1)
                ])
                acc_y_true = np.concatenate([acc_y_true, y.cpu().numpy()])
                acc_g_true = np.concatenate([acc_g_true, g.cpu().numpy()])
                indices = np.concatenate([indices, data_idx.cpu().numpy()])
                probs = np.concatenate([probs, outputs.detach().cpu().numpy()], axis=0)

            assert probs.shape[0] == indices.shape[0]
            output_df[f"y_pred"] = acc_y_pred
            output_df[f"y_true"] = acc_y_true
            output_df[f"indices"] = indices
            output_df[f"g_true"] = acc_g_true

            for class_ind in range(probs.shape[1]):
                output_df[f"pred_prob_{class_ind}"] = probs[:, class_ind]
        save_dir = "/".join(csv_logger.path.split("/")[:-1])
        output_df.to_csv(
            os.path.join(save_dir,
                         f"output.csv"))
        print("Saved", os.path.join(save_dir,
                                    f"output.csv"))


def analyze_pgl(csv_path, model_epoch, wandb):
    n_groups = 4
    n_classes = 2

    part2_df = pd.read_csv(csv_path)
    group = part2_df['g_true']
    y_true = part2_df['y_true']
    y_pred = part2_df['y_pred']
    group_count = {g: len(group[group == g]) for g in range(n_groups)}

    pgl = y_true * n_classes + y_pred  # can flip y_pred to get 1-y_pred...
    pgl_count = {g: len(pgl[pgl == g]) for g in range(n_groups)}

    recall = {}
    precision = {}
    for g in range(n_groups):
        recall[g] = round(sum((pgl == g) & (group == g)) / group_count[g], 2)
        precision[g] = round(sum((pgl == g) & (group == g)) / pgl_count[g], 2)

    if wandb is not None:
        pgl_quality_stats = {}
        for g in range(n_groups):
            pgl_quality_stats[f"pgl_count:g{g}"] = pgl_count[g]
            pgl_quality_stats[f"group_count:g{g}"] = group_count[g]
            pgl_quality_stats[f"recall:g{g}"] = recall[g]
            pgl_quality_stats[f"precision:g{g}"] = precision[g]
        pgl_quality_stats["model_epoch"] = model_epoch
        wandb.log(pgl_quality_stats)
    else:
        print(f"pgl_count: \t{pgl_count}")
        print(f"group_count: \t{group_count}")
        print(f"recall: \t{recall}")
        print(f"precision: \t{precision}")
        print(f"{pd.crosstab(pgl, group)}")


datasets = ['CelebA', 'CUB', 'MultiNLI']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--project_name", type=str, default="pgl_quality", help="wandb project name")
    parser.add_argument("-p", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--model_epochs", nargs="+", type=int, default=[0, 10, 20, 30, 100, 300])
    parser.add_argument("-d", "--dataset", choices=datasets, default='CUB')
    args = parser.parse_args()

    main(args)
