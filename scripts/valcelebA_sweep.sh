#!/bin/bash
set -e

part1_lr=$1
part1_wd=$2
part1_n_epochs=31

val_split=0.5
seed=3
part2_n_epochs=51

# first run to get part1
python scripts/celebA_sweep.py --val_split -p $val_split --seed $seed --part1_model_epochs -1 --part1_save_best --run_test --part2_loss_type group_dro --part2_reweight --part1_lr $part1_lr --part1_wd $part1_wd --part2_n_epochs $part2_n_epochs --part2_lr 1e-5 --part2_wd 1e-5 --run_part1 --part1_n_epochs $part1_n_epochs 

# then sweep over a bunch of part2
for part2_lr in 1e-1 1e-2 1e-3 1e-4 1e-5
do
for part2_wd in 1 1e-1 1e-2 1e-3 1e-4 0
do

    python scripts/celebA_sweep.py --val_split -p $val_split --seed $seed --part1_model_epochs -1 --part1_save_best --run_test --part2_loss_type group_dro --part2_reweight --part2_lr $part2_lr --part2_n_epochs $part2_n_epochs --part2_wd $part2_wd 
done
done
