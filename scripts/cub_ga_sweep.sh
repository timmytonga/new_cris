#!/bin/bash

for seed in 2 3 4
do
    for ga in 0.5 0.75 0.25 1; do python scripts/cub_sweep.py -p 0.7 --seed $seed --part1_model_epochs -1 --run_test --part2_loss_type group_dro --part2_reweight --part2_n_epochs 101 --part2_group_adjustment $ga ; done
done
