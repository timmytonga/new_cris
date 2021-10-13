from notebooks.example_args import MyCUBArgs
import run_expt


WANDB=True

for wd in [1e-4, 1e-1, 1]:
    for lr in [1e-3, 1e-4, 1e-5]:
        args = MyCUBArgs(n_epochs=300, wd=wd, lr=lr, wandb=WANDB)
        ERM_args = args.part1_args
        run_expt.main(ERM_args)

