from notebooks.example_args import MyMultinliArgs, set_args_and_run_sweep, set_two_parts_args


args = set_two_parts_args(seed=0,
                          p=[0.3, 0.5, 0.7],
                          gpu=0,
                          part1_lr=2e-5,
                          part1_wd=0,
                          part1_n_epochs=21,
                          part2_lr=2e-5,
                          part2_wd=0,
                          part2_n_epochs=21)

######### SET ARGS HERE ########
# misc args
project_name = "Rgl" if not args.part2_use_pgl else "Pgl"
################################

# initialize args
mainargs = MyMultinliArgs(wandb=not args.no_wandb,
                          seed=args.seed,
                          show_progress=args.show_progress,
                          project_name=project_name,
                          gpu=args.gpu,
                          part1_save_every=args.part1_save_every,
                          part1_use_all_data=args.part1_use_all_data)  # default gpu = 0

# run with args
set_args_and_run_sweep(mainargs, args)
