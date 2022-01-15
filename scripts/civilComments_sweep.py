from notebooks.example_args import MyCivilCommentsArgs, set_args_and_run_sweep, set_two_parts_args


args = set_two_parts_args(seed=0,
                          p=[0.3, 0.5, 0.7],
                          gpu=0,
                          part1_lr=1e-5,
                          part1_wd=1e-2,
                          part1_n_epochs=6,
                          part2_lr=1e-5,
                          part2_wd=0,
                          part2_n_epochs=6)

######### SET ARGS HERE ########
# misc args
project_name = "Rgl" if not args.part2_use_pgl else "Pgl"
################################

# initialize args
mainargs = MyCivilCommentsArgs(wandb=not args.no_wandb,
                               seed=args.seed,
                               show_progress=args.show_progress,
                               project_name=project_name,
                               gpu=args.gpu,
                               part1_save_every=args.part1_save_every,
                               part1_use_all_data=args.part1_use_all_data)

# run with args
set_args_and_run_sweep(mainargs, args)
