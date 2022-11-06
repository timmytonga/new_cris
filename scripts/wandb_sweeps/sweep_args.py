import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument("--reduce_val_fraction", type=float, default=0.05)

    args = parser.parse_args()
    assert 0 < args.reduce_val_fraction <= 1 
    assert 0 < args.p <= 1
    return args
