import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--p", type=float, default=1.0)

    args = parser.parse_args()

    assert 0 < args.p <= 1
    return args
