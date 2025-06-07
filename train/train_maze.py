import sys
sys.path.append('.')
import argparse
from models.metric import model_train_metric as md  # _newout_sqrtlog _newout_log2


def main(args=None):
    parser = argparse.ArgumentParser(description="Train the Maze model")
    parser.add_argument(
        "--data",
        default="./datasets/test/maze",
        help="Path to the dataset directory",
    )
    parser.add_argument(
        "--output",
        default="./Experiments/Maze",
        help="Directory where the model checkpoints will be saved",
    )

    opts = parser.parse_args(args)

    model = md.Model(
        opts.output,
        opts.data,
        2,
        [-0.0, -0.0],
        device="cuda:0",
    )

    model.train()


if __name__ == "__main__":
    main()


