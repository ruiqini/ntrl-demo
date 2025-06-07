import sys
sys.path.append('.')
import argparse
from models.metric import model_train_metric as md


def main(args=None):
    parser = argparse.ArgumentParser(description="Train the Gibson model")
    parser.add_argument(
        "--data",
        default="./datasets/gibson/Spotswood",
        help="Path to the dataset directory",
    )
    parser.add_argument(
        "--output",
        default="./Experiments/Gib",
        help="Directory where the model checkpoints will be saved",
    )

    opts = parser.parse_args(args)

    model = md.Model(
        opts.output,
        opts.data,
        3,
        [-0.15, 0.1, 0.1],
        device="cuda:0",
    )

    model.train()


if __name__ == "__main__":
    main()


