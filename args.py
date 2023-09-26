import argparse


def add_bool_arg(
    parser: argparse.ArgumentParser,
    name: str,
    default: bool = False,
    help_text: str = "",
):
    """Gives arguments with and without --no- prefix which are mutually exclusive.

    Args:
        parser (argparse.ArgumentParser): the parser.
        name (str): name of argument.
        default (bool): The default value. Defaults to False.
        help_text (str): Help text. Defaults to "".
    """
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "".join(["--", name]),
        dest=name,
        action="store_true",
        help=help_text,
    )
    group.add_argument(
        "".join(["--no-", name]),
        dest=name,
        action="store_false",
        help=help_text,
    )
    parser.set_defaults(**{name: default})


def get_training_args() -> argparse.ArgumentParser:
    """Returns argparse parser with arguments for training.

    Returns:
        argparse.ArgumentParser: the parser.
    """
    parser = argparse.ArgumentParser(
        description="Video Classification Training on K400-Tiny",
    )

    parser.add_argument(
        "--name",
        default="R2+1D_18_K400Tiny",
        type=str,
        help="name of experiment",
    )

    parser.add_argument(
        "--version",
        default="v0.1",
        type=str,
        help="version of experiment",
    )
    parser.add_argument(
        "--description",
        default="Baseline",
        type=str,
        help="description of experiment",
    )
    parser.add_argument(
        "--print_interval",
        default=10,
        type=int,
        help="Number of iterations to print information",
    )

    parser.add_argument(
        "--data_path",
        default="example_data/k400tiny_images",
        type=str,
        help="path to image directory",
    )
    parser.add_argument(
        "--annotation_path",
        default="k400tiny/annotations.json",
        type=str,
        help="dataset path",
    )
    parser.add_argument(
        "--num_classes",
        default=400,
        type=int,
        choices=[400],
        help="Number of classes for the dataset",
    )

    parser.add_argument(
        "--clip_len",
        default=8,
        type=int,
        help="Number of frames per clip",
    )
    parser.add_argument(
        "--resize_size",
        default=128,
        type=int,
        help="the min resize size used for train and validation",
    )
    parser.add_argument(
        "--crop_size",
        default=112,
        type=int,
        help="the min crop size used in training and validation",
    )

    parser.add_argument(
        "--batch_size",
        default=6,
        type=int,
        help="Amount of samples per GPU",
    )
    parser.add_argument(
        "--num_workers",
        default=1,
        type=int,
        help="number of data loading workers",
    )
    parser.add_argument(
        "--num_epochs",
        default=50,
        type=int,
        help="number of total epochs",
    )

    parser.add_argument(
        "--lr",
        default=0.1,
        type=float,
        help="Learning rate",
    )
    parser.add_argument(
        "--momentum",
        default=0.9,
        type=float,
        help="Momentum",
    )
    parser.add_argument(
        "--weight_decay",
        default=1e-4,
        type=float,
        help="weight decay (default: 1e-4)",
    )

    parser.add_argument(
        "--save_every",
        default=1,
        type=int,
        help="frequency to save the model to local checkpoint dir",
    )
    parser.add_argument(
        "--local_checkpoint_dir",
        default="checkpoints",
        type=str,
        help="path to save checkpoints locally",
    )

    parser.add_argument(
        "--onnx_export_dir",
        default="model_onnx",
        type=str,
        help="path to save ONNX export for deployment",
    )

    add_bool_arg(
        parser,
        "resume",
        default=False,
        help_text="Load model with weights",
    )
    parser.add_argument(
        "--load_model",
        default="",
        type=str,
        help="path of checkpoint to load when resume is True",
    )

    add_bool_arg(
        parser,
        "val_only",
        default=False,
        help_text="Only run the validation",
    )
    add_bool_arg(
        parser,
        "export_only",
        default=False,
        help_text="Only export to onnx, either best locally stored or given load_model",
    )

    parser.add_argument(
        "--weights",
        default=None,
        type=str,
        help="the torchvision pretrained weights name to load, e.g."
        "R2Plus1D_18_Weights.KINETICS400_V1",
    )

    return parser
