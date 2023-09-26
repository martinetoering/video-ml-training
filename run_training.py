import argparse
import os
import time

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchmetrics.classification as classification_metrics
import torchvision
from args import get_training_args
from datasets.k400tiny import K400tiny
from epoch_eval import evaluate
from epoch_train import train_model
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from transforms.videotransforms import InverseNormalize, VideoTransform
from utils import export_to_onnx


def set_random_seed(seed: int):
    """Set random seed for packages to ensure reproducability.

    Args:
        seed (int): Random seed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train(args: argparse.Namespace, distributed: bool = False):
    """Train the model.

    Args:
        args (argparse.Namespace): Arguments.
        distributed (bool): Indicates distributed mode.
    """
    print(args)
    local_checkpoint_dir = args.local_checkpoint_dir
    onnx_export_dir = args.onnx_export_dir
    os.makedirs(local_checkpoint_dir, exist_ok=True)
    os.makedirs(onnx_export_dir, exist_ok=True)
    experiment_name = f"{args.name}_{args.version}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device", device)

    first_rank = distributed and torch.distributed.get_rank() == 0

    if not distributed or first_rank:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            mlflow.create_experiment(name=experiment_name)
            experiment = mlflow.get_experiment_by_name(experiment_name)
        if mlflow.active_run():
            mlflow.end_run("KILLED")
        mlflow.start_run(experiment_id=experiment.experiment_id)
        mlflow.log_params(vars(args))

    set_random_seed(42)
    # Ensure deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("Loading data")
    train_transform = VideoTransform(
        args.clip_len,
        "train",
        args.resize_size,
        args.crop_size,
    )
    if args.weights and args.val_only:
        weights = torchvision.models.get_weight(args.weights)
        test_transform = weights.transforms()
        inv_normalize = InverseNormalize(
            test_transform.mean,
            test_transform.std,
        )
    else:
        test_transform = VideoTransform(
            args.clip_len,
            "test",
            args.resize_size,
            args.crop_size,
        )
        inv_normalize = InverseNormalize()

    train_dataset = K400tiny(
        args.data_path,
        args.annotation_path,
        "train",
        transform=train_transform,
    )
    val_dataset = K400tiny(
        args.data_path,
        args.annotation_path,
        "val",
        transform=test_transform,
    )

    if distributed:
        dis_train_samp = DistributedSampler(train_dataset)  # type:ignore[var-annotated]
        dis_val_samp = DistributedSampler(val_dataset)  # type: ignore[var-annotated]
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sampler=dis_train_samp,
        )
        val_loader = data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sampler=dis_val_samp,
        )
    else:
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        val_loader = data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    train_acc1_metric = classification_metrics.MulticlassAccuracy(
        num_classes=args.num_classes,
        top_k=1,
    )
    train_acc5_metric = classification_metrics.MulticlassAccuracy(
        num_classes=args.num_classes,
        top_k=5,
    )
    val_acc1_metric = classification_metrics.MulticlassAccuracy(
        num_classes=args.num_classes,
        top_k=1,
    )
    val_acc5_metric = classification_metrics.MulticlassAccuracy(
        num_classes=args.num_classes,
        top_k=5,
    )

    print("Creating model")
    model = torchvision.models.video.r2plus1d_18(weights=args.weights)
    # Append metrics to model for automatic ddp handling
    model.train_acc1_metric = train_acc1_metric
    model.train_acc5_metric = train_acc5_metric
    model.val_acc1_metric = val_acc1_metric
    model.val_acc5_metric = val_acc5_metric

    model.to(device)

    loss_module = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    model_without_ddp = model
    if distributed:
        model = DistributedDataParallel(model)
        model_without_ddp = model.module

    start_epoch = 0
    num_epochs = args.num_epochs
    if args.load_model == "" and args.export_only and args.weights is None:
        args.load_model = os.path.join(
            local_checkpoint_dir,
            f"{experiment_name}_best.pth",
        )

    if args.resume or (args.export_only and args.weights is None):
        print(f"Load model from {args.load_model}")
        checkpoint = torch.load(args.load_model, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1

    if args.val_only:
        eval_loss, eval_acc1, eval_acc5 = evaluate(
            start_epoch,
            num_epochs,
            device,
            model,
            val_loader,
            loss_module,
            inv_normalize,
            val_acc1_metric,
            val_acc5_metric,
            distributed,
            args.print_interval,
            val_dataset.label_by_id,
        )
        print(f"Eval loss: {eval_loss}")
        print(f"Eval acc@1: {eval_acc1}")
        print(f"Eval acc@5: {eval_acc5}")
        if not distributed or first_rank:
            mlflow.end_run()
        return

    if args.export_only:
        export_to_onnx(
            experiment_name,
            model_without_ddp.cpu(),
            train_dataset.id_by_label,
            onnx_export_dir,
            args.clip_len,
            args.crop_size,
        )
        return

    print("Start training")
    start_time = time.time()
    best_accuracy = 0.0
    for epoch in range(start_epoch, num_epochs):
        if distributed:
            train_loader.sampler.set_epoch(epoch)  # type: ignore[attr-defined]
            val_loader.sampler.set_epoch(epoch)  # type: ignore[attr-defined]

        train_loss, train_acc1, train_acc5 = train_model(
            epoch,
            num_epochs,
            device,
            model,
            optimizer,
            train_loader,
            loss_module,
            train_acc1_metric,
            train_acc5_metric,
            distributed,
            args.print_interval,
        )
        eval_loss, eval_acc1, eval_acc5 = evaluate(
            epoch,
            num_epochs,
            device,
            model,
            val_loader,
            loss_module,
            inv_normalize,
            val_acc1_metric,
            val_acc5_metric,
            distributed,
            args.print_interval,
            val_dataset.label_by_id,
        )

        if not distributed or first_rank:
            mlflow.log_metric("train_loss", train_loss, epoch)
            mlflow.log_metric("train_acc1", train_acc1, epoch)
            mlflow.log_metric("train_acc5", train_acc5, epoch)
            mlflow.log_metric("eval_loss", eval_loss, epoch)
            mlflow.log_metric("eval_acc1", eval_acc1, epoch)
            mlflow.log_metric("eval_acc5", eval_acc5, epoch)

            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            if eval_acc1 >= best_accuracy:
                best_model_save_path = os.path.join(
                    local_checkpoint_dir,
                    f"{experiment_name}_best.pth",
                )
                torch.save(checkpoint, best_model_save_path)
                print("Uploading the model checkpoint to mlflow")
                mlflow.log_artifact(
                    best_model_save_path,
                    artifact_path="checkpoint",
                )
                best_accuracy = eval_acc1.item()

            if epoch % args.save_every == 0:
                model_save_path = os.path.join(
                    local_checkpoint_dir,
                    f"{experiment_name}_{epoch}.pth",
                )
                torch.save(checkpoint, model_save_path)

    if not distributed or first_rank:
        mlflow.end_run()
    total_time = time.time() - start_time
    print(f"Training time {total_time}")

    export_to_onnx(
        experiment_name,
        model_without_ddp.cpu(),
        train_dataset.id_by_label,
        onnx_export_dir,
        args.clip_len,
        args.crop_size,
    )


if __name__ == "__main__":
    parser = get_training_args()
    args = parser.parse_args()
    train(args)
