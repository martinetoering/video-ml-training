from typing import Tuple

import mlflow
import numpy as np
import torch
import torchmetrics
import torchvision


def train_model(
    epoch: int,
    num_epochs: int,
    device: torch.device,
    model: torchvision.models.video.r2plus1d_18,
    optimizer: torch.optim.SGD,
    dataloader: torch.utils.data.DataLoader,
    loss_module: torch.nn.CrossEntropyLoss,
    acc1_metric: torchmetrics.classification.MulticlassAccuracy,
    acc5_metric: torchmetrics.classification.MulticlassAccuracy,
    distributed: bool,
    print_interval: int,
) -> Tuple[np.float64, torch.Tensor, torch.Tensor]:
    """Runs one epoch of training on the training set.

    Args:
        epoch (int): current epoch
        num_epochs (int): total number of epochs.
        device (torch.device): cpu or cuda device.
        model (torchvision.models.video.r2plus1d_18): video model
        optimizer (torch.optim): optimizer to use.
        dataloader (torch.utils.data.Dataloader): val dataloader.
        loss_module (torch.nn.CrossEntropyLoss): cross entropy loss module.
        acc1_metric (torchmetrics.classification.MulticlassAccuracy): track acc@1.
        acc5_metric: (torchmetrics.classification.MulticlassAccuracy): track acc@5.
        distributed (bool): Distributed training or not.
        print_interval (int): How often to print info.

    Returns:
        Average loss, acc1_metric, acc5_metric.
    """
    log = not distributed or torch.distributed.get_rank() == 0

    losses = []
    model.train()

    if log:
        print(f"Starting training of Epoch {epoch}")

    for idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)  # B

        optimizer.zero_grad()

        outputs = model(inputs)  # B, N
        loss = loss_module(outputs, targets)  # item

        loss.backward()
        optimizer.step()

        # Batch accuracy
        acc1 = acc1_metric(outputs, targets)
        acc5 = acc5_metric(outputs, targets)

        # Reduce the losses from all distributed servers
        if distributed:
            loss = loss.detach()
            torch.distributed.all_reduce(loss, torch.distributed.ReduceOp.SUM)

        losses.append(loss.item())

        if log and idx % print_interval == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] ",
                f"Iteration [{idx}/{len(dataloader)}] | "
                f"Loss: {loss.item():.6f} | "
                f"Acc@1: {acc1:.4f} | Acc@5: {acc5:.4f}",
            )
            step_value = epoch * len(dataloader) + idx
            mlflow.log_metric("train.step.loss", loss.item(), step_value)
            mlflow.log_metric("train.step.acc1", acc1, step_value)
            mlflow.log_metric("train.step.acc5", acc5, step_value)

    avg_loss = np.mean(losses)
    acc1 = acc1_metric.compute()
    acc5 = acc5_metric.compute()
    if log:
        print(
            f"\nEpoch [{epoch}/{num_epochs}] | Average Loss: {avg_loss:.4f} | "
            f"Average Acc@1: {acc1:.4f} | Average Acc@5: {acc5:.4f}",
        )
    acc1_metric.reset()
    acc5_metric.reset()
    return avg_loss, acc1, acc5
