from typing import Dict, Tuple

import mlflow
import numpy as np
import torch
import torchmetrics
import torchvision
from transforms.videotransforms import InverseNormalize
from utils import visualize_classification


def evaluate(
    epoch: int,
    num_epochs: int,
    device: torch.device,
    model: torchvision.models.video.r2plus1d_18,
    dataloader: torch.utils.data.DataLoader,
    loss_module: torch.nn.CrossEntropyLoss,
    inv_normalize: InverseNormalize,
    acc1_metric: torchmetrics.classification.MulticlassAccuracy,
    acc5_metric: torchmetrics.classification.MulticlassAccuracy,
    distributed: bool,
    print_interval: int,
    label_by_id: Dict[int, str],
) -> Tuple[np.float64, torch.Tensor, torch.Tensor]:
    """Runs one epoch of evaluation on the validation set.

    Args:
        epoch (int): current epoch
        num_epochs (int): total number of epochs.
        device (torch.device): cpu or cuda device.
        model (torchvision.models.video.r2plus1d_18): video model
        dataloader (torch.utils.data.Dataloader): val dataloader.
        loss_module (torch.nn.CrossEntropyLoss): cross entropy loss module.
        inv_normalize (InverseNormalize): custom transform.
        acc1_metric (torchmetrics.classification.MulticlassAccuracy): track acc@1.
        acc5_metric: (torchmetrics.classification.MulticlassAccuracy): track acc@5.
        distributed (bool): Distributed training or not.
        print_interval (int): How often to print info.
        label_by_id (Dict[int, str]): Annotation dict.

    Returns:
        Average loss, acc1_metric, acc5_metric.
    """
    log = not distributed or torch.distributed.get_rank() == 0

    losses = []
    model.eval()

    # Get random sample id from dataloader for later visualization
    vis_sample_id = torch.randint(len(dataloader), size=(1,)).item()

    if log:
        print(f"Starting evaluation of Epoch {epoch}")

    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)  # B

            outputs = model(inputs)
            loss = loss_module(outputs, targets)

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
                    f"Acc@1: {acc1:.4f} | Acc@5: {acc5:.4f} | ",
                )
                step_value = epoch * len(dataloader) + idx
                mlflow.log_metric("val.step.loss", loss.item(), step_value)
                mlflow.log_metric("val.step.acc1", acc1, step_value)
                mlflow.log_metric("val.step.acc5", acc5, step_value)

            if idx == vis_sample_id:
                # Store random batch for visualization later
                sample_batch = inputs
                sample_targets = targets
                sample_outputs = outputs

    avg_loss = np.mean(losses)
    acc1 = acc1_metric.compute()
    acc5 = acc5_metric.compute()
    if log:
        print(
            f"\nEpoch [{epoch}/{num_epochs}] | Average Loss: {avg_loss:.4f} | "
            f"Average Acc@1: {acc1:.4f} | Average Acc@5: {acc5:.4f}",
        )

        # Visualize random batch and upload to Mlflow
        sample_probs = torch.softmax(sample_outputs, dim=1)
        sample_preds = torch.argmax(sample_probs, dim=1).float()
        correct = sample_preds == sample_targets
        sample_batch = torch.stack([inv_normalize(t) for t in sample_batch], 0)
        sample_batch = sample_batch.detach().cpu()
        sample_targets = sample_targets.detach().cpu()

        sample_labels = []
        for target in sample_targets:
            label = label_by_id[target.item()]
            sample_labels.append(label)
        fig = visualize_classification(
            sample_batch,
            sample_labels,
            correct=correct,
        )
        mlflow.log_figure(fig, f"val_epoch_{epoch}.png")

    acc1_metric.reset()
    acc5_metric.reset()
    return avg_loss, acc1, acc5
