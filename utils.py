import json
import os
from typing import Dict, List, Optional, Union

import matplotlib.figure
import matplotlib.pyplot as plt
import torch


def export_to_onnx(
    experiment_name: str,
    model: torch.nn.Module,
    id_by_label: Dict[str, int],
    onnx_export_dir: str,
    clip_len: int,
    crop_size: int,
):
    """Export model to onnx format.

    Args:
        experiment_name (str): Name of experiment.
        model (torch.nn.Module): Model on CPU.
        id_by_label (Dict[str, int]): Annotation dictionary.
        onnx_export_dir (str): Directory path to save export.
        clip_len (int): number of frames per clip.
        crop_size (int): the min crop size used.
    """
    dummy_input = torch.randn(1, 3, clip_len, crop_size, crop_size)
    onnx_save_path = os.path.join(
        onnx_export_dir,
        f"{experiment_name}.onnx",
    )
    torch.onnx.export(
        model,
        dummy_input,
        onnx_save_path,
        export_params=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    dst_json_path = os.path.join(onnx_export_dir, "id_by_label.json")
    with open(dst_json_path, "w") as dst_file:
        json.dump(id_by_label, dst_file)
    print("ONNX model and id_by_label json saved")


@torch.no_grad()
def visualize_classification(
    batch: torch.Tensor,
    target_names: List[str],
    preds_names: Optional[List[List[str]]] = None,
    preds_scores: Optional[List[List[float]]] = None,
    correct: Optional[Union[List[bool], torch.BoolTensor]] = None,
) -> matplotlib.figure.Figure:
    """Creates a matplotlib figure with a grid of images with labels.

    Args:
        batch (torch.Tensor): The (B, C, T, H, W) or (B, C, H, W) tensor to
            visualize B images from.
        target_names (List[str]): The ground truth label names for each sample
            in the batch.
        preds_names (Optional[List[List[str]]]): Label names of top-5
            predictions for each sample.
        preds_scores (Optional[List[List[float]]]): Scores of top-5
            predictions for each sample.
        correct (Optional[Union[List[bool], torch.BoolTensor]]): Booleans
            to indicate whether a prediction was correct or not. Defaults to
            None.

    Returns:
        fig (matplotlib.pyplot.figure): The output figure.
    """
    if batch.dim() == 5:
        # For videos, take frame somewhere in the middle
        middle_frame_index = batch.shape[2] // 2
        batch = batch[:, :, middle_frame_index, :, :]

    # Make dynamic figure height to accomodate different batch sizes
    batch_size = batch.shape[0]
    n_cols = 4
    if batch_size < n_cols:
        n_cols = batch_size
    n_rows = batch_size // n_cols
    if batch_size % n_cols != 0:
        n_rows += 1

    fig = plt.figure(figsize=((3 * n_cols), (3 * n_rows)))

    # Need more space between images to show predictions if given
    if preds_names is not None and preds_scores is not None:
        horizontal_space = 1.0
    else:
        horizontal_space = 0.25
    plt.subplots_adjust(hspace=horizontal_space)

    for i in range(batch_size):
        if preds_names is not None and preds_scores is not None:
            pred_name = preds_names[i]
            pred_score = preds_scores[i]
        ax = plt.subplot(n_rows, n_cols, i + 1)
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)
        ax.set_xticks([])
        ax.set_yticks([])
        if preds_names is not None and preds_scores is not None:
            plt.text(
                n_rows,
                n_cols + batch.shape[-2] + 75,
                f"{target_names[i]}",
                fontsize="small",
                color="blue",
            )
            for j, (name, score) in enumerate(zip(pred_name, pred_score)):
                if j == 0 and target_names[i] == name:
                    display_color = "green"
                else:
                    display_color = "red"
                plt.text(
                    n_rows,
                    n_cols + batch.shape[-2] + 150 + (j * 75),
                    f"{name}: {(score*100):.2f}%",
                    fontsize="x-small",
                    color=display_color,
                )
        else:
            t = ax.set_title(target_names[i], wrap=True)
            if correct is not None:
                if correct[i]:
                    ax.title.set_color("green")
                else:
                    ax.title.set_color("red")
            t._get_wrap_line_width = lambda: 200  # type: ignore[attr-defined]

        out = batch[i].byte().permute(1, 2, 0).numpy()
        plt.imshow(out)

    return fig
