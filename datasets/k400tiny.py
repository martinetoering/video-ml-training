import json
import os
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class K400tiny(Dataset):
    """K400 tiny dataset."""

    def __init__(
        self,
        root_dir: str,
        annotation_path: str,
        mode: str,
        transform: Optional[Callable] = None,
    ):
        """K400 tiny dataset.

        Arguments:
            root_dir (str): Directory with the images.
            annotation_path (str): Path to the json file with annotations.
            mode (str): Train, val or test mode.
            transform (Optional[Callable]): Optional transform to be applied.
        """
        with open(annotation_path, "r") as file:
            annotation_dict = json.load(file)

        self.id_by_label = annotation_dict["categories"]
        self.label_by_id = {v: k for k, v in self.id_by_label.items()}
        self.dataset = annotation_dict[mode]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        video_folder = sample["video"]
        frame_start, frame_end = sample["segment"]
        target = sample["target"]
        frames = []
        for frame_num in range(frame_start, frame_end + 1):
            image_path = os.path.join(
                self.root_dir,
                video_folder,
                f"frame{frame_num:02}.jpg",
            )
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.asarray(image)
            image_tensor = torch.from_numpy(image)
            frames.append(image_tensor)
        video = torch.stack(frames, 0)
        # Return in T, C, H, W
        video = video.permute(0, 3, 1, 2)
        if self.transform:
            video = self.transform(video)
        return video, target
