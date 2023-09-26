import argparse
import json
import os
from typing import Any, Dict, Tuple

import pandas as pd
import requests


def clean_ground_truth(gt_path: str, root_dir: str) -> pd.DataFrame:
    """Obtains cleaned ground truth data based on what images are on disk.

    Args:
        gt_path (str): Path to ground truth csv file.
        root_dir (str): Path to directory with images.

    Returns:
        gt_df (pd.DataFrame): Dataframe with cleaned ground truth.
    """
    gt_df = pd.read_csv(gt_path)

    # Drop video folders that we do not exists in the img root dir
    if not os.path.exists(root_dir):
        print("root dir path does not exists")
        exit()
    for video in gt_df["folder_name"]:
        video_path = os.path.join(root_dir, video)
        if os.path.exists(video_path) is False:
            gt_df = gt_df.drop(gt_df[gt_df["folder_name"] == video].index)

    # Print some stats
    classes = gt_df["label"].unique().tolist()
    print(f"GT: Found data from {len(classes)} classes")
    video_clips = gt_df["folder_name"].unique().tolist()
    videos = gt_df["youtube_id"].unique().tolist()
    print(f"GT: Found {len(video_clips)} clips from {len(videos)} videos")

    return gt_df


def get_label_mapping() -> Dict[str, int]:
    """Downloads the available/most used label mapping for Kinetics-400.

    Returns:
        id_by_label (Dict[str, int]): Mapping from label name to class id.
    """
    try:
        r = requests.get(
            "https://gist.githubusercontent.com/willprice/"
            "f19da185c9c5f32847134b87c1960769/raw/"
            "9dc94028ecced572f302225c49fcdee2f3d748d8/"
            "kinetics_400_labels.csv",
            timeout=10,
        )
    except requests.exceptions.RequestException as e:
        print("Could not download the label mapping.")
        raise SystemExit(e)

    # Convert to dataframe and finally to mapping dictionary
    df = pd.DataFrame(
        [row.split(",") for row in r.text.split("\n")],
        columns=["id", "name"],
    )
    df.drop(index=df.index[0], axis=0, inplace=True)
    df.drop(index=df.index[-1], axis=0, inplace=True)
    df["id"] = df["id"].astype(int)
    return pd.Series(df.id.values, index=df.name).to_dict()


def perform_split(
    gt_df: pd.DataFrame,
    dst_folder: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Performs a reproducable train/val/test split on the ground truth.

    Args:
        gt_df (pd.DataFrame): Dataframe with cleaned ground truth.
        dst_folder (str): Destination folder for train/val/test files.

    Returns:
        train, val, test (Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame])
    """
    gt_df = gt_df.sort_values("folder_name")
    # Set random state to make split reproducable
    gt_df = gt_df.sample(frac=1, random_state=42)
    first_split = int(0.8 * len(gt_df))
    second_split = int(0.9 * len(gt_df))
    train = gt_df.iloc[:first_split, :]
    val = gt_df.iloc[first_split:second_split, :]
    test = gt_df.iloc[second_split:, :]

    # Store the splits also to csv files for easier sharing
    split_folder = os.path.join(dst_folder, "splits")
    os.makedirs(split_folder, exist_ok=True)

    train.to_csv(os.path.join(split_folder, "train.csv"), index=False)
    val.to_csv(os.path.join(split_folder, "val.csv"), index=False)
    test.to_csv(os.path.join(split_folder, "test.csv"), index=False)

    return train, val, test


def get_annotations(
    df: pd.DataFrame,
    root_dir: str,
    id_by_label: Dict[str, int],
) -> list[dict[str, Any]]:
    """Stores data annotation information in dictionary format.

    Args:
        df (pd.DataFrame): Dataframe with ground truth for the split.
        root_dir (str): Path to directory with images.
        id_by_label (Dict[str, int]): Mapping from label name to class id.

    Returns:
        out_annotations (List[Dict]): The annotation information for the
        split.
    """
    out_annotations = []
    folder_names = df["folder_name"].tolist()
    labels = df["label"].tolist()
    for folder_name, label in zip(folder_names, labels):
        vid_path = os.path.join(root_dir, folder_name)
        n_frames = len(
            [
                x
                for x in os.listdir(vid_path)
                if "frame" in x
                and x.endswith(
                    ".jpg",
                )
            ],
        )
        if n_frames != 40:
            print(f"Warning! {vid_path.split('/')[-1]} has {n_frames} frames")
        sample = {
            "video": folder_name,
            "segment": (0, n_frames - 1),
            "target": id_by_label[label],
        }
        out_annotations.append(sample)
    return out_annotations


def main(ground_truth_csv: str, img_root_dir: str, dst_folder: str):
    """Main function that converts the ground truth and makes split.

    Args:
        ground_truth_csv (str): The ground truth csv path filename.
        img_root_dir (str): The root directory with images.
        dst_folder (str): Destination folder.
    """
    out_dataset = {}
    os.makedirs(dst_folder, exist_ok=True)

    ground_truth = clean_ground_truth(ground_truth_csv, img_root_dir)
    id_by_label = get_label_mapping()
    train, val, test = perform_split(ground_truth, dst_folder)

    out_dataset["categories"] = id_by_label
    train_ann = get_annotations(train, img_root_dir, id_by_label)
    val_ann = get_annotations(val, img_root_dir, id_by_label)
    test_ann = get_annotations(test, img_root_dir, id_by_label)
    out_dataset["train"] = train_ann  # type: ignore[assignment]
    out_dataset["val"] = val_ann  # type: ignore[assignment]
    out_dataset["test"] = test_ann  # type: ignore[assignment]

    dst_json_path = os.path.join(dst_folder, "annotations.json")
    with open(dst_json_path, "w") as dst_file:
        json.dump(out_dataset, dst_file)
    print(f"Done! Data annotation json stored at {dst_json_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Making K400 Mini database")

    parser.add_argument(
        "--gt_csv",
        default="../example_data/k400tiny_groundtruth.csv",
        type=str,
        help="path to ground truth csv file",
    )

    parser.add_argument(
        "--root_dir",
        default="../example_data/k400tiny_images",
        type=str,
        help="path to images directory",
    )

    parser.add_argument(
        "--dst_folder",
        default="../k400tiny",
        type=str,
        help="destination folder for output annotation json file",
    )

    args = parser.parse_args()

    main(args.gt_csv, args.root_dir, args.dst_folder)
