#!/usr/bin/env python3
"""
Convert a LeRobot v3 dataset (G1_WBT_Brainco_Pickup_Pillow format) into the
layout expected by unifolm-wma's WMAData loader.

Required output structure under DATASET_DIR:
    {dataset_name}.csv
    videos/{dataset_name}/{episode_idx}.mp4   (symlinks to LeRobot videos)
    transitions/{dataset_name}/{episode_idx}.h5
    transitions/{dataset_name}/meta_data/stats.safetensors

Usage:
    python scripts/prepare_lerobot_dataset.py \
        --dataset_dir /path/to/G1_WBT_Brainco_Pickup_Pillow/snapshot \
        --dataset_name G1_WBT_Brainco_Pickup_Pillow \
        --camera_key observation.images.head_stereo_left
"""

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to the LeRobot dataset snapshot directory.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="G1_WBT_Brainco_Pickup_Pillow",
        help="Name of the dataset (used for directory/file naming).",
    )
    parser.add_argument(
        "--camera_key",
        type=str,
        default="observation.images.head_stereo_left",
        help="Which camera stream to use for the world-model video.",
    )
    parser.add_argument(
        "--action_key",
        type=str,
        default="action.ee_action",
        help="Column in parquet to use as the action vector.",
    )
    parser.add_argument(
        "--state_key",
        type=str,
        default="observation.state.ee_state",
        help="Column in parquet to use as the observation state vector.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="Pick up the pillow and place it on the sofa",
        help="Language instruction for all episodes.",
    )
    return parser.parse_args()


def make_csv(dataset_dir: Path, dataset_name: str, num_episodes: int, instruction: str):
    rows = []
    for i in range(num_episodes):
        rows.append({
            "data_dir": dataset_name,
            "videoid": str(i),
            "instruction": instruction,
            "embodiment": "x",
        })
    df = pd.DataFrame(rows)
    csv_path = dataset_dir / f"{dataset_name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"[CSV] Written {len(df)} rows → {csv_path}")


def make_video_clips(dataset_dir: Path, dataset_name: str, episodes_df: pd.DataFrame, camera_key: str):
    """Extract per-episode video clips from the concatenated LeRobot video files."""
    import subprocess

    video_out_dir = dataset_dir / "videos" / dataset_name
    video_out_dir.mkdir(parents=True, exist_ok=True)

    for _, row in episodes_df.iterrows():
        ep_idx = int(row["episode_index"])
        chunk_idx = int(row[f"videos/{camera_key}/chunk_index"])
        file_idx = int(row[f"videos/{camera_key}/file_index"])
        t_start = float(row[f"videos/{camera_key}/from_timestamp"])
        t_end = float(row[f"videos/{camera_key}/to_timestamp"])
        duration = t_end - t_start

        src = (
            dataset_dir
            / "videos"
            / camera_key
            / f"chunk-{chunk_idx:03d}"
            / f"file-{file_idx:03d}.mp4"
        )
        dst = video_out_dir / f"{ep_idx}.mp4"

        if dst.exists():
            continue

        # Use ffmpeg to trim the clip; -c copy for fast remux if codec compatible,
        # fall back to re-encode with libx264 for broad decord compatibility.
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{t_start:.6f}",
            "-t", f"{duration:.6f}",
            "-i", str(src),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
            "-an",
            str(dst),
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            print(f"  [WARN] ffmpeg failed for episode {ep_idx}: {result.stderr.decode()[-200:]}")

        if ep_idx % 20 == 0:
            print(f"  episode {ep_idx}/{len(episodes_df)} ...")

    print(f"[Video] Extracted {len(episodes_df)} clips to {video_out_dir}")


def make_h5_files(
    dataset_dir: Path,
    dataset_name: str,
    episodes_df: pd.DataFrame,
    data_df: pd.DataFrame,
    action_key: str,
    state_key: str,
):
    out_dir = dataset_dir / "transitions" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for _, row in episodes_df.iterrows():
        ep_idx = int(row["episode_index"])
        from_idx = int(row["dataset_from_index"])
        to_idx = int(row["dataset_to_index"])

        ep_data = data_df.iloc[from_idx:to_idx]

        actions = np.stack(ep_data[action_key].values).astype(np.float32)
        states = np.stack(ep_data[state_key].values).astype(np.float32)

        h5_path = out_dir / f"{ep_idx}.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("action", data=actions)
            f.create_dataset("observation.state", data=states)
            f.attrs["action_type"] = action_key
            f.attrs["state_type"] = state_key

    print(f"[H5] Written {len(episodes_df)} HDF5 files to {out_dir}")


def make_stats_safetensors(dataset_dir: Path, dataset_name: str, action_key: str, state_key: str):
    stats_json_path = dataset_dir / "meta" / "stats.json"
    with open(stats_json_path) as f:
        stats_json = json.load(f)

    meta_dir = dataset_dir / "transitions" / dataset_name / "meta_data"
    meta_dir.mkdir(parents=True, exist_ok=True)

    tensors = {}
    for stat_field in ("min", "max", "mean", "std"):
        tensors[f"action/{stat_field}"] = torch.tensor(
            stats_json[action_key][stat_field], dtype=torch.float32
        )
        tensors[f"observation.state/{stat_field}"] = torch.tensor(
            stats_json[state_key][stat_field], dtype=torch.float32
        )

    out_path = meta_dir / "stats.safetensors"
    save_file(tensors, str(out_path))
    print(f"[Stats] Written stats.safetensors → {out_path}")
    print(f"        action/{{}}: shape {tensors['action/max'].shape}")
    print(f"        observation.state/{{}}: shape {tensors['observation.state/max'].shape}")


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)

    print(f"Dataset dir : {dataset_dir}")
    print(f"Dataset name: {args.dataset_name}")
    print(f"Camera key  : {args.camera_key}")
    print(f"Action key  : {args.action_key}")
    print(f"State key   : {args.state_key}")

    # Load episodes metadata
    episodes_df = pd.read_parquet(dataset_dir / "meta" / "episodes")
    num_episodes = len(episodes_df)
    print(f"\nFound {num_episodes} episodes.")

    # Load full frame data (all chunks/files)
    data_parts = []
    for chunk_dir in sorted((dataset_dir / "data").iterdir()):
        for parquet_file in sorted(chunk_dir.glob("*.parquet")):
            data_parts.append(pd.read_parquet(parquet_file))
    data_df = pd.concat(data_parts, ignore_index=True)
    print(f"Loaded {len(data_df)} total frames from parquet.")

    print("\n--- Generating CSV ---")
    make_csv(dataset_dir, args.dataset_name, num_episodes, args.instruction)

    print("\n--- Extracting per-episode video clips ---")
    make_video_clips(dataset_dir, args.dataset_name, episodes_df, args.camera_key)

    print("\n--- Creating H5 transition files ---")
    make_h5_files(dataset_dir, args.dataset_name, episodes_df, data_df, args.action_key, args.state_key)

    print("\n--- Creating stats.safetensors ---")
    make_stats_safetensors(dataset_dir, args.dataset_name, args.action_key, args.state_key)

    print("\nDone. Dataset is ready for unifolm-wma.")


if __name__ == "__main__":
    main()
