import logging
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import requests
import torch
import torchvision
from datasets import load_from_disk
from datasets.features.features import register_feature
from safetensors.torch import load_file

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class LongConnectionClient:
    def __init__(self, base_url):
        self.session = requests.Session()
        self.base_url = base_url

    def send_post(self, endpoint, json_data):
        """send POST request to  endpoint"""
        url = f"{self.base_url}{endpoint}"
        response = None
        while True:
            try:
                response = self.session.post(url, json=json_data)
                if response.status_code == 200:
                    data = response.json()
                    if data["result"] == "ok":
                        response = data
                        break
                    else:
                        logging.info(data["desc"])

                time.sleep(1)
            except Exception as e:
                logging.error(f"An error occurred: {e}")
                logging.error(traceback.format_exc())

        return response

    def close(self):
        """ "close session"""
        self.session.close()

    def predict_action(self, language_instruction, batch) -> torch.Tensor:
        # collect data
        data = {
            "language_instruction": language_instruction,
            "observation.state": torch.stack(list(batch["observation.state"])).tolist(),
            "observation.images.top": torch.stack(list(batch["observation.images.top"])).tolist(),
            "action": torch.stack(list(batch["action"])).tolist(),
        }

        # send data
        endpoint = "/predict_action"
        response = self.send_post(endpoint, data)
        # action = torch.tensor(response['action']).unsqueeze(0)
        action = torch.tensor(response["action"])
        return action


class ACTTemporalEnsembler:
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int, exe_steps: int) -> None:
        """Temporal ensembling as described in Algorithm 2 of https://arxiv.org/abs/2304.13705.

        The weights are calculated as wᵢ = exp(-temporal_ensemble_coeff * i) where w₀ is the oldest action.
        They are then normalized to sum to 1 by dividing by Σwᵢ. Here's some intuition around how the
        coefficient works:
            - Setting it to 0 uniformly weighs all actions.
            - Setting it positive gives more weight to older actions.
            - Setting it negative gives more weight to newer actions.
        NOTE: The default value for `temporal_ensemble_coeff` used by the original ACT work is 0.01. This
        results in older actions being weighed more highly than newer actions (the experiments documented in
        https://github.com/huggingface/lerobot/pull/319 hint at why highly weighing new actions might be
        detrimental: doing so aggressively may diminish the benefits of action chunking).

        Here we use an online method for computing the average rather than caching a history of actions in
        order to compute the average offline. For a simple 1D sequence it looks something like:

        ```
        import torch

        seq = torch.linspace(8, 8.5, 100)
        print(seq)

        m = 0.01
        exp_weights = torch.exp(-m * torch.arange(len(seq)))
        print(exp_weights)

        # Calculate offline
        avg = (exp_weights * seq).sum() / exp_weights.sum()
        print("offline", avg)

        # Calculate online
        for i, item in enumerate(seq):
            if i == 0:
                avg = item
                continue
            avg *= exp_weights[:i].sum()
            avg += item * exp_weights[i]
            avg /= exp_weights[:i+1].sum()
        print("online", avg)
        ```
        """
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.exe_steps = exe_steps
        self.reset()

    def reset(self):
        """Resets the online computation variables."""
        self.ensembled_actions = None
        # (chunk_size,) count of how many actions are in the ensemble for each time step in the sequence.
        self.ensembled_actions_count = None

    def update(self, actions):
        """
        Takes a (batch, chunk_size, action_dim) sequence of actions, update the temporal ensemble for all
        time steps, and pop/return the next batch of actions in the sequence.
        """
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
        if self.ensembled_actions is None:
            # Initializes `self._ensembled_action` to the sequence of actions predicted during the first
            # time step of the episode.
            self.ensembled_actions = actions.clone()
            # Note: The last dimension is unsqueeze to make sure we can broadcast properly for tensor
            # operations later.
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=self.ensembled_actions.device
            )
        else:
            # self.ensembled_actions will have shape (batch_size, chunk_size - 1, action_dim). Compute
            # the online update for those entries.
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += (
                actions[:, : -self.exe_steps] * self.ensemble_weights[self.ensembled_actions_count]
            )
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
            # The last action, which has no prior online average, needs to get concatenated onto the end.
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -self.exe_steps :]], dim=1)
            self.ensembled_actions_count = torch.cat(
                # [self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-self.exe_steps:])]
                [
                    self.ensembled_actions_count,
                    torch.ones((self.exe_steps, 1), dtype=torch.long, device=self.ensembled_actions_count.device),
                ]
            )
        # "Consume" the first action.

        actions, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, : self.exe_steps],
            self.ensembled_actions[:, self.exe_steps :],
            self.ensembled_actions_count[self.exe_steps :],
        )
        return actions


@dataclass
class VideoFrame:
    """
    Provides a type for a dataset containing video frames.

    Example:

    ```python
    data_dict = [{"image": {"path": "videos/episode_0.mp4", "timestamp": 0.3}}]
    features = {"image": VideoFrame()}
    Dataset.from_dict(data_dict, features=Features(features))
    ```
    """

    pa_type: ClassVar[Any] = pa.struct({"path": pa.string(), "timestamp": pa.float32()})
    _type: str = field(default="VideoFrame", init=False, repr=False)

    def __call__(self):
        return self.pa_type


with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        "'register_feature' is experimental and might be subject to breaking changes in the future.",
        category=UserWarning,
    )
    # to make VideoFrame available in HuggingFace `datasets`
    register_feature(VideoFrame, "VideoFrame")


def get_image(cam_list, target_shape=None, save_image=False):
    curr_images = []
    for cam in cam_list:
        color, _ = cam.get_frame()
        if save_image:
            cv2.imwrite("/home/world-model-x/output.png", color)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        if target_shape:
            color = cv2.resize(color, target_shape)
        curr_images.append(color)
    curr_images = np.stack(curr_images, axis=0)
    return curr_images


def load_action_from_dataset(dataset_dir, episode_id):
    data = load_from_disk(dataset_dir + "/train")
    episode_data = load_file(dataset_dir + "/meta_data/episode_data_index.safetensors")
    start_id = episode_data["from"][episode_id]
    end_id = episode_data["to"][episode_id]
    actions = torch.FloatTensor(data["action"][start_id:end_id])
    return actions


def load_stats_from_prompt_dir(dataset_dir, prompt_dir, subdir=""):
    dataset_dir += subdir + "/meta_data"
    stats = load_file(dataset_dir + "/stats.safetensors")
    return stats


def populate_queues(queues, batch):
    for key in batch:
        # Ignore keys not in the queues already (leaving the responsibility to the caller to make sure the
        # queues have the keys they want).
        if key not in queues:
            continue
        if len(queues[key]) != queues[key].maxlen:
            # initialize by copying the first observation several times until the queue is full
            while len(queues[key]) != queues[key].maxlen:
                queues[key].append(batch[key])
        else:
            # add latest observation to the queue
            queues[key].append(batch[key])
    return queues


def action_safe_checking(action, action_max, action_min, threshold=0.01):
    over_max = any(action - threshold > action_max.cpu().numpy())
    over_min = any(action + threshold < action_min.cpu().numpy())
    return not (over_max or over_min)


def get_init_pose(dataset_dir, start_id=0):
    # load all par
    dataset_dir_path = Path(dataset_dir) / "data" / "chunk-000"
    parquet_files = list(dataset_dir_path.glob("*.parquet"))
    parquet_files = sorted([str(f) for f in parquet_files])
    first_rows = [pd.read_parquet(f, engine="pyarrow").iloc[[0]] for f in parquet_files]
    df = pd.concat(first_rows, ignore_index=True)
    action_array = np.stack(df["action"].values)
    init_pose = action_array[192:193, ...]
    return init_pose


def save_image(obs, num_step=None, output_dir=None):
    rgb_image = cv2.cvtColor(obs.observation["images"]["cam_left_high"], cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"{output_dir}/top_{num_step:06d}.png", rgb_image)


def log_to_tensorboard(writer, data, tag, fps=10):
    if isinstance(data, torch.Tensor) and data.dim() == 5:
        video = data
        n = video.shape[0]
        video = video.permute(2, 0, 1, 3, 4)  # t,n,c,h,w
        frame_grids = [
            torchvision.utils.make_grid(framesheet, nrow=int(n), padding=0) for framesheet in video
        ]  # [3, n*h, 1*w]
        grid = torch.stack(frame_grids, dim=0)  # stack in temporal dim [t, 3, n*h, w]
        grid = (grid + 1.0) / 2.0
        grid = grid.unsqueeze(dim=0)
        writer.add_video(tag, grid, fps=fps)
