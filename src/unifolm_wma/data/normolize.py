#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from torch import Tensor, nn
from typing import Dict, List


def create_stats_buffers(
    shapes: Dict[str, List[int]],
    modes: Dict[str, str],
    stats: Dict[str, Dict[str, Tensor]] = None,
) -> Dict[str, Dict[str, nn.ParameterDict]]:
    """
    Create buffers per modality (e.g. "observation.image", "action") containing their mean, std, min, max
    statistics.

    Args: (see Normalize and Unnormalize)

    Returns:
        Dict: A Dictionary where keys are modalities and values are `nn.ParameterDict` containing
            `nn.Parameters` set to `requires_grad=False`, suitable to not be updated during backpropagation.
    """
    stats_buffers = {}

    for key, mode in modes.items():
        assert mode in ["mean_std", "min_max"]

        shape = tuple(shapes[key])

        if "image" in key:
            # sanity checks
            assert len(
                shape) == 3, f"number of dimensions of {key} != 3 ({shape=}"
            c, h, w = shape
            assert c < h and c < w, f"{key} is not channel first ({shape=})"
            # override image shape to be invariant to height and width
            shape = (c, 1, 1)

        # Note: we initialize mean, std, min, max to infinity. They should be overwritten
        # downstream by `stats` or `policy.load_state_Dict`, as expected. During forward,
        # we assert they are not infinity anymore.

        if "action" in key:
            target_key = "action"
        elif "state" in key:
            target_key = 'observation.state'
        else:
            target_key = key

        buffer = {}
        if mode == "mean_std":
            mean = torch.ones(shape, dtype=torch.float32) * torch.inf
            std = torch.ones(shape, dtype=torch.float32) * torch.inf
            buffer = nn.ParameterDict({
                "mean":
                nn.Parameter(mean, requires_grad=False),
                "std":
                nn.Parameter(std, requires_grad=False),
            })
        elif mode == "min_max":
            min = torch.ones(shape, dtype=torch.float32) * torch.inf
            max = torch.ones(shape, dtype=torch.float32) * torch.inf
            buffer = nn.ParameterDict({
                "min":
                nn.Parameter(min, requires_grad=False),
                "max":
                nn.Parameter(max, requires_grad=False),
            })

        if stats is not None:
            # Note: The clone is needed to make sure that the logic in save_pretrained doesn't see duplicated
            # tensors anywhere (for example, when we use the same stats for normalization and
            # unnormalization). See the logic here
            if mode == "mean_std":
                buffer["mean"].data = stats[target_key]["mean"].clone()
                buffer["std"].data = stats[target_key]["std"].clone()
            elif mode == "min_max":
                buffer["min"].data = stats[target_key]["min"].clone()
                buffer["max"].data = stats[target_key]["max"].clone()

        stats_buffers[key] = buffer
    return stats_buffers


def _no_stats_error_str(name: str) -> str:
    return (
        f"`{name}` is infinity. You should either initialize with `stats` as an argument, or use a "
        "pretrained model.")


class Normalize(nn.Module):
    """Normalizes data (e.g. "observation.image") for more stable and faster convergence during training."""

    def __init__(
        self,
        shapes: Dict[str, List[int]],
        modes: Dict[str, str],
        stats: Dict[str, Dict[str, Tensor]] = None,
    ):
        """
        Args:
            shapes (Dict): A Dictionary where keys are input modalities (e.g. "observation.image") and values
            are their shapes (e.g. `[3,96,96]`]). These shapes are used to create the tensor buffer containing
            mean, std, min, max statistics. If the provided `shapes` contain keys related to images, the shape
            is adjusted to be invariant to height and width, assuming a channel-first (c, h, w) format.
            modes (Dict): A Dictionary where keys are output modalities (e.g. "observation.image") and values
                are their normalization modes among:
                    - "mean_std": subtract the mean and divide by standard deviation.
                    - "min_max": map to [-1, 1] range.
            stats (Dict, optional): A Dictionary where keys are output modalities (e.g. "observation.image")
                and values are Dictionaries of statistic types and their values (e.g.
                `{"mean": torch.randn(3,1,1)}, "std": torch.randn(3,1,1)}`). If provided, as expected for
                training the model for the first time, these statistics will overwrite the default buffers. If
                not provided, as expected for finetuning or evaluation, the default buffers should to be
                overwritten by a call to `policy.load_state_Dict(state_Dict)`. That way, initializing the
                dataset is not needed to get the stats, since they are already in the policy state_Dict.
        """
        super().__init__()
        self.shapes = shapes
        self.modes = modes
        self.stats = stats
        stats_buffers = create_stats_buffers(shapes, modes, stats)
        for key, buffer in stats_buffers.items():
            setattr(self, "buffer_" + key.replace(".", "_"), buffer)

    @torch.no_grad()
    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        for key, mode in self.modes.items():
            if key not in batch:
                continue

            buffer = getattr(self, "buffer_" + key.replace(".", "_"))

            if mode == "mean_std":
                mean = buffer["mean"]
                std = buffer["std"]

                assert not torch.isinf(mean).any(), _no_stats_error_str("mean")
                assert not torch.isinf(std).any(), _no_stats_error_str("std")
                batch[key] = (batch[key] - mean) / (std + 1e-8)
            elif mode == "min_max":
                min = buffer["min"]
                max = buffer["max"]

                assert not torch.isinf(min).any(), _no_stats_error_str("min")
                assert not torch.isinf(max).any(), _no_stats_error_str("max")
                # normalize to [0,1]
                batch[key] = (batch[key] - min) / (max - min + 1e-8)
                # normalize to [-1, 1]
                batch[key] = batch[key] * 2 - 1
            else:
                raise ValueError(mode)
        return batch


class Unnormalize(nn.Module):
    """
    Similar to `Normalize` but unnormalizes output data (e.g. `{"action": torch.randn(b,c)}`) in their
    original range used by the environment.
    """

    def __init__(
        self,
        shapes: Dict[str, List[int]],
        modes: Dict[str, str],
        stats: Dict[str, Dict[str, Tensor]] = None,
    ):
        """
        Args:
            shapes (Dict): A Dictionary where keys are input modalities (e.g. "observation.image") and values
            are their shapes (e.g. `[3,96,96]`]). These shapes are used to create the tensor buffer containing
            mean, std, min, max statistics. If the provided `shapes` contain keys related to images, the shape
            is adjusted to be invariant to height and width, assuming a channel-first (c, h, w) format.
            modes (Dict): A Dictionary where keys are output modalities (e.g. "observation.image") and values
                are their normalization modes among:
                    - "mean_std": subtract the mean and divide by standard deviation.
                    - "min_max": map to [-1, 1] range.
            stats (Dict, optional): A Dictionary where keys are output modalities (e.g. "observation.image")
                and values are Dictionaries of statistic types and their values (e.g.
                `{"mean": torch.randn(3,1,1)}, "std": torch.randn(3,1,1)}`). If provided, as expected for
                training the model for the first time, these statistics will overwrite the default buffers. If
                not provided, as expected for finetuning or evaluation, the default buffers should to be
                overwritten by a call to `policy.load_state_Dict(state_Dict)`. That way, initializing the
                dataset is not needed to get the stats, since they are already in the policy state_Dict.
        """
        super().__init__()
        self.shapes = shapes
        self.modes = modes
        self.stats = stats
        stats_buffers = create_stats_buffers(shapes, modes, stats)
        for key, buffer in stats_buffers.items():
            setattr(self, "buffer_" + key.replace(".", "_"), buffer)

    @torch.no_grad()
    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        for key, mode in self.modes.items():
            if key not in batch:
                continue

            buffer = getattr(self, "buffer_" + key.replace(".", "_"))

            if mode == "mean_std":
                mean = buffer["mean"]
                std = buffer["std"]
                assert not torch.isinf(mean).any(), _no_stats_error_str("mean")
                assert not torch.isinf(std).any(), _no_stats_error_str("std")
                batch[key] = batch[key] * std + mean
            elif mode == "min_max":
                min = buffer["min"]
                max = buffer["max"]
                assert not torch.isinf(min).any(), _no_stats_error_str("min")
                assert not torch.isinf(max).any(), _no_stats_error_str("max")
                batch[key] = (batch[key] + 1) / 2
                batch[key] = batch[key] * (max - min) + min
            else:
                raise ValueError(mode)
        return batch
