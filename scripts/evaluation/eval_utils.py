import torch
import warnings
import torchvision
import sys
import pyarrow as pa
import logging

from dataclasses import dataclass, field
from typing import Dict, Any, ClassVar, Deque, Mapping, Union
from datasets.features.features import register_feature
from torch.utils.tensorboard.writer import SummaryWriter

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


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

    pa_type: ClassVar[Any] = pa.struct({
        "path": pa.string(),
        "timestamp": pa.float32()
    })
    _type: str = field(default="VideoFrame", init=False, repr=False)

    def __call__(self):
        return self.pa_type


with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        "'register_feature' is experimental and might be subject to breaking changes in the future.",
        category=UserWarning,
    )
    register_feature(VideoFrame, "VideoFrame")


def populate_queues(
        queues: Dict[str, Deque[Any]],
        batch: Mapping[str, Any]) -> Dict[str, Deque[Any]]:

    for key in batch:
        if key not in queues:
            continue
        if len(queues[key]) != queues[key].maxlen:
            while len(queues[key]) != queues[key].maxlen:
                queues[key].append(batch[key])
        else:
            queues[key].append(batch[key])
    return queues


def log_to_tensorboard(
        writer: SummaryWriter,
        data: Union[torch.Tensor, Any],
        tag: str,
        fps: int = 10) -> None:
    if isinstance(data, torch.Tensor) and data.dim() == 5:
        video = data
        n = video.shape[0]
        video = video.permute(2, 0, 1, 3, 4)
        frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(n), padding=0) for framesheet in video]
        grid = torch.stack(frame_grids, dim=0)
        grid = (grid + 1.0) / 2.0
        grid = grid.unsqueeze(dim=0)
        writer.add_video(tag, grid, fps=fps)
