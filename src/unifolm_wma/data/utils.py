import torch

from huggingface_hub import hf_hub_download, snapshot_download
from typing import Dict, List, Union
from pathlib import Path
from safetensors.torch import load_file

def unflatten_dict(d, sep="/"):
    outdict = {}
    for key, value in d.items():
        parts = key.split(sep)
        d = outdict
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return outdict


def load_episode_data_index(repo_id, version, root) -> Dict[str, torch.Tensor]:
    """episode_data_index contains the range of indices for each episode

    Example:
    ```python
    from_id = episode_data_index["from"][episode_id].item()
    to_id = episode_data_index["to"][episode_id].item()
    episode_frames = [dataset[i] for i in range(from_id, to_id)]
    ```
    """
    if root is not None:
        path = Path(
            root) / repo_id / "meta_data" / "episode_data_index.safetensors"
    else:
        path = hf_hub_download(repo_id,
                               "meta_data/episode_data_index.safetensors",
                               repo_type="dataset",
                               revision=version)

    return load_file(path)


def load_stats(repo_id, version, root) -> Dict[str, Dict[str, torch.Tensor]]:
    """stats contains the statistics per modality computed over the full dataset, such as max, min, mean, std

    Example:
    ```python
    normalized_action = (action - stats["action"]["mean"]) / stats["action"]["std"]
    ```
    """
    if root is not None:
        path = Path(root) / repo_id / "meta_data" / "stats.safetensors"
    else:
        path = hf_hub_download(repo_id,
                               "meta_data/stats.safetensors",
                               repo_type="dataset",
                               revision=version)

    stats = load_file(path)
    return unflatten_dict(stats)
