import torch
import os
import random
import pandas as pd
import h5py

from decord import VideoReader, cpu
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path

from unifolm_wma.data.utils import load_stats
from unifolm_wma.data.normolize import Normalize, Unnormalize


class WMAData(Dataset):
    """
    Assuming the following dataset structure:
    dataset_dir/
        ├── videos
        │     ├──dataset_name
        │     │   ├──camera_view_dir
        │     │       ├── 0.mp4
        │     │       ├── 1.mp4
        │     │       └── ...
        │     └── ...
        ├── transitions
        │    ├── dataset_name
        │        ├── meta_data
        │        ├── 0.h5
        │        ├── 1.h5
        │        └── ...
        └──  dataset_name.csv
    """

    def __init__(
        self,
        meta_path,
        data_dir,
        subsample=None,
        video_length=16,
        resolution=[256, 512],
        frame_stride=1,
        frame_stride_min=1,
        spatial_transform=None,
        crop_resolution=None,
        fps_max=None,
        load_raw_resolution=False,
        fixed_fps=None,
        random_fs=False,
        cond_robot_label_prob=0.0,
        transition_dir=None,
        dataset_name=None,
        normalization_mode='min_max',
        individual_normalization=False,
        n_obs_steps=1,
        max_action_dim=7,
        max_state_dim=7,
    ):
        self.meta_path = meta_path
        self.data_dir = data_dir
        self.subsample = subsample
        self.video_length = video_length
        self.resolution = [resolution, resolution] if isinstance(
            resolution, int) else resolution
        self.fps_max = fps_max
        self.frame_stride = frame_stride
        self.frame_stride_min = frame_stride_min
        self.fixed_fps = fixed_fps
        self.load_raw_resolution = load_raw_resolution
        self.random_fs = random_fs
        self.cond_robot_label_prob = cond_robot_label_prob
        self.transition_dir = transition_dir
        self.dataset_name = dataset_name
        self.max_action_dim = max_action_dim
        self.max_state_dim = max_state_dim

        self._load_metadata()
        if spatial_transform is not None:
            if spatial_transform == "random_crop":
                self.spatial_transform = transforms.RandomCrop(crop_resolution)
            elif spatial_transform == "center_crop":
                self.spatial_transform = transforms.Compose([
                    transforms.CenterCrop(resolution),
                ])
            elif spatial_transform == "resize_center_crop":
                self.spatial_transform = transforms.Compose([
                    transforms.Resize(min(self.resolution)),
                    transforms.CenterCrop(self.resolution),
                ])
            elif spatial_transform == "resize":
                self.spatial_transform = transforms.Resize(self.resolution)
            else:
                raise NotImplementedError
        else:
            self.spatial_transform = None

        self.normalization_mode = normalization_mode
        self.individual_normalization = individual_normalization
        self.n_obs_steps = n_obs_steps
        self._load_stats()
        if individual_normalization:
            self._init_normalizers()

    def _load_metadata(self):
        metadata = pd.read_csv(self.meta_path, dtype=str)
        if self.subsample is not None:
            metadata = metadata.sample(self.subsample, random_state=0)

        self.metadata = metadata
        # drop the rows contain NaN values
        self.metadata.dropna(inplace=True)
        print(
            f">>> {metadata['data_dir'].iloc[0]}: {len(metadata)} data samples loaded."
        )

    def _load_stats(self):
        self.stats = load_stats(self.dataset_name, None, self.transition_dir)
        print(f">>> {self.metadata['data_dir'].iloc[0]}: data stats loaded.")

    def _init_normalizers(self):
        shape_dict = {
            'pre_action': [self.stats['action']['max'].shape[-1]],
            'action': [self.stats['action']['max'].shape[-1]],
            'observation.state':
            [self.stats['observation.state']['max'].shape[-1]],
            'next.state': [self.stats['observation.state']['max'].shape[-1]]
        }
        normalization_mode_dict = {
            'pre_action': self.normalization_mode,
            'action': self.normalization_mode,
            'observation.state': self.normalization_mode,
            'next.state': self.normalization_mode
        }
        self.normalizer = Normalize(shape_dict, normalization_mode_dict,
                                    self.stats)
        self.unnormalizer = Unnormalize(shape_dict, normalization_mode_dict,
                                        self.stats)
        print(
            f">>> {self.metadata['data_dir'].iloc[0]}: normalizer initiated.")

    def _get_video_path(self, sample):
        rel_video_fp = os.path.join(sample['data_dir'],
                                    str(sample['videoid']) + '.mp4')
        full_video_fp = os.path.join(self.data_dir, 'videos', rel_video_fp)
        return full_video_fp

    def _get_transition_path(self, sample):
        data_dir = Path(sample['data_dir'])
        if self.dataset_name == data_dir.name:
            rel_transition_fp = os.path.join(str(data_dir),
                                             str(sample['videoid']) + '.h5')
        else:
            rel_transition_fp = os.path.join(str(data_dir.parent),
                                             str(sample['videoid']) + '.h5')
        full_transition_fp = os.path.join(self.data_dir, 'transitions',
                                          rel_transition_fp)
        return full_transition_fp

    def get_uni_vec(self, action_state_dict, action_type, state_type):
        if 'pre_action' in action_state_dict:
            action_state_dict['pre_action'], _ = self._map_to_uni_action(
                action_state_dict['pre_action'], action_type)
        if 'action' in action_state_dict:
            action_state_dict['action'], action_state_dict[
                'action_mask'] = self._map_to_uni_action(
                    action_state_dict['action'], action_type)
        if 'observation.state' in action_state_dict:
            action_state_dict['observation.state'], _ = self._map_to_uni_state(
                action_state_dict['observation.state'], state_type)
        if 'next.state' in action_state_dict:
            action_state_dict['next.state'], action_state_dict[
                'state_mask'] = self._map_to_uni_state(
                    action_state_dict['next.state'], state_type)
        return action_state_dict

    def _map_to_uni_action(self, action, action_type):
        action_dim = action.shape[-1]
        uni_action = torch.nn.functional.pad(
            action, (0, self.max_action_dim - action_dim),
            mode='constant',
            value=0)
        uni_action_mask = torch.zeros_like(uni_action)
        uni_action_mask[:, :action_dim] = 1
        return uni_action, uni_action_mask

    def _map_to_uni_state(self, state, state_type):
        state_dim = state.shape[-1]
        uni_state = torch.nn.functional.pad(
            state, (0, self.max_state_dim - state_dim),
            mode='constant',
            value=0)
        uni_state_mask = torch.zeros_like(uni_state)
        uni_state_mask[:, :state_dim] = 1
        return uni_state, uni_state_mask

    def __getitem__(self, index):

        if self.random_fs:
            frame_stride = random.randint(self.frame_stride_min,
                                          self.frame_stride)
        else:
            frame_stride = self.frame_stride

        # Get frames until success
        while True:
            index = index % len(self.metadata)
            sample = self.metadata.iloc[index]
            video_path = self._get_video_path(sample)

            instruction = sample['instruction']
            if self.cond_robot_label_prob > 0.0 and random.random(
            ) < self.cond_robot_label_prob:
                if sample['embodiment'] != 'x':
                    instruction = sample['embodiment'] + ' [SEP] ' + sample[
                        'instruction']
            try:
                if self.load_raw_resolution:
                    video_reader = VideoReader(video_path, ctx=cpu(0))
                else:
                    video_reader = VideoReader(video_path,
                                               ctx=cpu(0),
                                               width=530,
                                               height=300)
                if len(video_reader) < self.video_length:
                    print(
                        f">>> Video length ({len(video_reader)}) is smaller than target length({self.video_length})"
                    )
                    index += 1
                    continue
                else:
                    pass
            except:
                index += 1
                print(f">>> Error: load video failed! path = {video_path}")
                continue

            fps_ori = video_reader.get_avg_fps()
            if self.fixed_fps is not None:
                frame_stride = int(frame_stride *
                                   (1.0 * fps_ori / self.fixed_fps))

            # To avoid extreme cases when fixed_fps is used
            frame_stride = max(frame_stride, 1)

            # Get valid range (adapting case by case)
            required_frame_num = frame_stride * (self.video_length - 1) + 1
            frame_num = len(video_reader)
            if frame_num < required_frame_num:
                # Drop extra samples if fixed fps is required
                if self.fixed_fps is not None and frame_num < required_frame_num * 0.5:
                    index += 1
                    continue
                else:
                    frame_stride = frame_num // self.video_length
                    required_frame_num = frame_stride * (self.video_length -
                                                         1) + 1

            # Select a random clip
            random_range = frame_num - required_frame_num
            start_idx = random.randint(
                0, random_range -
                frame_stride) if random_range - frame_stride > 0 else 0

            # Calculate frame indices
            frame_indices = [
                start_idx + frame_stride * i for i in range(self.video_length)
            ]
            try:
                next_frame_indices = [
                    idx + frame_stride for idx in frame_indices
                ]
                frames = video_reader.get_batch(next_frame_indices)
                break
            except:
                print(
                    f">>> Error: Get frames failed! path = {video_path}; [max_ind vs frame_total:{max(frame_indices)} / {frame_num}]"
                )
                index += 1
                continue

        # Load transition data
        transition_path = self._get_transition_path(sample)
        with h5py.File(transition_path, 'r') as h5f:
            transition_dict = {}
            for key in h5f.keys():
                transition_dict[key] = torch.tensor(h5f[key][()])
            for key in h5f.attrs.keys():
                transition_dict[key] = h5f.attrs[key]

        # Load observable states
        if start_idx < self.n_obs_steps - 1:
            state_indices = list(range(0, start_idx + 1))
            states = transition_dict['observation.state'][state_indices, :]
            num_padding = self.n_obs_steps - 1 - start_idx
            first_slice = states[0:1, :]  # (t, d)
            padding = first_slice.repeat(num_padding, 1)
            states = torch.cat((padding, states), dim=0)
        else:
            state_indices = list(
                range(start_idx - self.n_obs_steps + 1, start_idx + 1))
            states = transition_dict['observation.state'][state_indices, :]
        assert states.shape[
            0] == self.n_obs_steps, '>>> Do not have enough previous states as observation.'

        # Load observable actions
        if start_idx < self.n_obs_steps:
            pre_action_indices = list(range(0, start_idx))
            pre_actions = transition_dict['action'][pre_action_indices, :]
            num_padding = self.n_obs_steps - start_idx
            first_slice = torch.zeros_like(transition_dict['action'][:1, :])
            padding = first_slice.repeat(num_padding, 1)
            pre_actions = torch.cat((padding, pre_actions), dim=0)
        else:
            pre_action_indices = list(
                range(start_idx - self.n_obs_steps, start_idx))
            pre_actions = transition_dict['action'][pre_action_indices, :]
        assert pre_actions.shape[
            0] == self.n_obs_steps, ">>> Do not have enough previous actions as observation"

        # Load future actions
        actions = transition_dict['action'][frame_indices, :]
        # Load future states
        next_state_indices = [idx + frame_stride for idx in frame_indices]
        next_states = transition_dict['observation.state'][
            next_state_indices, :]
        frames_action_state_dict = {
            'pre_action': pre_actions,
            'action': actions,
            'observation.state': states,
            'next.state': next_states
        }
        if self.individual_normalization:
            frames_action_state_dict = self.normalizer(
                frames_action_state_dict)

        # Update action and states to unified vector
        frames_action_state_dict = self.get_uni_vec(
            frames_action_state_dict,
            transition_dict['action_type'],
            transition_dict['state_type'],
        )

        # Load observable images
        if start_idx < self.n_obs_steps - 1:
            action_net_frame_indices = list(range(0, start_idx + 1))
            action_net_frames = video_reader.get_batch(
                action_net_frame_indices)
            action_net_frames = torch.tensor(
                action_net_frames.asnumpy()).permute(0, 3, 1, 2).float()
            first_slice = action_net_frames[0:1, :]
            num_padding = self.n_obs_steps - 1 - start_idx
            padding = first_slice.repeat(num_padding, 1, 1, 1)
            action_net_frames = torch.cat((padding, action_net_frames), dim=0)
            assert (
                action_net_frames.shape[0] == self.n_obs_steps
            ), f'{len(action_net_frames)}, self.n_obs_steps={self.n_obs_steps}'
            action_net_frames = action_net_frames.permute(1, 0, 2, 3)
        else:
            action_net_frame_indices = list(
                range(start_idx - self.n_obs_steps + 1, start_idx + 1))
            action_net_frames = video_reader.get_batch(
                action_net_frame_indices)
            assert (
                action_net_frames.shape[0] == self.n_obs_steps
            ), f'{len(action_net_frames)}, self.n_obs_steps={self.n_obs_steps}'
            action_net_frames = torch.tensor(
                action_net_frames.asnumpy()).permute(3, 0, 1, 2).float()

        assert (frames.shape[0] == self.video_length
                ), f'{len(frames)}, self.video_length={self.video_length}'
        frames = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float()

        if self.spatial_transform is not None:
            frames = self.spatial_transform(frames)
            action_net_frames = self.spatial_transform(action_net_frames)

        if self.resolution is not None:
            assert (frames.shape[2], frames.shape[3]) == (
                self.resolution[0], self.resolution[1]
            ), f'frames={frames.shape}, self.resolution={self.resolution}'
            assert (
                action_net_frames.shape[2], action_net_frames.shape[3]
            ) == (
                self.resolution[0], self.resolution[1]
            ), f'action_net_frames={action_net_frames.shape}, self.resolution={self.resolution}'

        # Normalize frames tensors to [-1,1]
        frames = (frames / 255 - 0.5) * 2
        action_net_frames = (action_net_frames / 255 - 0.5) * 2
        fps_clip = fps_ori // frame_stride
        if self.fps_max is not None and fps_clip > self.fps_max:
            fps_clip = self.fps_max

        data = {
            'video': frames,
            'instruction': instruction,
            'path': video_path,
            'fps': fps_clip,
            'frame_stride': frame_stride,
            'observation.image': action_net_frames,
        }
        data.update(frames_action_state_dict)

        return data

    def __len__(self):
        return len(self.metadata)
