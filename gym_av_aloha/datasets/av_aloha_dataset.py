import datasets
import torch
from os import path
import numpy as np
from typing import Callable

import gym_av_aloha
from gym_av_aloha.common.replay_buffer import ReplayBuffer
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import (
    check_delta_timestamps,
    get_delta_indices,
    get_episode_data_index,
    check_timestamps_sync,
    get_hf_features_from_features,
)

ROOT = path.dirname(path.dirname(gym_av_aloha.__file__))
print(f"Loading AVAlohaImageDataset zarr from {ROOT}")

def get_ds_meta_from_zarr(zarr_path: str = None, repo_id: str = None, root: str = ROOT) -> LeRobotDatasetMetadata:
    if zarr_path is None:
        assert repo_id is not None, "Either `repo_id` or `zarr_path` must be provided."
        zarr_path = path.join(root, "outputs", repo_id)
    replay_buffer = ReplayBuffer.copy_from_path(zarr_path)
    meta_repo_id = str(np.array(replay_buffer.meta['repo_id']))
    return LeRobotDatasetMetadata(meta_repo_id)


class AVAlohaImageDataset(torch.utils.data.Dataset):
    def __init__(self,
                 repo_id: str | None = None,
                 root: str | None = ROOT,
                 zarr_path: str | None = None,
                 episodes: list[int] | None = None,
                 image_transforms: Callable | None = None,
                 delta_timestamps: dict[list[float]] | None = None,
                 tolerance_s: float = 1e-4,
                 ):
        super().__init__()
        if zarr_path is None:
            assert repo_id is not None, "Either `repo_id` or `zarr_path` must be provided."
            zarr_path = path.join(root, "outputs", repo_id)
        self.zarr_path = zarr_path
        self.delta_timestamps = delta_timestamps
        self.tolerance_s = tolerance_s
        self.episodes = episodes

        self.replay_buffer = ReplayBuffer.copy_from_path(self.zarr_path)
        meta_repo_id = str(np.array(self.replay_buffer.meta['repo_id']))
        if meta_repo_id != repo_id: print("[AVAlohaImageDataset] Warning: repo_id mismatch, using metadata from zarr.")
        self.meta = LeRobotDatasetMetadata(meta_repo_id)

        self.image_transforms = image_transforms

        self.episode_data_index = get_episode_data_index({
            i: {
                'episode_index': i,
                'length': length
            }
            for i, length in enumerate(self.replay_buffer.episode_lengths)
        }, self.episodes)

        # Check timestamps
        timestamps = np.array(self.replay_buffer['timestamp'])
        episode_indices = np.array(self.replay_buffer['episode_index'])
        # keep only timestamps and episode_indices for the selected episodes
        if self.episodes is not None:
            mask = np.isin(episode_indices, self.episodes)
            timestamps = timestamps[mask]
            episode_indices = episode_indices[mask]
        ep_data_index_np = {k: t.numpy() for k, t in self.episode_data_index.items()}
        check_timestamps_sync(timestamps, episode_indices, ep_data_index_np, self.fps, self.tolerance_s)

        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

    def _get_query_indices(self, idx: int, ep_idx: int) -> tuple[dict[str, list[int | bool]]]:
        ep_start = self.episode_data_index["from"][ep_idx]
        ep_end = self.episode_data_index["to"][ep_idx]
        query_indices = {
            key: [max(ep_start.item(), min(ep_end.item() - 1, idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {  # Pad values outside of current episode range
            f"{key}_is_pad": torch.BoolTensor(
                [(idx + delta < ep_start.item()) | (idx + delta >= ep_end.item()) for delta in delta_idx]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _query_replay_buffer(self, query_indices: list[str, list[int]]) -> dict:
        return {
            key: self.replay_buffer[key][q_idx]
            for key, q_idx in query_indices.items()
        }

    @property
    def stats(self):
        return self.meta.stats

    @property
    def features(self):
        return self.meta.features

    @property
    def hf_features(self) -> datasets.Features:
        """Features of the hf_dataset."""
        return get_hf_features_from_features(self.features)

    @property
    def fps(self):
        return self.meta.fps

    @property
    def num_frames(self) -> int:
        """Number of frames in selected episodes."""
        return self.meta.total_frames

    @property
    def num_episodes(self):
        return len(self.episodes) if self.episodes is not None else self.meta.total_episodes

    @property
    def video_keys(self):
        return self.meta.video_keys

    @property
    def image_keys(self):
        return self.meta.image_keys

    def __len__(self) -> int:
        return self.replay_buffer.n_steps

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep_idx = self.replay_buffer["episode_index"][idx]
        item = {"episode_index": torch.tensor(ep_idx)}

        query_indices, padding = self._get_query_indices(idx, ep_idx)
        query_result = self._query_replay_buffer(query_indices)
        item = {**item, **padding}
        for key, val in query_result.items():
            if key in self.image_keys or key in self.video_keys:
                item[key] = torch.from_numpy(val).type(torch.float32).permute(0, 3, 1, 2) / 255.0
            else:
                item[key] = torch.from_numpy(val)

        if self.image_transforms is not None:
            image_keys = self.meta.camera_keys
            for cam in image_keys:
                item[cam] = self.image_transforms(item[cam])

        return item
