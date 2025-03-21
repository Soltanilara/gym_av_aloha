from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from gym_av_aloha.common.replay_buffer import ReplayBuffer
import torch
import numpy as np

from gym_av_aloha.common.replay_buffer import ReplayBuffer
from typing import Dict
from lerobot.common.datasets.utils import (
    check_delta_timestamps,
    get_delta_indices,
    get_episode_data_index,
    check_timestamps_sync,
)

class AVAlohaImageDataset(torch.utils.data.Dataset):
    def __init__(self,
            zarr_path: str | None = None,
            delta_timestamps: dict[list[float]] | None = None,
            tolerance_s: float = 1e-4,
            episodes: list[int] | None = None,
        ):
        super().__init__()
        self.zarr_path = zarr_path
        self.delta_timestamps = delta_timestamps
        self.tolerance_s = tolerance_s
        self.episodes = episodes
        
        self.replay_buffer = ReplayBuffer.copy_from_path(self.zarr_path)
        repo_id = str(np.array(self.replay_buffer.meta['repo_id']))
        self.ds_meta = LeRobotDatasetMetadata(repo_id)

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
    
    def _query_replay_buffer(self, query_indices: dict[str, list[int]]) -> dict:
        return {
            key: self.replay_buffer[key][q_idx]
            for key, q_idx in query_indices.items()
        }

    @property
    def stats(self):
        return self.ds_meta.stats
    
    @property
    def features(self):
        return self.ds_meta.features
    
    @property
    def fps(self):
        return self.ds_meta.fps
    
    @property
    def total_episodes(self):
        return len(self.episodes)
    
    @property
    def video_keys(self):
        return self.ds_meta.video_keys
    
    @property
    def image_keys(self):
        return self.ds_meta.image_keys

    def __len__(self) -> int:
        return self.replay_buffer.n_steps
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
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

        return item