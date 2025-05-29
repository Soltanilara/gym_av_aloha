import datasets
import torch
from pathlib import Path
import os
import numpy as np
from typing import Callable
import json
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
from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Resize
from tqdm import tqdm

ROOT = Path(os.path.dirname(os.path.dirname(gym_av_aloha.__file__))) / "outputs"

def create_av_aloha_dataset_from_lerobot(
    repo_id: str | None = None,
    root: str | Path | None = None,
    new_repo_id: str | None = None,
    new_root: str | Path | None = None,
    image_size: tuple[int, int] | None = None,
    remove_keys: list[str] = [],
):
    root = Path(root) if root else ROOT / repo_id
    new_root = Path(new_root) if new_root else ROOT / new_repo_id

    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
    )
    replay_buffer = ReplayBuffer.create_from_path(zarr_path=new_root, mode="a")

    # metadata
    config = {
        "repo_id": dataset.repo_id,
    }
    replay_buffer.update_meta(config)
    with open(new_root / "config.json", 'w') as f:
        json.dump(config, f, indent=4)
    
    # shallow copy
    features = dataset.meta.features.copy()
    # remove any keys in remove_keys
    features = {k: v for k, v in features.items() if k not in remove_keys}

    def convert(k, v: torch.Tensor):
        dtype = features[k]['dtype']
        if dtype in ['image', 'video']:
            if image_size is not None:
                v = Resize(image_size)(v)
            # (B, C, H, W) to (B, H, W, C)
            v = v.permute(0, 2, 3, 1)
            # convert from torch float32 to numpy uint8
            v = (v * 255).to(torch.uint8).numpy()
        else:
            v = v.numpy()
        return v
    
    # iterate through dataset
    for i in range(replay_buffer.n_episodes, dataset.meta.total_episodes):
        print(f"Converting episode {i}...")
        from_idx = dataset.episode_data_index['from'][i]
        to_idx = dataset.episode_data_index['to'][i]
        subset = Subset(dataset, range(from_idx, to_idx))
        dataloader = DataLoader(subset, batch_size=16, shuffle=False, num_workers=8)

        data = []
        for batch in tqdm(dataloader):
            if "task" in batch:
                del batch["task"]
            data.append(batch)
        # since batch is a dict go through keys and cat them into a batch
        batch = {k: torch.cat([d[k] for d in data], dim=0) for k in data[0].keys()}

        assert batch['action'].shape[0] == to_idx - from_idx, f"Batch size does not match episode length. Expected {to_idx - from_idx}, got {batch['action'].shape[0]}."

        batch = {k:convert(k,v) for k,v in batch.items() if k in features}
        replay_buffer.add_episode(batch, compressors='disk')
        print(f"Episode {i} converted and added to replay buffer.")

    print(f"Converted dataset saved to {new_root}.")

def get_ds_meta(
    repo_id: str | None = None,
    root: str | Path | None = None,
) -> LeRobotDatasetMetadata:
    root = Path(root) if root else ROOT / repo_id
    config_path = root / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config: dict = json.load(f)
        meta_repo_id = config.get('repo_id', repo_id)
    else:
        # legacy
        replay_buffer = ReplayBuffer.copy_from_path(str(root))
        meta_repo_id = str(np.array(replay_buffer.meta['repo_id']))
    return LeRobotDatasetMetadata(meta_repo_id)

class AVAlohaDataset(torch.utils.data.Dataset):
    def __init__(self,
                 repo_id: str | None = None,
                 root: str | Path | None = None,
                 episodes: list[int] | None = None,
                 image_transforms: Callable | None = None,
                 delta_timestamps: dict[list[float]] | None = None,
                 tolerance_s: float = 1e-4,
                 ):
        super().__init__()

        self.repo_id = repo_id
        self.root = Path(root) if root else ROOT / repo_id
        self.episodes = episodes
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.tolerance_s = tolerance_s
        self.episodes = episodes

        # create zarr dataset + lerobot metadata
        self.replay_buffer = ReplayBuffer.copy_from_path(self.root)
        self.meta = get_ds_meta(
            repo_id=self.repo_id,
            root=self.root
        )

        # if no episodes are specified, use all episodes in the replay buffer
        if self.episodes is None: 
            self.episodes = list(range(self.meta.total_episodes))

        # calculate length of the dataset
        self.length = sum([self.replay_buffer.episode_lengths[i] for i in self.episodes])

        # add task index to delta timestamps
        if 'task_index' in self.meta.features:
            self.delta_timestamps['task_index'] = [0]  

        # from and to indices for episodes
        self.replay_buffer_data_index = get_episode_data_index(self.meta.episodes)
        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)

        # create valid indices for the dataset
        self.valid_indices = np.concatenate([
            np.arange(self.replay_buffer_data_index["from"][ep], self.replay_buffer_data_index["to"][ep])
            for ep in self.episodes
        ])

        # Check timestamps
        timestamps = np.array(self.replay_buffer['timestamp'])
        episode_indices = np.array(self.replay_buffer['episode_index'])
        ep_data_index_np = {k: t.numpy() for k, t in self.replay_buffer_data_index.items()}
        check_timestamps_sync(timestamps, episode_indices, ep_data_index_np, self.fps, self.tolerance_s)

        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

    def _get_query_indices(self, idx: int, ep_idx: int) -> tuple[dict[str, list[int | bool]]]:
        ep_start = self.replay_buffer_data_index["from"][ep_idx]
        ep_end = self.replay_buffer_data_index["to"][ep_idx]
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
        return self.length

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
        return self.num_frames

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        global_idx = self.valid_indices[idx]
        ep_idx = self.replay_buffer["episode_index"][global_idx]
        item = {"episode_index": torch.tensor(ep_idx)}

        query_indices, padding = self._get_query_indices(global_idx, ep_idx)
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

        # Add task as a string
        if "task_index" in item:
            task_idx = item["task_index"].item()
            item["task"] = self.meta.tasks[task_idx]

        return item
