from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from torch.utils.data import DataLoader, Subset
from gym_av_aloha.common.replay_buffer import ReplayBuffer
from torchvision.transforms import Resize
import torch
import os
from tqdm import tqdm

def main(args):
    repo_id = args.repo_id
    image_size = args.image_size

    zarr_path = os.path.join("outputs", repo_id)

    replay_buffer = ReplayBuffer.create_from_path(zarr_path=zarr_path, mode="a")

    ds_meta = LeRobotDatasetMetadata(repo_id)
    replay_buffer.update_meta({
        "repo_id": ds_meta.repo_id,
    })

    def convert(k, v: torch.Tensor, ds_meta: LeRobotDatasetMetadata):
        dtype = ds_meta.features[k]['dtype']
        if dtype in ['image', 'video']:
            v = Resize(image_size)(v)
            # (B, C, H, W) to (B, H, W, C)
            v = v.permute(0, 2, 3, 1)
            # convert from torch float32 to numpy uint8
            v = (v * 255).to(torch.uint8).numpy()
        else:
            v = v.numpy()
        return v
    
    # shallow copy
    features = ds_meta.features.copy()
    # remove any keys that start with "observation.images" 
    for key in list(features.keys()):
        valid_keys = [
            "observation.images.zed_cam_left",
            "observation.images.zed_cam_right",
            "observation.images.left_eye_cam",
            "observation.images.right_eye_cam",
        ]
        if key.startswith("observation.images") and key not in valid_keys:
            del features[key]
            print(f"Removed {key} from features because it is not an AV image.")

    dataset = LeRobotDataset(repo_id)
    # iterate through dataset
    for i in range(replay_buffer.n_episodes, ds_meta.total_episodes):
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

        batch = {k:convert(k,v,ds_meta) for k,v in batch.items() if k in features}
        replay_buffer.add_episode(batch, compressors='disk')
        print(f"Episode {i} converted and added to replay buffer.")

    episode_lengths = replay_buffer.episode_lengths
    # print number of episodes
    print(f"Total number of episodes: {len(episode_lengths)}")

    print(f"Converted dataset saved to {zarr_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert LeRobot dataset to Zarr format.")
    parser.add_argument("--repo_id", type=str, default="iantc104/av_aloha_sim_peg_insertion", help="Repository ID for the dataset.")
    parser.add_argument("--image_size", type=int, nargs=2, default=(240, 320), help="Size to resize images to (height, width).")
    parser.add_argument("--av_images_only", action='store_true', help="Only convert AV images.")

    """"
    python convert_lerobot_to_zarr.py --repo_id iantc104/av_aloha_sim_thread_needle --av_images_only --image_size 240 320"
    """

    args = parser.parse_args()
    main(args)