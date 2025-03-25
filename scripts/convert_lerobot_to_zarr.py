from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from torch.utils.data import DataLoader, Subset
from gym_av_aloha.common.replay_buffer import ReplayBuffer
from torchvision.transforms import Resize
import torch
import os

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
        subset = Subset(dataset, range(dataset.episode_data_index['from'][i], dataset.episode_data_index['to'][i]))
        dataloader = DataLoader(subset, batch_size=len(subset), shuffle=False)
        batch = next(iter(dataloader))  # Fetches the batch
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
    python convert_lerobot_to_zarr.py --repo_id iantc104/av_aloha_sim_peg_insertion --av_images_only --image_size 240 320"
    """

    args = parser.parse_args()
    main(args)