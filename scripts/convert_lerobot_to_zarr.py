from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from torch.utils.data import DataLoader
from gym_av_aloha.common.replay_buffer import ReplayBuffer
from torchvision.transforms import Resize
import torch
import os
from tqdm import tqdm

def main(args):
    repo_id = args.repo_id
    image_size = args.image_size

    zarr_path = os.path.join("outputs", repo_id)

    # check that zarr_path exists
    if os.path.exists(zarr_path):
        print(f"Zarr path {zarr_path} already exists. Please delete it and try again.")
        return

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

    # iterate through dataset
    for i in tqdm(range(ds_meta.total_episodes)):
        dataset = LeRobotDataset(repo_id, episodes=[i])
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        batch = next(iter(dataloader))  # Fetches the batch
        batch = {k:convert(k,v,ds_meta) for k,v in batch.items() if k in ds_meta.features}
        replay_buffer.add_episode(batch, compressors='disk')

    print(f"Converted dataset saved to {zarr_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert LeRobot dataset to Zarr format.")
    parser.add_argument("--repo_id", type=str, default="iantc104/av_aloha_sim_peg_insertion", help="Repository ID for the dataset.")
    parser.add_argument("--image_size", type=int, nargs=2, default=(240, 320), help="Size to resize images to (height, width).")

    """"
    python convert_lerobot_to_zarr.py --repo_id iantc104/av_aloha_sim_peg_insertion --image_size 480 640"
    """

    args = parser.parse_args()
    main(args)