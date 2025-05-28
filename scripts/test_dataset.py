import time
from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDatasetMetadata,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from gym_av_aloha.datasets.av_aloha_dataset import AVAlohaDataset
from gym_av_aloha.datasets.multi_av_aloha_dataset import MultiAVAlohaDataset
import imageio
import einops

repo_id = "iantc104/av_aloha_sim_peg_insertion"
meta_ds = LeRobotDatasetMetadata(repo_id)

delta_timestamps = {
    "observation.images.zed_cam_left": [0],
    # "observation.images.zed_cam_right": [t / meta_ds.fps for t in range(1 - 2, 1)],
    # "observation.state": [t / meta_ds.fps for t in range(1 - 2, 1)],
    # "action": [t / meta_ds.fps for t in range(16)],
}

# zarr_dataset = AVAlohaImageDataset(
#     repo_id="iantc104/av_aloha_sim_peg_insertion_240x320", 
#     delta_timestamps=delta_timestamps,
#     episodes=[1,48],
# )
# zarr_dataloader = DataLoader(zarr_dataset, batch_size=32, shuffle=False)

# print("Testing zarr dataset")
# avg_time = 0
# for i in range(10):
#     start_time = time.time()
#     zarr_batch = next(iter(zarr_dataloader))
#     end_time = time.time()
#     avg_time += (end_time - start_time)

# print(f"Zarr dataset average time per batch: {avg_time/10:.4f} seconds")

# images = []
# for batch in tqdm(zarr_dataloader):
#     images.extend((batch["observation.images.zed_cam_left"].squeeze(1).permute(0, 2, 3, 1).numpy() * 255.0).astype("uint8"))
# # save mp4
# imageio.mimwrite(
#     "zarr_dataset_test.mp4",
#     images,
#     fps=meta_ds.fps,
# )


zarr_dataset = MultiAVAlohaDataset(
    repo_ids=["iantc104/av_aloha_sim_peg_insertion_240x320", "iantc104/av_aloha_sim_thread_needle_240x320"],
    delta_timestamps=delta_timestamps,
    episodes={
        "iantc104/av_aloha_sim_peg_insertion_240x320": [49],
        "iantc104/av_aloha_sim_thread_needle_240x320": [56, 199],
    }
)
zarr_dataloader = DataLoader(zarr_dataset, batch_size=32, shuffle=False)
images = []
for batch in tqdm(zarr_dataloader):
    images.extend((batch["observation.images.zed_cam_left"].squeeze(1).permute(0, 2, 3, 1).numpy() * 255.0).astype("uint8"))
# save mp4
imageio.mimwrite(
    "zarr_dataset_test.mp4",
    images,
    fps=meta_ds.fps,
)