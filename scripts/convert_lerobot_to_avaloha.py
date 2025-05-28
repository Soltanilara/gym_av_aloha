from gym_av_aloha.datasets.av_aloha_dataset import create_av_aloha_dataset_from_lerobot

def main(args):
    repo_id = args.repo_id
    image_size = args.image_size
    rename = args.rename
    av_images_only = args.av_images_only

    if not rename:
        rename = repo_id

    remove_keys = []
    if av_images_only:
        remove_keys = [
            "observation.images.wrist_cam_left",
            "observation.images.wrist_cam_right",
            "observation.images.worms_eye_cam",
            "observation.images.overhead_cam",
        ]

    create_av_aloha_dataset_from_lerobot(
        repo_id=repo_id,
        new_repo_id=rename,
        image_size=image_size,
        remove_keys=remove_keys,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert LeRobot dataset to Zarr format.")
    parser.add_argument("--repo_id", type=str, default="iantc104/av_aloha_sim_peg_insertion", help="Repository ID for the dataset.")
    parser.add_argument("--image_size", type=int, nargs=2, default=(240, 320), help="Size to resize images to (height, width).")
    parser.add_argument("--rename", type=str, default=None, help="Rename the dataset to this name.")
    parser.add_argument("--av_images_only", action='store_true', help="Only convert AV images.")

    """"
    python scripts/convert_lerobot_to_zarr.py --repo_id iantc104/av_aloha_sim_thread_needle --av_images_only --image_size 240 320 --rename iantc104/av_aloha_sim_thread_needle_240x320
    python scripts/convert_lerobot_to_zarr.py --repo_id lerobot/pusht_keypoints
    python scripts/convert_lerobot_to_zarr.py --repo_id lerobot/pusht --image_size 96 96
    python scripts/convert_lerobot_to_zarr.py --repo_id iantc104/av_aloha_sim_peg_insertion_v0 --av_images_only --image_size 240 320 --rename iantc104/av_aloha_sim_peg_insertion_240x320
    """

    args = parser.parse_args()
    main(args)