import argparse
from gym_av_aloha.datasets.av_aloha_dataset import create_av_aloha_dataset_from_lerobot


def main():
    parser = argparse.ArgumentParser(description="Convert LeRobot dataset to AV-Aloha format.")
    parser.add_argument(
        "--repo_id",
        type=str,
        default="iantc104/av_aloha_sim_thread_needle",
        help="Hugging Face repo ID containing the dataset episodes",
    )
    parser.add_argument(
        "--start_episode",
        type=int,
        default=0,
        help="Start index of episodes to include",
    )
    parser.add_argument(
        "--end_episode",
        type=int,
        default=100,
        help="End index (exclusive) of episodes to include",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=(240, 320),
        help="Tuple for image size (height, width)",
    )
    parser.add_argument(
        "--skip_episodes",
        type=int,
        nargs="+",
        default=[30],
        help="Episodes to skip",
    )

    parser.add_argument(
        "--remove_keys",
        type=str,
        nargs="+",
        default=[
            "observation.images.wrist_cam_left",
            "observation.images.wrist_cam_right",
            "observation.images.worms_eye_cam",
            "observation.images.overhead_cam",
            "observation.environment_state",
        ],
        help="Keys to remove from the dataset",
    )

    args = parser.parse_args()


    episodes_to_include = [ep for ep in range(args.start_episode, args.end_episode) if ep not in args.skip_episodes]
    print(episodes_to_include)
    episodes = {
        args.repo_id: episodes_to_include,
    }
    print(episodes)
    create_av_aloha_dataset_from_lerobot(
        episodes=episodes,
        repo_id=args.repo_id,
        remove_keys=args.remove_keys,
        root_o="/home/jinyu/GitHub/dairy/temp_download_coin",
        image_size=tuple(args.image_size),
    )


if __name__ == "__main__":
    main()
