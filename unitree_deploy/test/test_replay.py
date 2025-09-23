"""
python test/test_replay.py --repo-id unitreerobotics/G1_CameraPackaging_NewDataset --robot_type g1_dex1
python test/test_replay.py --repo-id unitreerobotics/Z1_StackBox_Dataset --robot_type z1_realsense
python test/test_replay.py --repo-id unitreerobotics/Z1_Dual_Dex1_StackBox_Dataset_V2 --robot_type z1_dual_dex1_realsense
"""

import tyro
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from unitree_deploy.real_unitree_env import make_real_env
from unitree_deploy.utils.rerun_visualizer import RerunLogger, flatten_images, visualization_data
from unitree_deploy.utils.rich_logger import log_info


# Replay a specific episode from the LeRobot dataset using the real environment  robot_type：(e.g., g1_dex1, z1_realsense, z1_dual_dex1_realsense)
def replay_lerobot_data(repo_id: str, robot_type: str, root: str | None = None, episode: int = 145):
    dataset = LeRobotDataset(repo_id, root=root, episodes=[episode])
    actions = dataset.hf_dataset.select_columns("action")
    init_pose_arm = actions[0]["action"].numpy()[:14] if robot_type == "g1" else actions[0]["action"].numpy()
    rerun_logger = RerunLogger()

    env = make_real_env(robot_type=robot_type, dt=1 / 30, init_pose_arm=init_pose_arm)
    env.connect()

    try:
        # Wait for user input to start the motion loop
        user_input = input("Please enter the start signal (enter 's' to start the subsequent program): \n")
        if user_input.lower() == "s":
            log_info("Replaying episode")
            for idx in range(dataset.num_frames):
                action = actions[idx]["action"].numpy()
                if robot_type == "z1_realsense":
                    action[-1] = -action[-1]
                step_type, reward, _, observation = env.step(action)
                visualization_data(idx, flatten_images(observation), observation["qpos"], action, rerun_logger)
            env.close()
    except KeyboardInterrupt:
        # Handle Ctrl+C to safely disconnect
        log_info("\n🛑 Ctrl+C detected. Disconnecting arm...")
        env.close()


if __name__ == "__main__":
    tyro.cli(replay_lerobot_data)
