import gym_av_aloha
import gymnasium as gym
from gym_av_aloha.env.sim_env import AVAlohaEnv
from gym_av_aloha.kinematics.diff_ik import DiffIK, DiffIKConfig
from gym_av_aloha.kinematics.grad_ik import GradIK, GradIKConfig
import numpy as np
import time

FPS = 25
CAMERAS = {
    "zed_cam_left": [480, 640],
    "zed_cam_right": [480, 640],
}

class TeleopEnv():
    def __init__(self, env_name, fps=FPS, cameras=CAMERAS):
        gym_env = gym.make(f"gym_av_aloha/{env_name}", fps=fps, cameras=cameras)
        self.env: AVAlohaEnv = gym_env.unwrapped

        self.left_controller = GradIK(
            config=GradIKConfig(),
            physics=self.env.physics,
            joints=self.env.left_joints,
            eef_site=self.env.left_eef_site,
        )
        self.right_controller = GradIK(
            config=GradIKConfig(),
            physics=self.env.physics,
            joints=self.env.right_joints,
            eef_site=self.env.right_eef_site,
        )
        self.middle_controller = DiffIK(
            config=DiffIKConfig(),
            physics=self.env.physics,
            joints=self.env.middle_joints,
            eef_site=self.env.middle_eef_site,
        )

    def step(
        self,
        left_pos,
        left_rot,
        left_gripper,
        right_pos,
        right_rot,
        right_gripper,
        middle_pos,
        middle_rot,
    ):
        left_joints = self.left_controller.run(
            q=self.env.left_arm.get_joint_positions(),
            target_pos=left_pos,
            target_mat=left_rot,
        )
        right_joints = self.right_controller.run(
            q=self.env.right_arm.get_joint_positions(),
            target_pos=right_pos,
            target_mat=right_rot,
        )
        middle_joints = self.middle_controller.run(
            q=self.env.middle_arm.get_joint_positions(),
            target_pos=middle_pos,
            target_mat=middle_rot,
        )
        action = np.zeros(21)
        action[:6] = left_joints
        action[6] = left_gripper
        action[7:13] = right_joints
        action[13] = right_gripper
        action[14:21] = middle_joints

        # Apply the action to the environment
        observation, reward, terminated, truncated, info = self.env.step(action)

        info['action'] = action
        info['left_pos'] = self.env.left_arm.get_eef_position()
        info['left_rot'] = self.env.left_arm.get_eef_rotation()
        info['right_pos'] = self.env.right_arm.get_eef_position()
        info['right_rot'] = self.env.right_arm.get_eef_rotation()
        info['middle_pos'] = self.env.middle_arm.get_eef_position()
        info['middle_rot'] = self.env.middle_arm.get_eef_rotation()

        return observation, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        info['left_pos'] = self.env.left_arm.get_eef_position()
        info['left_rot'] = self.env.left_arm.get_eef_rotation()
        info['right_pos'] = self.env.right_arm.get_eef_position()
        info['right_rot'] = self.env.right_arm.get_eef_rotation()
        info['middle_pos'] = self.env.middle_arm.get_eef_position()
        info['middle_rot'] = self.env.middle_arm.get_eef_rotation()
        return observation, info
    
    def render_viewer(self):
        self.env.render_viewer()

if __name__ == "__main__":
    env = TeleopEnv(
        env_name="peg-insertion-v1",
        cameras={},
    )

    obs, info = env.reset()

    left_pos = info['left_pos']
    left_rot = info['left_rot']
    left_gripper = 1.0
    right_pos = info['right_pos']
    right_rot = info['right_rot']
    right_gripper = 1.0
    middle_pos = info['middle_pos']
    middle_rot = info['middle_rot']

    while True:
        start_time = time.time()
        observation, reward, terminated, truncated, info = env.step(
            left_pos=left_pos,
            left_rot=left_rot,
            left_gripper=left_gripper,
            right_pos=right_pos,
            right_rot=right_rot,
            right_gripper=right_gripper,
            middle_pos=middle_pos,
            middle_rot=middle_rot,
        )
        action = info['action']
        env.render_viewer()
        end_time = time.time()
        print(f"Step time: {end_time - start_time:.4f} seconds")
        time.sleep(max(0, 1.0 / FPS - (end_time - start_time)))

        left_pos += np.random.uniform(-0.01, 0.01, size=3)



