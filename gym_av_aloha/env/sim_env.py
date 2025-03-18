import numpy as np
import mujoco.viewer
from dm_control import mjcf
import gymnasium as gym
from gymnasium import spaces
from gym_av_aloha.env.sim_config import (
    XML_DIR, CAMERAS, RENDER_CAMERA,
    STATE_DIM, ACTION_DIM,
    SIM_DT, SIM_PHYSICS_DT, SIM_PHYSICS_ENV_STEP_RATIO,
    LEFT_JOINT_NAMES, LEFT_GRIPPER_JOINT_NAMES, 
    RIGHT_JOINT_NAMES, RIGHT_GRIPPER_JOINT_NAMES, 
    MIDDLE_JOINT_NAMES,
    LEFT_ACTUATOR_NAMES, LEFT_GRIPPER_ACTUATOR_NAME,
    RIGHT_ACTUATOR_NAMES, RIGHT_GRIPPER_ACTUATOR_NAME,
    MIDDLE_ACTUATOR_NAMES,
    LIGHT_NAME,
)
from gym_av_aloha.env.robot import SimRobotArm
import os

class AVAlohaEnv(gym.Env):

    metadata = {"render_modes": ["rgb_array"], "render_fps": 1/SIM_DT}
    XML = os.path.join(XML_DIR, 'scene.xml')
    LEFT_POSE = [0, -0.082, 1.06, 0, -0.953, 0]
    LEFT_GRIPPER_POSE = 1
    RIGHT_POSE = [0, -0.082, 1.06, 0, -0.953, 0]
    RIGHT_GRIPPER_POSE = 1
    MIDDLE_POSE = [0, -0.6, 0.5, 0, 0.5, 0, 0]

    def __init__(
        self, 
        fps = 25, 
        cameras=CAMERAS,
        render_camera=RENDER_CAMERA,
    ):
        
        super().__init__()
        self.fps = fps
        self.cameras = cameras
        self.render_camera = render_camera

        self.max_reward = 0
        self.viewer = None 
        self.metadata["render_fps"] = self.fps

        self.step_dt = 1/self.fps
        self.n_ctrl_steps = round(self.step_dt/SIM_DT)
        if self.n_ctrl_steps < 1:
            raise ValueError("FPS too high for simulation")
        
        # print(f"Sim step dt: {self.step_dt}, Sim ctrl dt: {SIM_DT}, Sim physics dt: {SIM_PHYSICS_DT}")
        # print(f"n ctrl steps / sim step: {self.n_ctrl_steps}, n physics steps / ctrl step: {SIM_PHYSICS_ENV_STEP_RATIO}")

        self.mjcf_root = mjcf.from_path(self.XML)
        self.mjcf_root.option.timestep = SIM_PHYSICS_DT

        self.physics = mjcf.Physics.from_mjcf_model(self.mjcf_root)

        self.left_joints = [self.mjcf_root.find('joint', name) for name in LEFT_JOINT_NAMES]
        self.left_gripper_joints = [self.mjcf_root.find('joint', name) for name in LEFT_GRIPPER_JOINT_NAMES]
        self.right_joints = [self.mjcf_root.find('joint', name) for name in RIGHT_JOINT_NAMES]
        self.right_gripper_joints = [self.mjcf_root.find('joint', name) for name in RIGHT_GRIPPER_JOINT_NAMES]
        self.middle_joints = [self.mjcf_root.find('joint', name) for name in MIDDLE_JOINT_NAMES]
        self.left_actuators = [self.mjcf_root.find('actuator', name) for name in LEFT_ACTUATOR_NAMES]
        self.left_gripper_actuator = self.mjcf_root.find('actuator', LEFT_GRIPPER_ACTUATOR_NAME)
        self.right_actuators = [self.mjcf_root.find('actuator', name) for name in RIGHT_ACTUATOR_NAMES]
        self.right_gripper_actuator = self.mjcf_root.find('actuator', RIGHT_GRIPPER_ACTUATOR_NAME)
        self.middle_actuators = [self.mjcf_root.find('actuator', name) for name in MIDDLE_ACTUATOR_NAMES]

        self.left_arm = SimRobotArm(
            physics=self.physics,
            joints=self.left_joints,
            actuators=self.left_actuators,
            has_gripper=True,
            gripper_joints=self.left_gripper_joints,
            gripper_actuator=self.left_gripper_actuator,
        )

        self.right_arm = SimRobotArm(
            physics=self.physics,
            joints=self.right_joints,
            actuators=self.right_actuators,
            has_gripper=True,
            gripper_joints=self.right_gripper_joints,
            gripper_actuator=self.right_gripper_actuator,
        )

        self.middle_arm = SimRobotArm(
            physics=self.physics,
            joints=self.middle_joints,
            actuators=self.middle_actuators,
            has_gripper=False,
        )

        observation_space_dict = {}
        observation_space_dict['agent_pos'] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(STATE_DIM,),
            dtype=np.float64,
        )
        if len(self.cameras) > 0:
            observation_space_dict['pixels'] = spaces.Dict(
                {
                    camera: spaces.Box(
                        low=0,
                        high=255,
                        shape=(*dim,3),
                        dtype=np.uint8,
                    )
                    for camera, dim in self.cameras.items()
                }
            )

        self.observation_space = spaces.Dict(observation_space_dict)
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(ACTION_DIM,), dtype=np.float64) 

    def get_obs(self) -> np.ndarray:
        obs = {}

        obs['agent_pos'] = np.concatenate([
            self.left_arm.get_joint_positions(),
            self.left_arm.get_gripper_position(),
            self.right_arm.get_joint_positions(),
            self.right_arm.get_gripper_position(),
            self.middle_arm.get_joint_positions(),
        ])

        if len(self.cameras) > 0:
            obs['pixels'] = {
                camera: self.physics.render(
                    height=dim[0],
                    width=dim[1],
                    camera_id=camera
                )
                for camera, dim in self.cameras.items()
            }

        return obs
    
    def get_qpos(self):
        return self.physics.data.qpos.copy()
    
    def set_qpos(self, qpos):
        self.physics.data.qpos[:] = qpos
        self.physics.forward()
    
    def get_ctrl(self):
        return np.concatenate([
            self.left_arm.get_joint_ctrl(),
            np.array([self.left_arm.get_gripper_ctrl()]),
            self.right_arm.get_joint_ctrl(),
            np.array([self.right_arm.get_gripper_ctrl()]),
            self.middle_arm.get_joint_ctrl(),
        ])
    
    def set_ctrl(self, ctrl):
        self.left_arm.set_joint_ctrl(ctrl[0:6])
        self.left_arm.set_gripper_ctrl(ctrl[6])
        self.right_arm.set_joint_ctrl(ctrl[7:13])
        self.right_arm.set_gripper_ctrl(ctrl[13])
        self.middle_arm.set_joint_ctrl(ctrl[14:21])
    
    def get_reward(self):
        return 0
    
    def render(self):
        return self.physics.render(
            height=225,
            width=300, 
            camera_id=RENDER_CAMERA
        )
        
    def step(self, action: np.ndarray) -> tuple:
        ctrl = action
        prev_ctrl = self.get_ctrl()
        actions = np.linspace(prev_ctrl, ctrl, self.n_ctrl_steps+1)[1:]
        for i in range(self.n_ctrl_steps):
            self.set_ctrl(actions[i])
            self.physics.step(nstep=SIM_PHYSICS_ENV_STEP_RATIO)
        
        observation = self.get_obs()
        reward = self.get_reward()
        terminated = False
        truncated = False
        info = {"is_success": reward == self.max_reward}

        return observation, reward, terminated, truncated, info 
    
    def randomize_light(self):
        x_range = [-0.3, 0.3]
        y_range = [-0.3, 0.3]
        z_range = [1.5, 1.5]
        ranges = np.vstack([x_range, y_range, z_range])
        self.physics.named.model.light_directional[LIGHT_NAME] = False
        self.physics.named.model.light_pos[LIGHT_NAME] = np.random.uniform(ranges[:, 0], ranges[:, 1])
        self.physics.named.model.light_ambient[LIGHT_NAME] = np.random.uniform(0, 1.0, size=3)
        self.physics.named.model.light_diffuse[LIGHT_NAME] = np.random.uniform(.1, 1.0, size=3)

    def reset_light(self):
        self.physics.named.model.light_directional[LIGHT_NAME] = True
        self.physics.named.model.light_pos[LIGHT_NAME] = np.array([0, 1, 1.5])
        self.physics.named.model.light_ambient[LIGHT_NAME] = np.array([0.0, 0.0, 0.0])
        self.physics.named.model.light_dir[LIGHT_NAME] = np.array([0, 0, -1])
        self.physics.named.model.light_diffuse[LIGHT_NAME] = np.array([0.7, 0.7, 0.7])

    def reset(self, seed=None, options:dict=None) -> tuple:
        super().reset(seed=seed, options=options)

        # reset physics
        self.physics.reset()

        # random light
        if options and options.get('randomize_light', False):
            self.randomize_light()
        else:
            self.reset_light()

        self.left_arm.set_joint_positions(self.LEFT_POSE)
        self.left_arm.set_gripper_position(self.LEFT_GRIPPER_POSE)
        self.right_arm.set_joint_positions(self.RIGHT_POSE)
        self.right_arm.set_gripper_position(self.RIGHT_GRIPPER_POSE)
        self.middle_arm.set_joint_positions(self.MIDDLE_POSE)
        self.left_arm.set_joint_ctrl(self.LEFT_POSE)
        self.left_arm.set_gripper_ctrl(self.LEFT_GRIPPER_POSE)
        self.right_arm.set_joint_ctrl(self.RIGHT_POSE)
        self.right_arm.set_gripper_ctrl(self.RIGHT_GRIPPER_POSE)
        self.middle_arm.set_joint_ctrl(self.MIDDLE_POSE)

        self.physics.forward()

        observation = self.get_obs()
        info = {"is_success": False}

        return observation, info   

    def render_viewer(self) -> np.ndarray:
        if self.viewer is None:
            # launch viewer
            self.viewer = mujoco.viewer.launch_passive(
                self.physics.model.ptr,
                self.physics.data.ptr,
                show_left_ui=True,
                show_right_ui=True,
            )

        # render
        self.viewer.sync()        

    def close(self) -> None:
        """
        Closes the viewer if it's open.
        """
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


def main():
    import gym_av_aloha
    import gymnasium as gym
    import time

    env = gym.make("gym_av_aloha/av-aloha-v0", cameras={}, fps=8.33)

    action = np.concatenate([
        env.unwrapped.LEFT_POSE,
        [env.unwrapped.LEFT_GRIPPER_POSE],
        env.unwrapped.RIGHT_POSE,
        [env.unwrapped.RIGHT_GRIPPER_POSE],
        env.unwrapped.MIDDLE_POSE
    ])

    observation, info = env.reset(seed=42, options = {"randomize_light": True})

    i= 0
    while True:
        step_start = time.time()

        # Take a step in the environment using the chosen action
        observation, reward, terminated, truncated, info = env.step(action)

        env.unwrapped.render_viewer()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = SIM_DT - (time.time() - step_start)
        time.sleep(max(0, time_until_next_step))

        if i % 10 == 0:
            env.reset(seed=42, options = {"randomize_light": True})
            i = 0

        i += 1
        

if __name__ == '__main__':
    main()

    

    
