from gym_av_aloha.env.sim_env import AVAlohaEnv
from gym_av_aloha.env.sim_config import XML_DIR
import numpy as np
import os
from gymnasium import spaces


class PourTestTubeEnv(AVAlohaEnv):
    XML = os.path.join(XML_DIR, 'task_pour_test_tube.xml')
    LEFT_POSE = [0, -0.082, 1.06, 0, -0.953, 0]
    LEFT_GRIPPER_POSE = 1
    RIGHT_POSE = [0, -0.082, 1.06, 0, -0.953, 0]
    RIGHT_GRIPPER_POSE = 1
    MIDDLE_POSE = [0, -0.6, 0.5, 0, 0.5, 0, 0]
    ENV_STATE_DIM = 21
    PROMPTS = [
        "pour ball into test tube"
    ]

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.max_reward = 3

        self.ball_joint = self.mjcf_root.find('joint', 'ball_joint')
        self.tube1_joint = self.mjcf_root.find('joint', 'tube1_joint')
        self.tube2_joint = self.mjcf_root.find('joint', 'tube2_joint')

        self.distractor1_geom = self.mjcf_root.find('geom', 'distractor1')
        self.distractor2_geom = self.mjcf_root.find('geom', 'distractor2')
        self.distractor3_geom = self.mjcf_root.find('geom', 'distractor3')
        self.adverse_geom = self.mjcf_root.find('geom', 'adverse')
        self.distractor1_joint = self.mjcf_root.find('joint', 'distractor1_joint')
        self.distractor2_joint = self.mjcf_root.find('joint', 'distractor2_joint')
        self.distractor3_joint = self.mjcf_root.find('joint', 'distractor3_joint')
        self.adverse_joint = self.mjcf_root.find('joint', 'adverse_joint')

        self.observation_space_dict['environment_state'] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.ENV_STATE_DIM,),
            dtype=np.float64,
        )
        self.observation_space = spaces.Dict(self.observation_space_dict)

    def get_obs(self) -> np.ndarray:
        obs = super().get_obs()
        obs['environment_state'] = np.concatenate([
            self.physics.bind(self.ball_joint).qpos,
            self.physics.bind(self.tube1_joint).qpos,
            self.physics.bind(self.tube2_joint).qpos,
        ])
        return obs
    
    def set_state(self, state, environment_state):
        super().set_state(state, environment_state)
        self.physics.bind(self.ball_joint).qpos = environment_state[:7]
        self.physics.bind(self.tube1_joint).qpos = environment_state[7:14]
        self.physics.bind(self.tube2_joint).qpos = environment_state[14:21]
        self.physics.forward()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        # reset physics
        x_range = [0.05, 0.1]
        y_range = [-0.05, 0.05]
        z_range = [0.0, 0.0]
        ranges = np.vstack([x_range, y_range, z_range])
        ball_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
        ball_quat = np.array([1, 0, 0, 0])
        tube1_position = ball_position
        tube1_quat = ball_quat

        x_range = [-.1, -0.05]
        y_range = [-0.05, 0.05]
        z_range = [0.0, 0.0]
        ranges = np.vstack([x_range, y_range, z_range])
        tube2_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
        tube2_quat = np.array([1, 0, 0, 0])

        self.physics.bind(self.ball_joint).qpos = np.concatenate([ball_position, ball_quat])
        self.physics.bind(self.tube1_joint).qpos = np.concatenate([tube1_position, tube1_quat])
        self.physics.bind(self.tube2_joint).qpos = np.concatenate([tube2_position, tube2_quat])

        # reset distractors
        distractor_geoms = [self.distractor1_geom, self.distractor2_geom, self.distractor3_geom, self.adverse_geom]
        distractor_joints = [self.distractor1_joint, self.distractor2_joint, self.distractor3_joint, self.adverse_joint]
        position = np.array([0.0, 0.0, -1.0])
        quat = np.array([1, 0, 0, 0])
        for geom, joint in zip(distractor_geoms, distractor_joints):
            self.physics.bind(geom).contype = 0
            self.physics.bind(geom).conaffinity = 0
            self.physics.bind(joint).damping = 1e8
            self.physics.bind(joint).qpos = np.concatenate([position, quat])

        if (options and options.get('distractors', False)):
            distractor_geoms = [self.distractor1_geom, self.distractor2_geom, self.distractor3_geom]
            distractor_joints = [self.distractor1_joint, self.distractor2_joint, self.distractor3_joint]
            for geom, joint in zip(distractor_geoms, distractor_joints):
                self.physics.bind(geom).contype = 1
                self.physics.bind(geom).conaffinity = 1
                self.physics.bind(joint).damping = 0

            # find random positions that are not too close to each other or cube
            distractor_positions = []
            min_distance = 0.08  # Minimum distance to maintain

            x_range = [-.15, 0.15]
            y_range = [-0.1, 0.1]
            z_range = [0.0, 0.0]
            ranges = np.vstack([x_range, y_range, z_range])

            max_tries = 50
            while len(distractor_positions) < len(distractor_geoms):
                for i in range(max_tries):
                    d_pos = np.random.uniform(ranges[:, 0], ranges[:, 1])
                    if np.linalg.norm(d_pos - tube1_position) > min_distance and  \
                        np.linalg.norm(d_pos - tube2_position) > min_distance and \
                        all(np.linalg.norm(d_pos - dp) > min_distance for dp in distractor_positions):
                        distractor_positions.append(d_pos)
                        break
                else:
                    distractor_positions = []

            random_quats = []
            for _ in range(3):
                yaw = np.random.uniform(0, 2 * np.pi)
                quat = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
                random_quats.append(quat)

            # Assign positions to distractors
            for i, joint in enumerate(distractor_joints):
                self.physics.bind(joint).qpos = np.concatenate([distractor_positions[i], random_quats[i]])

        if (options and options.get('adverse', False)):
            self.physics.bind(self.adverse_geom).contype = 1
            self.physics.bind(self.adverse_geom).conaffinity = 1
            self.physics.bind(self.adverse_joint).damping = 0

            x_range = [-.15, 0.15]
            y_range = [-0.1, 0.1]
            z_range = [0.0, 0.0]
            ranges = np.vstack([x_range, y_range, z_range])

            min_distance = 0.08  # Minimum distance to maintain
            while True:
                d_pos = np.random.uniform(ranges[:, 0], ranges[:, 1])
                if np.linalg.norm(d_pos - tube1_position) > min_distance and  \
                        np.linalg.norm(d_pos - tube2_position) > min_distance:
                    adverse_position = d_pos
                    yaw = np.random.uniform(0, 2 * np.pi)
                    adverse_quat = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
                    self.physics.bind(self.adverse_joint).qpos = np.concatenate([adverse_position, adverse_quat])
                    break

        self.physics.forward()

        observation = self.get_obs()
        info = {"is_success": False}

        return observation, info

    def get_reward(self):

        touch_left_gripper = False
        touch_right_gripper = False
        tube1_touch_table = False
        tube2_touch_table = False
        pin_touched = False

        # return whether peg touches the pin
        contact_pairs = []
        for i_contact in range(self.physics.data.ncon):
            id_geom_1 = self.physics.data.contact[i_contact].geom1
            id_geom_2 = self.physics.data.contact[i_contact].geom2
            geom1 = self.physics.model.id2name(id_geom_1, 'geom')
            geom2 = self.physics.model.id2name(id_geom_2, 'geom')
            contact_pairs.append((geom1, geom2))
            contact_pairs.append((geom2, geom1))

        for geom1, geom2 in contact_pairs:
            if geom1.startswith("tube1-") and geom2.startswith("right"):
                touch_right_gripper = True

            if geom1.startswith("tube2-") and geom2.startswith("left"):
                touch_left_gripper = True

            if geom1 == "table" and geom2.startswith("tube1-"):
                tube1_touch_table = True

            if geom1 == "table" and geom2.startswith("tube2-"):
                tube2_touch_table = True

            if geom1 == "ball" and geom2 == "pin":
                pin_touched = True

        reward = 0
        if touch_left_gripper and touch_right_gripper:  # touch both
            reward = 1
        if touch_left_gripper and touch_right_gripper and (not tube1_touch_table) and (not tube2_touch_table):  # grasp both
            reward = 2
        if pin_touched:
            reward = 3
        return reward


def main():
    import gym_av_aloha
    from gym_av_aloha.env.sim_env import SIM_DT
    import gymnasium as gym
    import time

    env = gym.make("gym_av_aloha/pour-test-tube-v1", cameras={}, fps=8.33)

    action = np.concatenate([
        env.unwrapped.LEFT_POSE,
        [env.unwrapped.LEFT_GRIPPER_POSE],
        env.unwrapped.RIGHT_POSE,
        [env.unwrapped.RIGHT_GRIPPER_POSE],
        env.unwrapped.MIDDLE_POSE
    ])

    options_list = [
        {"randomize_light": True},
        {"distractors": True},
        {"adverse": True},
        {}
    ]

    observation, info = env.reset(seed=42, options=options_list[0])

    i = 0
    j = 0
    while True:
        step_start = time.time()

        # Take a step in the environment using the chosen action
        observation, reward, terminated, truncated, info = env.step(action)

        env.unwrapped.render_viewer()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = SIM_DT - (time.time() - step_start)
        time.sleep(max(0, time_until_next_step))

        if i % 10 == 0:
            env.reset(seed=42, options=options_list[j % len(options_list)])
            j += 1

        i += 1


if __name__ == '__main__':
    main()
