import collections
import time
from typing import List, Optional

import cv2
import dm_env
import numpy as np
import torch

from unitree_deploy.robot.robot_utils import make_robot
from unitree_deploy.robot_devices.robots_devices_utils import precise_wait
from unitree_deploy.utils.rich_logger import log_success


class UnitreeEnv:
    def __init__(
        self,
        robot_type: str = "z1_realsense",
        dt: float = 1 / 30,
        init_pose_arm: np.ndarray | List[float] | None = None,
    ):
        self.control_dt = dt
        self.init_pose_arm = init_pose_arm
        self.state: Optional[np.ndarray] = None
        self.robot_type = robot_type
        self.robot = make_robot(self.robot_type)

    def connect(self):
        self.robot.connect()

    def _get_obs(self):
        observation = self.robot.capture_observation()

        # Process images
        image_dict = {
            key.split("observation.images.")[-1]: cv2.cvtColor(value.numpy(), cv2.COLOR_BGR2RGB)
            for key, value in observation.items()
            if key.startswith("observation.images.")
        }
        # for image_key, image in image_dict.items():
        #     cv2.imwrite(f"{image_key}.png", image)

        # Process state
        self.state = observation["observation.state"].numpy()

        # Construct observation dictionary
        obs = collections.OrderedDict(
            qpos=self.state,
            qvel=np.zeros_like(self.state),
            effort=np.zeros_like(self.state),
            images=image_dict,
        )

        return obs

    def get_observation(self, t=0):
        step_type = dm_env.StepType.FIRST if t == 0 else dm_env.StepType.MID
        return dm_env.TimeStep(step_type=step_type, reward=0, discount=None, observation=self._get_obs())

    def step(self, action) -> dm_env.TimeStep:
        t_cycle_end = time.monotonic() + self.control_dt
        t_command_target = t_cycle_end + self.control_dt
        self.robot.send_action(torch.from_numpy(action), t_command_target)
        precise_wait(t_cycle_end)

        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=0,
            discount=None,
            observation=self._get_obs(),
        )

    def close(self) -> None:
        self.robot.disconnect()
        log_success("Robot disconnected successfully! 🎉")


def make_real_env(
    robot_type: str, dt: float | None, init_pose_arm: np.ndarray | List[float] | None = None
) -> UnitreeEnv:
    return UnitreeEnv(robot_type, dt, init_pose_arm)
