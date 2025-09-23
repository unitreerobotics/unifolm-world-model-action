import time

import torch

from unitree_deploy.robot.robot_configs import UnitreeRobotConfig
from unitree_deploy.robot_devices.arm.utils import make_arm_motors_buses_from_configs
from unitree_deploy.robot_devices.cameras.utils import make_cameras_from_configs
from unitree_deploy.robot_devices.endeffector.utils import (
    make_endeffector_motors_buses_from_configs,
)
from unitree_deploy.utils.rich_logger import log_success


class UnitreeRobot:
    def __init__(
        self,
        config: UnitreeRobotConfig,
    ):
        self.config = config
        self.robot_type = self.config.type
        self.cameras = make_cameras_from_configs(self.config.cameras)
        self.arm = make_arm_motors_buses_from_configs(self.config.arm)
        self.endeffector = make_endeffector_motors_buses_from_configs(self.config.endeffector)

        self.initial_data_received = True

    def connect(self):
        if not self.arm and self.endeffector and not self.cameras:
            raise ValueError(
                "UnitreeRobot doesn't have any device to connect. See example of usage in docstring of the class."
            )
        # Connect the cameras
        for name in self.cameras:
            self.cameras[name].connect()
            log_success(f"Connecting {name} cameras.")

        for _ in range(20):
            for name in self.cameras:
                self.cameras[name].async_read()
            time.sleep(1 / 30)

        for name in self.arm:
            self.arm[name].connect()
            log_success(f"Connecting {name} arm.")

        for name in self.endeffector:
            self.endeffector[name].connect()
            log_success(f"Connecting {name} endeffector.")

        time.sleep(2)
        log_success("All Device Connect Success!!!.✅")

    def capture_observation(self):
        """The returned observations do not have a batch dimension."""

        # Create state by concatenating follower current position
        state = []
        arm_state_list = []
        endeffector_state_list = []
        for arm_name in self.arm:
            arm_state = self.arm[arm_name].read_current_arm_q()
            arm_state_list.append(torch.from_numpy(arm_state))

        for endeffector_name in self.endeffector:
            endeffector_state = self.endeffector[endeffector_name].read_current_endeffector_q()
            endeffector_state_list.append(torch.from_numpy(endeffector_state))

        state = (
            torch.cat(arm_state_list + endeffector_state_list, dim=0)
            if arm_state_list or endeffector_state_list
            else torch.tensor([])
        )

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            output = self.cameras[name].async_read()
            if isinstance(output, dict):
                images.update({k: torch.from_numpy(v) for k, v in output.items()})
            else:
                images[name] = torch.from_numpy(output)

        # Populate output dictionnaries and format to pytorch
        obs_dict = {}
        obs_dict["observation.state"] = state
        for name, value in images.items():
            obs_dict[f"observation.images.{name}"] = value
        return obs_dict

    def send_action(self, action: torch.Tensor, t_command_target: float | None = None) -> torch.Tensor:
        from_idx_arm = 0
        to_idx_arm = 0
        action_sent_arm = []
        cmd_target = "drive_to_waypoint" if self.initial_data_received else "schedule_waypoint"

        for arm_name in self.arm:
            to_idx_arm += len(self.arm[arm_name].motor_names)
            action_arm = action[from_idx_arm:to_idx_arm].numpy()
            from_idx_arm = to_idx_arm

            action_sent_arm.append(torch.from_numpy(action_arm))

            self.arm[arm_name].write_arm(
                action_arm,
                time_target=t_command_target - time.monotonic() + time.perf_counter(),
                cmd_target=cmd_target,
            )

        from_idx_endeffector = to_idx_arm
        to_idx_endeffector = to_idx_arm

        action_endeffector_set = []
        for endeffector_name in self.endeffector:
            to_idx_endeffector += len(self.endeffector[endeffector_name].motor_names)
            action_endeffector = action[from_idx_endeffector:to_idx_endeffector].numpy()
            from_idx_endeffector = to_idx_endeffector

            action_endeffector_set.append(torch.from_numpy(action_endeffector))

            self.endeffector[endeffector_name].write_endeffector(
                action_endeffector,
                time_target=t_command_target - time.monotonic() + time.perf_counter(),
                cmd_target=cmd_target,
            )

        self.initial_data_received = False

        return torch.cat(action_sent_arm + action_endeffector_set, dim=0)

    def disconnect(self):
        # disconnect the arms
        for name in self.arm:
            self.arm[name].disconnect()
            log_success(f"disconnect {name} arm.")

        for name in self.endeffector:
            self.endeffector[name].disconnect()
            log_success(f"disconnect {name} endeffector.")

        # disconnect the cameras
        for name in self.cameras:
            self.cameras[name].disconnect()
            log_success(f"disconnect {name} cameras.")

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
