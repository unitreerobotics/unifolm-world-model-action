# Getting Started

### Code framework


| Module Name              | Documentation Link                                 |
| ------------------------- | -------------------------------------------------- |
| robots                    | [build_robot](./build_robot.md)                    |
| robot_devices/arm         | [add_robot_arm](./add_robot_arm.md)                |
| robot_devices/cameras     | [add_robot_camera](./add_robot_camera.md)          |
| robot_devices/endeffector | [add_robot_endeffector](./add_robot_endeffector.md)|

### Simple Usage (Example code, not executable)

```python
import time
import math
import torch

from unitree_deploy.robot.robot_utils import make_robot
from unitree_deploy.robot_devices.robots_devices_utils import precise_wait

class YourPolicy:
    def predict_action(self, observation, policy):
        # Logic for predicting action
        pass

class UnitreeEnv:
    def __init__(self):
        self.robot = make_robot(self.robot_type)
        if not self.robot.is_connected:
            self.robot.connect()
            # If disconnection is needed, call disconnect() here
            # self.robot.disconnect()

    def get_obs(self):
        # Get observation
        observation = self.robot.capture_observation()
        return observation

    def step(self, pred_action, t_command_target):
        # Execute action
        t_cycle_end = time.monotonic() + self.control_dt
        t_command_target = t_cycle_end + self.control_dt
        action = self.robot.send_action(torch.from_numpy(pred_action), t_command_target)
        precise_wait(t_cycle_end)
        return action

if __name__ == "__main__":
    policy = YourPolicy()  # Create policy instance
    env = UnitreeEnv()     # Create environment instance

    t_start = time.monotonic()   # Get start time
    iter_idx = 0                 # Initialize iteration index
    control_dt = 1 / 30          # Control loop interval (30Hz)

    try:
        while True:
            t_cycle_end = t_start + (iter_idx + 1) * control_dt   # Calculate end time of current cycle
            t_command_target = t_cycle_end + control_dt           # Calculate command target time

            observation = env.get_obs()                          # Get environment observation
            pred_action = policy.predict_action(observation, policy)  # Predict action
            env.step(pred_action, t_command_target)               # Execute action

            precise_wait(t_cycle_end)                             # Wait until cycle end
            iter_idx += 1                                         # Update iteration index
    finally:
        # Perform cleanup operations on exit (e.g., disconnect robot)
        pass
