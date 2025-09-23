import time

import tyro

from unitree_deploy.robot_devices.endeffector.utils import (
    Dex1_GripperConfig,
    make_endeffector_motors_buses_from_configs,
)
from unitree_deploy.robot_devices.robots_devices_utils import precise_wait
from unitree_deploy.utils.rich_logger import log_success
from unitree_deploy.utils.trajectory_generator import sinusoidal_single_gripper_motion

period = 2.0
motion_period = 2.0
motion_amplitude = 0.99


def gripper_default_factory():
    return {
        "left": Dex1_GripperConfig(
            unit_test=True,
            motors={
                "kLeftGripper": [0, "z1_gripper-joint"],
            },
            topic_gripper_state="rt/dex1/left/state",
            topic_gripper_command="rt/dex1/left/cmd",
        ),
        "right": Dex1_GripperConfig(
            unit_test=True,
            motors={
                "kRightGripper": [1, "z1_gripper-joint"],
            },
            topic_gripper_state="rt/dex1/right/state",
            topic_gripper_command="rt/dex1/right/cmd",
        ),
    }


def run_gripper():
    control_dt = 1 / 30

    log_success("Running gripper in style 1 (multi-bus from config)")
    endeffectors = make_endeffector_motors_buses_from_configs(gripper_default_factory())

    for name in endeffectors:
        endeffectors[name].connect()
        log_success(f"Connected endeffector '{name}'.")

    while True:
        t_cycle_end = time.monotonic() + control_dt
        target_q = sinusoidal_single_gripper_motion(
            period=motion_period, amplitude=motion_amplitude, current_time=time.perf_counter()
        )
        for name in endeffectors:
            endeffectors[name].write_endeffector(q_target=target_q)
        precise_wait(t_cycle_end)


if __name__ == "__main__":
    tyro.cli(run_gripper)
