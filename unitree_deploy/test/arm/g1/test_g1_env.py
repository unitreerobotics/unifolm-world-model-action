import math
import time

import numpy as np
import pinocchio as pin

from unitree_deploy.real_unitree_env import make_real_env
from unitree_deploy.utils.rerun_visualizer import RerunLogger, flatten_images, visualization_data
from unitree_deploy.utils.rich_logger import log_info
from unitree_deploy.utils.trajectory_generator import sinusoidal_gripper_motion

if __name__ == "__main__":
    period = 2.0
    motion_period = 2.0
    motion_amplitude = 0.99

    rerun_logger = RerunLogger()
    env = make_real_env(robot_type="g1_dex1", dt=1 / 30)
    env.connect()

    # Define initial target poses for left and right arms
    L_tf_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.25, +0.25, 0.1]),
    )

    R_tf_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.25, -0.25, 0.1]),
    )

    rotation_speed = 0.005  # Rotation speed in radians per iteration

    # Motion parameters
    control_dt = 1 / 50  # Control cycle duration (20ms)
    step = 0
    max_step = 240

    initial_data_received = True  # Used to switch from drive to schedule mode
    # Wait for user input to start the motion loop
    user_input = input("Please enter the start signal (enter 's' to start the subsequent program): \n")
    if user_input.lower() == "s":
        try:
            current_time = math.pi / 2
            idx = 0  # Initialize index for logging
            while True:
                # Define timing for the control cycle
                t_cycle_end = time.monotonic() + control_dt
                t_command_target = t_cycle_end + control_dt

                direction = 1 if step <= 120 else -1
                angle = rotation_speed * (step if step <= 120 else (240 - step))

                cos_half_angle = np.cos(angle / 2)
                sin_half_angle = np.sin(angle / 2)

                L_quat = pin.Quaternion(cos_half_angle, 0, sin_half_angle, 0)  # 绕 Y 轴旋转
                R_quat = pin.Quaternion(cos_half_angle, 0, 0, sin_half_angle)  # 绕 Z 轴旋转

                delta_l = np.array([0.001, 0.001, 0.001]) * direction
                delta_r = np.array([0.001, -0.001, 0.001]) * direction

                L_tf_target.translation += delta_l
                R_tf_target.translation += delta_r

                L_tf_target.rotation = L_quat.toRotationMatrix()
                R_tf_target.rotation = R_quat.toRotationMatrix()

                # Solve inverse kinematics for the left arm
                for arm_name in env.robot.arm:
                    arm_sol_q, arm_sol_tauff = env.robot.arm[arm_name].arm_ik(
                        L_tf_target.homogeneous, R_tf_target.homogeneous
                    )

                gripper_target_q = sinusoidal_gripper_motion(
                    period=motion_period, amplitude=motion_amplitude, current_time=time.perf_counter()
                )
                action = np.concatenate([arm_sol_q, gripper_target_q], axis=0)
                step_type, reward, _, observation = env.step(action)

                idx += 1
                visualization_data(idx, flatten_images(observation), observation["qpos"], arm_sol_q, rerun_logger)

                # Update step and reset after full cycle
                current_time += control_dt
                step = (step + 1) % (max_step + 1)

        except KeyboardInterrupt:
            # Handle Ctrl+C to safely disconnect
            log_info("\n🛑 Ctrl+C detected. Disconnecting arm...")
            env.close()
