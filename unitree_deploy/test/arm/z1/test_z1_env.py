import math
import time

import numpy as np
import pinocchio as pin

from unitree_deploy.real_unitree_env import make_real_env
from unitree_deploy.utils.rerun_visualizer import RerunLogger, flatten_images, visualization_data
from unitree_deploy.utils.rich_logger import log_info
from unitree_deploy.utils.trajectory_generator import generate_rotation, sinusoidal_gripper_motion

if __name__ == "__main__":
    rerun_logger = RerunLogger()
    env = make_real_env(robot_type="z1_realsense", dt=1 / 30)
    env.connect()

    # Define initial target poses for left and right arms
    arm_tf_target = pin.SE3(pin.Quaternion(1, 0, 0, 0), np.array([0.2, 0, 0.4]))

    # Motion parameters
    rotation_speed = 0.01  # Rotation speed (rad per step)
    control_dt = 1 / 30  # Control cycle duration (20ms)
    step = 0
    max_step = 240  # Full motion cycle

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

                # Generate target rotation and translation
                L_quat, R_quat, delta_l, delta_r = generate_rotation(step, rotation_speed, max_step)
                arm_tf_target.translation += delta_l
                # delta_r is not used in this context
                arm_tf_target.rotation = L_quat.toRotationMatrix()

                # Solve inverse kinematics for the left arm
                for arm_name in env.robot.arm:
                    arm_sol_q, arm_sol_tauff = env.robot.arm[arm_name].arm_ik(arm_tf_target.homogeneous)

                # Generate sinusoidal motion for the gripper
                target_gripper = (
                    sinusoidal_gripper_motion(period=4.0, amplitude=0.99, current_time=current_time) - 1
                )  # Adjust target_q by subtracting 1

                target_arm = np.concatenate((arm_sol_q, target_gripper), axis=0)  # Add a zero for the gripper
                step_type, reward, _, observation = env.step(target_arm)

                idx += 1
                visualization_data(idx, flatten_images(observation), observation["qpos"], target_arm, rerun_logger)

                # Update step and reset after full cycle
                current_time += control_dt
                step = (step + 1) % (max_step + 1)

        except KeyboardInterrupt:
            # Handle Ctrl+C to safely disconnect
            log_info("\n🛑 Ctrl+C detected. Disconnecting arm...")
            env.close()
