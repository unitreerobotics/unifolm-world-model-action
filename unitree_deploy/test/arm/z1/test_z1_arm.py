import math
import time

import numpy as np
import pinocchio as pin

from unitree_deploy.robot.robot_configs import z1_motors
from unitree_deploy.robot_devices.arm.utils import make_arm_motors_bus
from unitree_deploy.robot_devices.robots_devices_utils import precise_wait
from unitree_deploy.utils.trajectory_generator import generate_rotation, sinusoidal_gripper_motion

if __name__ == "__main__":
    # ============== Arm Configuration ==============
    arm_type = "z1"
    arm_kwargs = {
        "arm_type": arm_type,
        "init_pose": [0.00623, 1.11164, -0.77531, -0.32167, -0.005, 0.0, 0.0],  # Initial joint pose
        "motors": z1_motors,
    }
    # ==============================================

    # Initialize and connect to the robotic arm
    arm = make_arm_motors_bus(**arm_kwargs)
    arm.connect()
    time.sleep(2)
    print("✅ Arm connected. Waiting to start...")

    # Define arm initial target poses
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
                arm_sol_q, arm_sol_tauff = arm.arm_ik(arm_tf_target.homogeneous)

                # Generate sinusoidal motion for the gripper
                target_gripper = (
                    sinusoidal_gripper_motion(period=4.0, amplitude=0.99, current_time=current_time) - 1
                )  # Adjust target_q by subtracting 1

                target_arm = np.concatenate((arm_sol_q, target_gripper), axis=0)  # Add a zero for the gripper

                arm.write_arm(
                    q_target=target_arm,
                    # tauff_target=left_sol_tauff,   # Optional: send torque feedforward
                    time_target=t_command_target - time.monotonic() + time.perf_counter(),
                    cmd_target="schedule_waypoint",
                )

                # Update step and reset after full cycle
                step = (step + 1) % (max_step + 1)
                current_time += control_dt

                # Wait until end of control cycle
                precise_wait(t_cycle_end)

        except KeyboardInterrupt:
            # Handle Ctrl+C to safely disconnect
            print("\n🛑 Ctrl+C detected. Disconnecting arm...")
            arm.disconnect()
            print("✅ Arm disconnected. Exiting.")
