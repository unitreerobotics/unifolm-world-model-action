import time

import numpy as np
import pinocchio as pin

from unitree_deploy.robot_devices.arm.configs import Z1DualArmConfig
from unitree_deploy.robot_devices.arm.utils import make_arm_motors_buses_from_configs
from unitree_deploy.robot_devices.robots_devices_utils import precise_wait
from unitree_deploy.utils.trajectory_generator import generate_rotation

if __name__ == "__main__":
    # ============== Arm Configuration ==============
    def z1_dual_arm_single_config_factory():
        return {
            "z1_dual": Z1DualArmConfig(
                left_robot_ip="127.0.0.1",
                left_robot_port1=8073,
                left_robot_port2=8074,
                right_robot_ip="127.0.0.1",
                right_robot_port1=8071,
                right_robot_port2=8072,
                init_pose_left=[0, 0, 0, 0, 0, 0],
                init_pose_right=[0, 0, 0, 0, 0, 0],
                control_dt=1 / 250.0,
                motors={
                    # name: (index, model)
                    "kLeftWaist": [0, "z1-joint"],
                    "kLeftShoulder": [1, "z1-joint"],
                    "kLeftElbow": [2, "z1-joint"],
                    "kLeftForearmRoll": [3, "z1-joint"],
                    "kLeftWristAngle": [4, "z1-joint"],
                    "kLeftWristRotate": [5, "z1-joint"],
                    "kRightWaist": [7, "z1-joint"],
                    "kRightShoulder": [8, "z1-joint"],
                    "kRightElbow": [9, "z1-joint"],
                    "kRightForearmRoll": [10, "z1-joint"],
                    "kRightWristAngle": [11, "z1-joint"],
                    "kRightWristRotate": [12, "z1-joint"],
                },
            ),
        }

    # ==============================================

    # Initialize and connect to the robotic arm
    arm = make_arm_motors_buses_from_configs(z1_dual_arm_single_config_factory())
    for name in arm:
        arm[name].connect()
    time.sleep(1.5)

    print("✅ Arm connected. Waiting to start...")

    # Define initial target poses for left and right arms
    L_tf_target = pin.SE3(pin.Quaternion(1, 0, 0, 0), np.array([0.2, 0, 0.4]))
    R_tf_target = pin.SE3(pin.Quaternion(1, 0, 0, 0), np.array([0.2, 0, 0.3]))

    # Motion parameters
    rotation_speed = 0.01  # Rotation speed (rad per step)
    control_dt = 1 / 30  # Control cycle duration (20ms)
    step = 0
    max_step = 240  # Full motion cycle
    initial_data_received = True  # Used to switch from drive to schedule mode

    # Wait for user input to start the motion loop
    user_input = input("Please enter the start signal (enter 's' to start the subsequent program): \n")
    if user_input.lower() == "s":
        try:
            while True:
                # Define timing for the control cycle
                t_cycle_end = time.monotonic() + control_dt
                t_command_target = t_cycle_end + control_dt

                # Generate target rotation and translation
                L_quat, R_quat, delta_l, delta_r = generate_rotation(step, rotation_speed, max_step)

                # Apply translation deltas to target pose
                L_tf_target.translation += delta_l
                R_tf_target.translation += delta_r

                # Apply rotation to target pose
                L_tf_target.rotation = L_quat.toRotationMatrix()
                R_tf_target.rotation = R_quat.toRotationMatrix()

                # Solve inverse kinematics for the left arm
                for name in arm:
                    sol_q, sol_tauff = arm[name].arm_ik(L_tf_target.homogeneous, R_tf_target.homogeneous)

                # Determine command mode
                cmd_target = "drive_to_waypoint" if initial_data_received else "schedule_waypoint"

                # Send joint target command to arm
                for name in arm:
                    arm[name].write_arm(
                        q_target=sol_q,
                        # tauff_target=sol_tauff,   # Optional: send torque feedforward
                        time_target=t_command_target - time.monotonic() + time.perf_counter(),
                        cmd_target=cmd_target,
                    )

                # Update step and reset after full cycle
                step = (step + 1) % (max_step + 1)
                initial_data_received = False

                # Wait until end of control cycle
                precise_wait(t_cycle_end)

        except KeyboardInterrupt:
            # Handle Ctrl+C to safely disconnect
            print("\n🛑 Ctrl+C detected. Disconnecting arm...")
            for name in arm:
                arm[name].disconnect()
            print("✅ Arm disconnected. Exiting.")
