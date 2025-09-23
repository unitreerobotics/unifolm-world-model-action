import time

import numpy as np
import pinocchio as pin

from unitree_deploy.robot.robot_configs import g1_motors
from unitree_deploy.robot_devices.arm.configs import G1ArmConfig
from unitree_deploy.robot_devices.arm.utils import make_arm_motors_buses_from_configs
from unitree_deploy.robot_devices.robots_devices_utils import precise_wait

if __name__ == "__main__":
    # ============== Arm Configuration ==============
    def g1_dual_arm_default_factory():
        return {
            "g1": G1ArmConfig(
                init_pose=np.zeros(14),
                motors=g1_motors,
                mock=False,
            ),
        }

    # ==============================================

    # Initialize and connect to the robotic arm
    arm = make_arm_motors_buses_from_configs(g1_dual_arm_default_factory())
    for name in arm:
        arm[name].connect()
    time.sleep(1.5)
    print("✅ Arm connected. Waiting to start...")

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

                # Solve inverse kinematics for the arm
                for name in arm:
                    sol_q, sol_tauff = arm[name].arm_ik(L_tf_target.homogeneous, R_tf_target.homogeneous)
                    print(f"Arm {name} solution: q={sol_q}, tauff={sol_tauff}")
                    # Determine command mode
                    cmd_target = "drive_to_waypoint" if initial_data_received else "schedule_waypoint"

                    # Send joint target command to arm
                    arm[name].write_arm(
                        q_target=sol_q,
                        tauff_target=sol_tauff,  # Optional: send torque feedforward
                        time_target=t_command_target - time.monotonic() + time.perf_counter(),
                        cmd_target="schedule_waypoint",
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
