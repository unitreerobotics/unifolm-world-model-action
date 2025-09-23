import os
import sys
import threading
import time
from typing import Callable

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))
import unitree_arm_interface as unitree_z1  # type: ignore

from unitree_deploy.robot_devices.arm.arm_indexs import Z1GripperArmJointIndex
from unitree_deploy.robot_devices.arm.configs import Z1ArmConfig
from unitree_deploy.robot_devices.arm.z1_arm_ik import Z1_Arm_IK
from unitree_deploy.robot_devices.robots_devices_utils import (
    DataBuffer,
    MotorState,
    Robot_Num_Motors,
    RobotDeviceAlreadyConnectedError,
)
from unitree_deploy.utils.joint_trajcetory_inter import JointTrajectoryInterpolator
from unitree_deploy.utils.rich_logger import RichLogger


class Z1LowState:
    def __init__(self) -> None:
        self.motor_state: list[MotorState] = [MotorState() for _ in range(Robot_Num_Motors.Z1_7_Num_Motors)]


class Z1ArmController:
    def __init__(self, config: Z1ArmConfig):
        self.motors = config.motors

        self.init_pose = config.init_pose
        self.unit_test = config.unit_test
        self.control_dt = config.control_dt

        self.robot_kp = config.robot_kp
        self.robot_kd = config.robot_kd
        self.max_pos_speed = config.max_pos_speed
        self.log_level = config.log_level

        self.q_target = self.init_pose
        self.dq_target = np.zeros(len(Z1GripperArmJointIndex) - 1, dtype=np.float16)
        self.ddq_target = np.zeros(len(Z1GripperArmJointIndex) - 1, dtype=np.float16)
        self.tauff_target = np.zeros(len(Z1GripperArmJointIndex) - 1, dtype=np.float16)
        self.ftip_target = np.zeros(len(Z1GripperArmJointIndex) - 1, dtype=np.float16)
        self.time_target = time.monotonic()

        self.DELTA_GRIPPER_CMD = 5.0 / 20.0 / 25.6
        self.arm_cmd = "schedule_waypoint"

        self.ctrl_lock = threading.Lock()

        self.lowstate_buffer = DataBuffer()
        self.z1_arm_ik = Z1_Arm_IK(unit_test=self.unit_test, visualization=False)
        self.logger = RichLogger(self.log_level)

        self.is_connected = False
        self.grasped = False

    @property
    def motor_names(self) -> list[str]:
        return list(self.motors.keys())

    @property
    def motor_models(self) -> list[str]:
        return [model for _, model in self.motors.values()]

    @property
    def motor_indices(self) -> list[int]:
        return [idx for idx, _ in self.motors.values()]

    def _start_daemon_thread(self, target_fn: Callable[[], None], name: str | None = None) -> threading.Thread:
        thread = threading.Thread(target=target_fn, name=name)
        thread.daemon = True
        thread.start()
        return thread

    def connect(self):
        try:
            if self.is_connected:
                raise RobotDeviceAlreadyConnectedError(
                    "Z1_Arm is already connected. Do not run `robot.connect()` twice."
                )
            # Initialize arms
            self.z1 = unitree_z1.ArmInterface()
            self.z1_model = self.z1._ctrlComp.armModel
            self.z1.setFsmLowcmd()
            self.z1.lowcmd.setControlGain(self.robot_kp, self.robot_kd)
            self.z1.sendRecv()

            self.subscribe_thread = self._start_daemon_thread(
                self._subscribe_motor_state, name="z1._subscribe_motor_state"
            )
            while not self.lowstate_buffer.get_data():
                time.sleep(0.01)
                self.logger.warning("[Z1_ArmController] Waiting Get Data...")

            self.publish_thread = self._start_daemon_thread(self._ctrl_motor_state, name="z1._ctrl_motor_state")
            self.is_connected = True

        except Exception as e:
            self.disconnect()
            self.logger.error(f"❌ Error in Z1ArmController.connect: {e}")

    def _subscribe_motor_state(self):
        try:
            while True:
                lowstate = Z1LowState()
                for motor_id in range(Robot_Num_Motors.Z1_7_Num_Motors - 1):
                    lowstate.motor_state[motor_id].q = self.z1.lowstate.getQ()[motor_id]
                    lowstate.motor_state[motor_id].dq = self.z1.lowstate.getQd()[motor_id]

                gripper_q = self.z1.lowstate.getGripperQ()
                lowstate.motor_state[Robot_Num_Motors.Z1_7_Num_Motors - 1].q = gripper_q
                lowstate.motor_state[Robot_Num_Motors.Z1_7_Num_Motors - 1].dq = 0.0

                self.lowstate_buffer.set_data(lowstate)
                time.sleep(self.control_dt)

        except Exception as e:
            self.disconnect()
            self.logger.error(f"❌ Error in Z1ArmController._subscribe_motor_state: {e}")

    def _update_z1_arm(
        self,
        q: np.ndarray,
        qd: np.ndarray | None = None,
        qdd: np.ndarray | None = None,
        ftip: np.ndarray | None = None,
        tau: np.ndarray | None = None,
    ):
        """Update the state and command of a given robotic arm."""
        current_gripper_q = self.read_current_gripper_q()
        self.z1.q = q[: len(Z1GripperArmJointIndex) - 1]
        self.z1.qd = self.dq_target if qd is None else qd
        qdd = self.ddq_target if qdd is None else qdd
        ftip = self.ftip_target if ftip is None else ftip
        self.z1.tau = self.z1_model.inverseDynamics(self.z1.q, self.z1.qd, qdd, ftip) if tau is None else tau
        self.z1.setArmCmd(self.z1.q, self.z1.qd, self.z1.tau)

        gripper_q = q[len(Z1GripperArmJointIndex) - 1]
        self.z1.gripperQ = np.clip(
            gripper_q,
            current_gripper_q - self.DELTA_GRIPPER_CMD * 3,
            current_gripper_q + self.DELTA_GRIPPER_CMD * 3,
        )
        # self.z1.gripperQ = np.clip(gripper_q, current_gripper_q - self.DELTA_GRIPPER_CMD, current_gripper_q + self.DELTA_GRIPPER_CMD) if self.grasped else np.clip(gripper_q, current_gripper_q - self.DELTA_GRIPPER_CMD*4, current_gripper_q + self.DELTA_GRIPPER_CMD*4) # np.clip(gripper_q, current_gripper_q - self.DELTA_GRIPPER_CMD*3, current_gripper_q + self.DELTA_GRIPPER_CMD*3)
        self.z1.setGripperCmd(self.z1.gripperQ, self.z1.gripperQd, self.z1.gripperTau)
        self.z1.sendRecv()
        time.sleep(self.control_dt)
        self.grasped = abs(self.read_current_gripper_q() - current_gripper_q) < self.DELTA_GRIPPER_CMD / 12.0

    def _drive_to_waypoint(self, target_pose: np.ndarray, t_insert_time: float):
        curr_time = time.monotonic() + self.control_dt
        t_insert = curr_time + t_insert_time
        self.pose_interp = self.pose_interp.drive_to_waypoint(
            pose=target_pose,
            time=t_insert,
            curr_time=curr_time,
            max_pos_speed=self.max_pos_speed,
        )

        while time.monotonic() < t_insert:
            self._update_z1_arm(self.pose_interp(time.monotonic()))

    def _schedule_waypoint(
        self,
        arm_q_target: np.ndarray,
        arm_time_target: float,
        t_now: float,
        start_time: float,
        last_waypoint_time: float,
        arm_tauff_target: np.ndarray | None = None,
    ) -> float:
        target_time = time.monotonic() - time.perf_counter() + arm_time_target
        curr_time = t_now + self.control_dt
        target_time = max(target_time, curr_time + self.control_dt)

        self.pose_interp = self.pose_interp.schedule_waypoint(
            pose=arm_q_target,
            time=target_time,
            max_pos_speed=self.max_pos_speed,
            curr_time=curr_time,
            last_waypoint_time=last_waypoint_time,
        )
        last_waypoint_time = target_time
        self._update_z1_arm(q=self.pose_interp(t_now), tau=arm_tauff_target)

    def _ctrl_motor_state(self):
        try:
            self.pose_interp = JointTrajectoryInterpolator(
                times=[time.monotonic()],
                joint_positions=[self.read_current_arm_q()],
            )
            self.go_start()

            arm_q_target = self.read_current_arm_q()
            arm_tauff_target = self.tauff_target
            arm_time_target = time.monotonic()
            arm_cmd = "schedule_waypoint"
            last_waypoint_time = time.monotonic()

            while True:
                start_time = time.perf_counter()
                t_now = time.monotonic()
                with self.ctrl_lock:
                    arm_q_target = self.q_target
                    arm_tauff_target = self.tauff_target
                    arm_time_target = self.time_target
                    arm_cmd = self.arm_cmd

                if arm_cmd == "drive_to_waypoint":
                    self._drive_to_waypoint(target_pose=arm_q_target, t_insert_time=0.8)

                elif arm_cmd == "schedule_waypoint":
                    self._schedule_waypoint(
                        arm_q_target=arm_q_target,
                        arm_time_target=arm_time_target,
                        t_now=t_now,
                        start_time=start_time,
                        last_waypoint_time=last_waypoint_time,
                        arm_tauff_target=arm_tauff_target,
                    )

        except Exception as e:
            self.disconnect()
            self.logger.error(f"❌ Error in Z1ArmController._ctrl_motor_state: {e}")

    def read_current_arm_q(self) -> np.ndarray:
        """Return current state q of the left and right arm motors."""
        return np.array([self.lowstate_buffer.get_data().motor_state[id].q for id in Z1GripperArmJointIndex])

    def read_current_arm_q_without_gripper(self) -> np.ndarray:
        """Return current state q of the left and right arm motors."""
        return np.array([self.lowstate_buffer.get_data().motor_state[id].q for id in list(Z1GripperArmJointIndex)[:-1]])

    def read_current_gripper_q(self) -> np.ndarray:
        """Return current state q of the left and right arm motors."""
        return np.array([self.lowstate_buffer.get_data().motor_state[list(Z1GripperArmJointIndex)[-1].value].q])

    def read_current_arm_dq(self) -> np.ndarray:
        """Return current state dq of the left and right arm motors."""
        return np.array([self.lowstate_buffer.get_data().motor_state[id].dq for id in Z1GripperArmJointIndex])

    def read_current_arm_dq_without_gripper(self) -> np.ndarray:
        """Return current state dq of the left and right arm motors."""
        return np.array(
            [self.lowstate_buffer.get_data().motor_state[id].dq for id in list(Z1GripperArmJointIndex)[:-1]]
        )

    def write_arm(
        self,
        q_target: list[float] | np.ndarray,
        tauff_target: list[float] | np.ndarray = None,
        time_target: float | None = None,
        cmd_target: str | None = None,
    ):
        """Set control target values q & tau of the left and right arm motors."""
        with self.ctrl_lock:
            self.q_target = q_target
            self.tauff_target = tauff_target
            self.time_target = time_target
            self.arm_cmd = cmd_target

    def arm_ik(self, ee_target: list[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        return self.z1_arm_ik.solve_ik(
            ee_target, self.read_current_arm_q_without_gripper(), self.read_current_arm_dq_without_gripper()
        )

    def arm_fk(self, q: np.ndarray | None = None) -> np.ndarray | None:
        return self.z1_model.forwardKinematics(
            q if q is not None else self.read_current_arm_q(), len(Z1GripperArmJointIndex)
        )

    def go_start(self):
        self._drive_to_waypoint(target_pose=self.init_pose, t_insert_time=1.0)
        self.logger.success("Go Start OK!\n")

    def go_home(self):
        self.z1.loopOn()
        self.z1.backToStart()
        self.z1.loopOff()
        time.sleep(0.5)
        self.logger.success("Go Home OK!\n")

    def disconnect(self):
        self.is_connected = False
        self.go_home()

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
