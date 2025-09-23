import os
import sys
import threading
import time
from typing import Callable

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))
import unitree_arm_interface as unitree_z1  # type: ignore

from unitree_deploy.robot_devices.arm.arm_indexs import Z1_12_JointArmIndex
from unitree_deploy.robot_devices.arm.configs import Z1DualArmConfig
from unitree_deploy.robot_devices.arm.z1_arm_ik import Z1_Arm_IK
from unitree_deploy.robot_devices.robots_devices_utils import (
    DataBuffer,
    MotorState,
    Robot_Num_Motors,
    RobotDeviceAlreadyConnectedError,
)
from unitree_deploy.utils.joint_trajcetory_inter import JointTrajectoryInterpolator
from unitree_deploy.utils.rich_logger import log_error, log_info, log_success, log_warning


class Z1_12_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(Robot_Num_Motors.Z1_12_Num_Motors)]


class Z1_12_ArmController:
    def __init__(self, config: Z1DualArmConfig):
        log_info("Initialize Z1_12_ArmController...")

        self.left_robot_ip = config.left_robot_ip
        self.left_robot_port1 = config.left_robot_port1
        self.left_robot_port2 = config.left_robot_port2
        self.right_robot_ip = config.right_robot_ip
        self.right_robot_port1 = config.right_robot_port1
        self.right_robot_port2 = config.right_robot_port2

        self.robot_kp = config.robot_kp
        self.robot_kd = config.robot_kd

        self.max_pos_speed = config.max_pos_speed
        self.init_pose_left = np.array(config.init_pose_left)
        self.init_pose_right = np.array(config.init_pose_right)
        self.init_pose = np.concatenate((self.init_pose_left, self.init_pose_right), axis=0)
        self.unit_test = config.unit_test
        self.control_dt = config.control_dt
        self.motors = config.motors
        self.mock = config.mock

        self.q_target = np.concatenate((self.init_pose_left, self.init_pose_right))
        self.tauff_target = np.zeros(len(Z1_12_JointArmIndex), dtype=np.float64)
        self.dq_target = np.zeros(len(Z1_12_JointArmIndex) // 2, dtype=np.float64)
        self.ddq_target = np.zeros(len(Z1_12_JointArmIndex) // 2, dtype=np.float64)
        self.ftip_target = np.zeros(len(Z1_12_JointArmIndex) // 2, dtype=np.float64)
        self.time_target = time.monotonic()
        self.arm_cmd = "schedule_waypoint"

        self.ctrl_lock = threading.Lock()
        self.lowstate_buffer = DataBuffer()
        self.stop_event = threading.Event()

        self.z1_left_arm_ik = Z1_Arm_IK(unit_test=self.unit_test, visualization=False)
        self.z1_right_arm_ik = Z1_Arm_IK(unit_test=self.unit_test, visualization=False)

        self.arm_indices_len = len(Z1_12_JointArmIndex) // 2

        self.is_connected = False

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

    def initialize_arm(self, ip: str, port1: int, port2: int, name: str):
        """Initialize z1."""
        arm = unitree_z1.ArmInterface(ip, port1, port2)
        arm_model = arm._ctrlComp.armModel
        arm.setFsmLowcmd()
        return arm, arm_model

    def set_control_gains(self, kp, kd):
        """Initialize kp kd."""
        for arm in [self.z1_left, self.z1_right]:
            arm.lowcmd.setControlGain(kp, kd)
            arm.sendRecv()

    def connect(self):
        try:
            if self.is_connected:
                raise RobotDeviceAlreadyConnectedError(
                    "Z1_Dual_Arm is already connected. Do not run `robot.connect()` twice."
                )
            # Initialize arms
            self.z1_left, self.z1_left_model = self.initialize_arm(
                self.left_robot_ip, self.left_robot_port1, self.left_robot_port2, "left"
            )
            self.z1_right, self.z1_right_model = self.initialize_arm(
                self.right_robot_ip, self.right_robot_port1, self.right_robot_port2, "right"
            )

            # Set control gains
            self.set_control_gains(self.robot_kp, self.robot_kd)

            # initialize subscribe thread
            self.subscribe_thread = self._start_daemon_thread(self._subscribe_motor_state, name="z1._subscribe_motor_state")

            while not self.lowstate_buffer.get_data():
                time.sleep(0.01)
                log_warning("[Z1_12_ArmController] Waiting Get Data...")

            self.publish_thread = self._start_daemon_thread(self._ctrl_motor_state, name="z1_dual._ctrl_motor_state")
            self.is_connected = True

        except Exception as e:
            self.disconnect()
            log_error(f"❌ Error in Z1_12_ArmController.connect: {e}")

    def _subscribe_motor_state(self):
        while True:
            msg = {
                "q": np.concatenate([self.z1_left.lowstate.getQ(), self.z1_right.lowstate.getQ()], axis=0),
                "dq": np.concatenate([self.z1_left.lowstate.getQd(), self.z1_right.lowstate.getQd()], axis=0),
            }
            if msg is not None:
                lowstate = Z1_12_LowState()
                for id in range(Robot_Num_Motors.Z1_12_Num_Motors):
                    lowstate.motor_state[id].q = msg["q"][id]
                    lowstate.motor_state[id].dq = msg["dq"][id]
                self.lowstate_buffer.set_data(lowstate)
            time.sleep(self.control_dt)

    def _update_z1_arm(
        self,
        arm,
        arm_model,
        q: np.ndarray,
        qd: np.ndarray | None = None,
        qdd: np.ndarray | None = None,
        ftip: np.ndarray | None = None,
        tau: np.ndarray | None = None,
    ):
        """Update the state and command of a given robotic arm."""
        arm.q = q
        arm.qd = self.dq_target if qd is None else qd
        qdd = self.ddq_target if qdd is None else qdd
        ftip = self.ftip_target if ftip is None else ftip
        arm.tau = arm_model.inverseDynamics(arm.q, arm.qd, qdd, ftip) if tau is None else tau
        arm.setArmCmd(arm.q, arm.qd, arm.tau)
        arm.sendRecv()

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
            self._update_z1_arm(
                self.z1_left, self.z1_left_model, self.pose_interp(time.monotonic())[: self.arm_indices_len]
            )
            self._update_z1_arm(
                self.z1_right, self.z1_right_model, self.pose_interp(time.monotonic())[self.arm_indices_len :]
            )
            time.sleep(self.control_dt)

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
        self._update_z1_arm(
            arm=self.z1_left,
            arm_model=self.z1_left_model,
            q=self.pose_interp(t_now)[: self.arm_indices_len],
            tau=arm_tauff_target[: self.arm_indices_len] if arm_tauff_target is not None else arm_tauff_target,
        )
        self._update_z1_arm(
            arm=self.z1_right,
            arm_model=self.z1_right_model,
            q=self.pose_interp(t_now)[self.arm_indices_len :],
            tau=arm_tauff_target[self.arm_indices_len :] if arm_tauff_target is not None else arm_tauff_target,
        )

        time.sleep(max(0, self.control_dt - (time.perf_counter() - start_time)))

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
            log_error(f"❌ Error in Z1ArmController._ctrl_motor_state: {e}")

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

    def read_current_arm_q(self) -> np.ndarray | None:
        """Return current state q of the left and right arm motors."""
        return np.array([self.lowstate_buffer.get_data().motor_state[id].q for id in Z1_12_JointArmIndex])

    def read_current_arm_dq(self) -> np.ndarray | None:
        """Return current state dq of the left and right arm motors."""
        return np.array([self.lowstate_buffer.get_data().motor_state[id].dq for id in Z1_12_JointArmIndex])

    def arm_ik(self, l_tf_target, r_tf_target) -> np.ndarray | None:
        current_lr_arm_q = self.read_current_arm_q()
        current_lr_arm_dq = self.read_current_arm_dq()

        left_sol_q, left_sol_tauff = self.z1_left_arm_ik.solve_ik(
            l_tf_target,
            current_lr_arm_q[: self.arm_indices_len],
            current_lr_arm_dq[: self.arm_indices_len],
        )
        right_sol_q, right_sol_tauff = self.z1_right_arm_ik.solve_ik(
            r_tf_target,
            current_lr_arm_q[self.arm_indices_len :],
            current_lr_arm_dq[self.arm_indices_len :],
        )

        sol_q = np.concatenate([left_sol_q, right_sol_q], axis=0)
        sol_tauff = np.concatenate([left_sol_tauff, right_sol_tauff], axis=0)

        return sol_q, sol_tauff

    def arm_fk(self, left_q: np.ndarray | None = None, right_q: np.ndarray | None = None) -> np.ndarray | None:
        left = self.z1_left_arm_ik.solve_fk(
            left_q if left_q is not None else self.read_current_arm_q()[: self.arm_indices_len]
        )
        right = self.z1_right_arm_ik.solve_fk(
            right_q if right_q is not None else self.read_current_arm_q()[self.arm_indices_len :]
        )

        return left, right

    def go_start(self):
        self._drive_to_waypoint(target_pose=self.init_pose, t_insert_time=1.0)
        log_success("Go Start OK!\n")

    def go_home(self):
        if self.mock:
            self.stop_event.set()
            # self.subscribe_thread.join()
            # self.publish_thread.join()
            time.sleep(1)
        else:
            self.is_connected = False

            self.z1_right.loopOn()
            self.z1_right.backToStart()
            self.z1_right.loopOff()

            self.z1_left.loopOn()
            self.z1_left.backToStart()
            self.z1_left.loopOff()

        time.sleep(0.5)
        log_success("Go Home OK!\n")

    def disconnect(self):
        self.is_connected = False
        self.go_home()

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
