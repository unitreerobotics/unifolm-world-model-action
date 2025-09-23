import threading
import time
from typing import Callable

import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber  # dds
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_  # idl
from unitree_sdk2py.utils.crc import CRC

from unitree_deploy.robot_devices.arm.arm_indexs import G1_29_JointArmIndex, G1_29_JointIndex
from unitree_deploy.robot_devices.arm.configs import G1ArmConfig
from unitree_deploy.robot_devices.arm.g1_arm_ik import G1_29_ArmIK
from unitree_deploy.robot_devices.robots_devices_utils import (
    DataBuffer,
    MotorState,
    Robot_Num_Motors,
    RobotDeviceAlreadyConnectedError,
)
from unitree_deploy.utils.joint_trajcetory_inter import JointTrajectoryInterpolator
from unitree_deploy.utils.rich_logger import log_error, log_info, log_success, log_warning
from unitree_deploy.utils.run_simulation import MujicoSimulation, get_mujoco_sim_config


class G1_29_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(Robot_Num_Motors.G1_29_Num_Motors)]


class G1_29_ArmController:
    def __init__(self, config: G1ArmConfig):
        self.motors = config.motors
        self.mock = config.mock
        self.unit_test = config.unit_test
        self.init_pose = config.init_pose
        self.control_dt = config.control_dt

        self.max_pos_speed = config.max_pos_speed

        self.topic_low_command = config.topic_low_command
        self.topic_low_state = config.topic_low_state

        self.kp_high = config.kp_high
        self.kd_high = config.kd_high
        self.kp_low = config.kp_low
        self.kd_low = config.kd_low
        self.kp_wrist = config.kp_wrist
        self.kd_wrist = config.kd_wrist

        self.all_motor_q = None
        self.q_target = np.zeros(14)
        self.dq_target = np.zeros(14)
        self.tauff_target = np.zeros(14)
        self.time_target = time.monotonic()
        self.arm_cmd = "schedule_waypoint"

        self.lowstate_buffer = DataBuffer()
        self.g1_arm_ik = G1_29_ArmIK(unit_test=self.unit_test, visualization=False)

        self.stop_event = threading.Event()
        self.ctrl_lock = threading.Lock()

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

    def connect(self):
        try:
            if self.is_connected:
                raise RobotDeviceAlreadyConnectedError(
                    "G1_Arm is already connected. Do not call `robot.connect()` twice."
                )
            if self.mock:
                config = get_mujoco_sim_config(robot_type="g1")
                self.g1 = MujicoSimulation(config)
                time.sleep(1)
            else:
                # initialize lowcmd publisher and lowstate subscriber
                ChannelFactoryInitialize(0)
                self.lowcmd_publisher = ChannelPublisher(self.topic_low_command, LowCmd_)
                self.lowcmd_publisher.Init()
                self.lowstate_subscriber = ChannelSubscriber(self.topic_low_state, LowState_)
                self.lowstate_subscriber.Init()

            # initialize subscribe thread
            self.subscribe_thread = self._start_daemon_thread(
                self._subscribe_motor_state, name="g1._subscribe_motor_state"
            )

            while not self.lowstate_buffer.get_data():
                time.sleep(0.01)
                log_warning("[G1_29_ArmController] Waiting to subscribe dds...")

            if not self.mock:
                # initialize hg's lowcmd msg
                self.crc = CRC()
                self.msg = unitree_hg_msg_dds__LowCmd_()
                self.msg.mode_pr = 0
                self.msg.mode_machine = self._read_mode_machine()

                self.all_motor_q = self._read_current_motor_q()
                log_info(f"Current all body motor state q:\n{self.all_motor_q} \n")
                log_info(f"Current two arms motor state q:\n{self.read_current_arm_q()}\n")
                log_info("Lock all joints except two arms...\n")

                arm_indices = {member.value for member in G1_29_JointArmIndex}
                for id in G1_29_JointIndex:
                    self.msg.motor_cmd[id].mode = 1
                    if id.value in arm_indices:
                        if self._is_wrist_motor(id):
                            self.msg.motor_cmd[id].kp = self.kp_wrist
                            self.msg.motor_cmd[id].kd = self.kd_wrist
                        else:
                            self.msg.motor_cmd[id].kp = self.kp_low
                            self.msg.motor_cmd[id].kd = self.kd_low
                    else:
                        if self._is_weak_motor(id):
                            self.msg.motor_cmd[id].kp = self.kp_low
                            self.msg.motor_cmd[id].kd = self.kd_low
                        else:
                            self.msg.motor_cmd[id].kp = self.kp_high
                            self.msg.motor_cmd[id].kd = self.kd_high
                    self.msg.motor_cmd[id].q = self.all_motor_q[id]
                log_info("Lock OK!\n")

            # initialize publish thread
            self.publish_thread = self._start_daemon_thread(self._ctrl_motor_state, name="g1._ctrl_motor_state")
            self.is_connected = True

        except Exception as e:
            self.disconnect()
            log_error(f"❌ Error in G1_29_ArmController.connect: {e}")

    def _subscribe_motor_state(self):
        try:
            while not self.stop_event.is_set():
                lowstate = G1_29_LowState()
                if self.mock:
                    if self.g1.get_current_positions() is not None and len(self.g1.get_current_positions()) != 0:
                        for motor_id in range(Robot_Num_Motors.G1_29_Num_Motors):
                            lowstate.motor_state[motor_id].q = self.g1.get_current_positions()[motor_id]
                            lowstate.motor_state[motor_id].dq = 0.0
                    else:
                        print("[WARN] get_current_positions() failed: queue is empty.")
                else:
                    msg = self.lowstate_subscriber.Read()
                    if msg is not None:
                        for id in range(Robot_Num_Motors.G1_29_Num_Motors):
                            lowstate.motor_state[id].q = msg.motor_state[id].q
                            lowstate.motor_state[id].dq = msg.motor_state[id].dq
                self.lowstate_buffer.set_data(lowstate)
                time.sleep(self.control_dt)
        except Exception as e:
            self.disconnect()
            log_error(f"❌ Error in G1_29_ArmController._subscribe_motor_state: {e}")

    def _update_g1_arm(
        self,
        arm_q_target: np.ndarray,
        arm_dq_target: np.ndarray | None = None,
        arm_tauff_target: np.ndarray | None = None,
    ):
        if self.mock:
            self.g1.set_positions(arm_q_target)
        else:
            for idx, id in enumerate(G1_29_JointArmIndex):
                self.msg.motor_cmd[id].q = arm_q_target[idx]
                self.msg.motor_cmd[id].dq = arm_dq_target[idx]
                self.msg.motor_cmd[id].tau = arm_tauff_target[idx]

            self.msg.crc = self.crc.Crc(self.msg)
            self.lowcmd_publisher.Write(self.msg)

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
            start_time = time.perf_counter()

            cliped_arm_q_target = self.pose_interp(time.monotonic())
            self._update_g1_arm(cliped_arm_q_target, self.dq_target, self.tauff_target)

            time.sleep(max(0, (self.control_dt - (time.perf_counter() - start_time))))

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

        cliped_arm_q_target = self.pose_interp(t_now)
        self._update_g1_arm(cliped_arm_q_target, self.dq_target, arm_tauff_target)

        time.sleep(max(0, (self.control_dt - (time.perf_counter() - start_time))))

    def _ctrl_motor_state(self):
        # wait dds init done !!!
        time.sleep(2)

        self.pose_interp = JointTrajectoryInterpolator(
            times=[time.monotonic()], joint_positions=[self.read_current_arm_q()]
        )
        self.go_start()

        arm_q_target = self.read_current_arm_q()
        arm_tauff_target = self.tauff_target
        arm_time_target = time.monotonic()
        arm_cmd = "schedule_waypoint"

        last_waypoint_time = time.monotonic()
        while not self.stop_event.is_set():
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

                # target_time = time.monotonic() - time.perf_counter() + arm_time_target
                # curr_time = t_now + self.control_dt

                # target_time = max(target_time, curr_time + self.control_dt)

                # self.pose_interp = self.pose_interp.schedule_waypoint(
                #     pose=arm_q_target,
                #     time=target_time,
                #     max_pos_speed=self.max_pos_speed,
                #     curr_time=curr_time,
                #     last_waypoint_time=last_waypoint_time
                # )
                # last_waypoint_time = target_time

                # cliped_arm_q_target = self.pose_interp(t_now)
                # self._update_g1_arm(cliped_arm_q_target, self.dq_target, arm_tauff_target)

                # time.sleep(max(0, (self.control_dt - (time.perf_counter() - start_time))))

    def _read_mode_machine(self):
        """Return current dds mode machine."""
        return self.lowstate_subscriber.Read().mode_machine

    def _read_current_motor_q(self) -> np.ndarray | None:
        """Return current state q of all body motors."""
        return np.array([self.lowstate_buffer.get_data().motor_state[id].q for id in G1_29_JointIndex])

    def read_current_arm_q(self) -> np.ndarray | None:
        """Return current state q of the left and right arm motors."""
        return np.array([self.lowstate_buffer.get_data().motor_state[id].q for id in G1_29_JointArmIndex])

    def read_current_arm_dq(self) -> np.ndarray | None:
        """Return current state dq of the left and right arm motors."""
        return np.array([self.lowstate_buffer.get_data().motor_state[id].dq for id in G1_29_JointArmIndex])

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
            self.tauff_target = tauff_target if tauff_target is not None else self.arm_tau(self.q_target)
            self.time_target = time_target
            self.arm_cmd = cmd_target

    def arm_ik(
        self, l_ee_target: list[float] | np.ndarray, r_ee_target: list[float] | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | None:
        return self.g1_arm_ik.solve_ik(l_ee_target, r_ee_target, self.read_current_arm_q(), self.read_current_arm_dq())

    def arm_tau(
        self, current_arm_q: np.ndarray | None = None, current_arm_dq: np.ndarray | None = None
    ) -> np.ndarray | None:
        return self.g1_arm_ik.solve_tau(current_arm_q, current_arm_dq)

    def arm_fk(self, q: np.ndarray | None = None) -> np.ndarray | None:
        pass

    def go_start(self):
        self._drive_to_waypoint(target_pose=self.init_pose, t_insert_time=2.0)
        log_success("[G1_29_ArmController] Go Start OK!\n")

    def go_home(self):
        if self.mock:
            self.stop_event.set()
            # self.subscribe_thread.join()
            # self.publish_thread.join()

            time.sleep(1)
            # self.g1.stop()

        else:
            self.stop_event.set()
            self.publish_thread.join()

            self._drive_to_waypoint(target_pose=self.init_pose, t_insert_time=2.0)
        log_success("[G1_29_ArmController] Go Home OK!\n")

    def disconnect(self):
        self.is_connected = False
        self.go_home()

    def _is_weak_motor(self, motor_index):
        weak_motors = [
            G1_29_JointIndex.kLeftAnklePitch.value,
            G1_29_JointIndex.kRightAnklePitch.value,
            # Left arm
            G1_29_JointIndex.kLeftShoulderPitch.value,
            G1_29_JointIndex.kLeftShoulderRoll.value,
            G1_29_JointIndex.kLeftShoulderYaw.value,
            G1_29_JointIndex.kLeftElbow.value,
            # Right arm
            G1_29_JointIndex.kRightShoulderPitch.value,
            G1_29_JointIndex.kRightShoulderRoll.value,
            G1_29_JointIndex.kRightShoulderYaw.value,
            G1_29_JointIndex.kRightElbow.value,
        ]
        return motor_index.value in weak_motors

    def _is_wrist_motor(self, motor_index):
        wrist_motors = [
            G1_29_JointIndex.kLeftWristRoll.value,
            G1_29_JointIndex.kLeftWristPitch.value,
            G1_29_JointIndex.kLeftWristyaw.value,
            G1_29_JointIndex.kRightWristRoll.value,
            G1_29_JointIndex.kRightWristPitch.value,
            G1_29_JointIndex.kRightWristYaw.value,
        ]
        return motor_index.value in wrist_motors
