# for gripper
import threading
import time

import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber  # dds
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_  # idl

from unitree_deploy.robot_devices.arm.arm_indexs import Gripper_Sigle_JointIndex
from unitree_deploy.robot_devices.endeffector.configs import Dex1_GripperConfig
from unitree_deploy.robot_devices.robots_devices_utils import DataBuffer, MotorState, Robot_Num_Motors
from unitree_deploy.utils.joint_trajcetory_inter import JointTrajectoryInterpolator
from unitree_deploy.utils.rich_logger import log_error, log_info, log_warning


class Gripper_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(Robot_Num_Motors.Dex1_Gripper_Num_Motors)]


class Dex1_Gripper_Controller:
    def __init__(self, config: Dex1_GripperConfig):
        log_info("Initialize Dex1_Gripper_Controller...")

        self.init_pose = np.array(config.init_pose)

        self.motors = config.motors
        self.mock = config.mock
        self.control_dt = config.control_dt
        self.unit_test = config.unit_test
        self.max_pos_speed = config.max_pos_speed

        self.topic_gripper_command = config.topic_gripper_command
        self.topic_gripper_state = config.topic_gripper_state

        self.q_target = np.zeros(1)
        self.tauff_target = np.zeros(1)
        self.time_target = time.monotonic()
        self.gripper_cmd = "schedule_waypoint"

        self.lowstate_buffer = DataBuffer()
        self.ctrl_lock = threading.Lock()

        self.MAX_DIST = 5.45
        self.MIN_DIST = 0.0
        self.DELTA_GRIPPER_CMD = 0.18

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

    def connect(self):
        try:
            if self.unit_test:
                ChannelFactoryInitialize(0)

            dq = 0.0
            tau = 0.0
            kp = 10.0
            kd = 0.05

            # initialize gripper cmd msg
            self.gripper_msg = MotorCmds_()
            self.gripper_msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(len(Gripper_Sigle_JointIndex))]
            for id in Gripper_Sigle_JointIndex:
                self.gripper_msg.cmds[id].dq = dq
                self.gripper_msg.cmds[id].tau_est = tau
                self.gripper_msg.cmds[id].kp = kp
                self.gripper_msg.cmds[id].kd = kd

            # initialize handcmd publisher and handstate subscriber
            self.GripperCmb_publisher = ChannelPublisher(self.topic_gripper_command, MotorCmds_)
            self.GripperCmb_publisher.Init()

            self.GripperState_subscriber = ChannelSubscriber(self.topic_gripper_state, MotorStates_)
            self.GripperState_subscriber.Init()

            # initialize subscribe thread
            self.subscribe_state_thread = threading.Thread(target=self._subscribe_gripper_motor_state)
            self.subscribe_state_thread.daemon = True
            self.subscribe_state_thread.start()

            while not self.lowstate_buffer.get_data():
                time.sleep(0.01)
                log_warning("[Dex1_Gripper_Controller] Waiting to subscribe dds...")

            self.gripper_control_thread = threading.Thread(target=self._ctrl_gripper_motor)
            self.gripper_control_thread.daemon = True
            self.gripper_control_thread.start()

            self.is_connected = True

        except Exception as e:
            self.disconnect()
            log_error(f"❌ Error in Dex1_Gripper_Controller.connect: {e}")

    def _subscribe_gripper_motor_state(self):
        try:
            while True:
                gripper_msg = self.GripperState_subscriber.Read()
                if gripper_msg is not None:
                    lowstate = Gripper_LowState()
                    for idx, id in enumerate(Gripper_Sigle_JointIndex):
                        lowstate.motor_state[idx].q = gripper_msg.states[id].q
                        lowstate.motor_state[idx].dq = gripper_msg.states[id].dq
                    self.lowstate_buffer.set_data(lowstate)
                time.sleep(0.002)
        except Exception as e:
            self.disconnect()
            log_error(f"❌ Error in Dex1_Gripper_Controller._subscribe_gripper_motor_state: {e}")

    def _update_gripper(self, gripper_q_target: np.ndarray):
        current_qs = np.array([self.lowstate_buffer.get_data().motor_state[id].q for id in Gripper_Sigle_JointIndex])
        clamped_qs = np.clip(gripper_q_target, current_qs - self.DELTA_GRIPPER_CMD, current_qs + self.DELTA_GRIPPER_CMD)
        """set current left, right gripper motor state target q"""
        for idx, id in enumerate(Gripper_Sigle_JointIndex):
            self.gripper_msg.cmds[id].q = np.array(clamped_qs)[idx]
        self.GripperCmb_publisher.Write(self.gripper_msg)

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
            self._update_gripper(self.pose_interp(time.monotonic()))
            time.sleep(self.control_dt)

    def _ctrl_gripper_motor(self):
        try:
            self.pose_interp = JointTrajectoryInterpolator(
                times=[time.monotonic()],
                joint_positions=[self.read_current_endeffector_q()],
            )

            gripper_q_target = self.read_current_endeffector_q()
            gripper_tauff_target = self.tauff_target
            gripper_time_target = time.monotonic()
            gripper_cmd = "schedule_waypoint"

            last_waypoint_time = time.monotonic()
            while True:
                start_time = time.perf_counter()
                t_now = time.monotonic()
                with self.ctrl_lock:
                    gripper_q_target = self.q_target
                    gripper_tauff_target = self.tauff_target  # noqa: F841
                    gripper_time_target = self.time_target
                    gripper_cmd = self.gripper_cmd

                if gripper_cmd is None:
                    self._update_gripper(gripper_q_target)
                    # time.sleep(max(0, (self.control_dt - (time.perf_counter() - start_time))))
                elif gripper_cmd == "drive_to_waypoint":
                    self._drive_to_waypoint(target_pose=gripper_q_target, t_insert_time=0.8)

                elif gripper_cmd == "schedule_waypoint":
                    target_time = time.monotonic() - time.perf_counter() + gripper_time_target
                    curr_time = t_now + self.control_dt
                    target_time = max(target_time, curr_time + self.control_dt)

                    self.pose_interp = self.pose_interp.schedule_waypoint(
                        pose=gripper_q_target,
                        time=target_time,
                        max_pos_speed=self.max_pos_speed,
                        curr_time=curr_time,
                        last_waypoint_time=last_waypoint_time,
                    )
                    last_waypoint_time = target_time

                    self._update_gripper(self.pose_interp(t_now))
                time.sleep(max(0, (self.control_dt - (time.perf_counter() - start_time))))

        except Exception as e:
            self.disconnect()
            log_error(f"❌ Error in Dex1_Gripper_Controller._ctrl_gripper_motor: {e}")

    def read_current_endeffector_q(self) -> np.ndarray:
        # Motor inversion left is 1 and right is 0      TODO(gh): Correct this
        motor_states = np.array([self.lowstate_buffer.get_data().motor_state[id].q for id in Gripper_Sigle_JointIndex])
        return np.array(motor_states)

    def read_current_endeffector_dq(self) -> np.ndarray:
        # Motor inversion left is 1 and right is 0      TODO(gh): Correct this
        motor_states_dq = np.array(
            [self.lowstate_buffer.get_data().motor_state[id].dq for id in Gripper_Sigle_JointIndex]
        )
        return np.array(motor_states_dq)

    def write_endeffector(
        self,
        q_target: list[float] | np.ndarray,
        tauff_target: list[float] | np.ndarray = None,
        time_target: float | None = None,
        cmd_target: str | None = None,
    ):
        with self.ctrl_lock:
            self.q_target = q_target
            self.tauff_target = tauff_target
            self.time_target = time_target
            self.gripper_cmd = cmd_target

    def go_start(self):
        self._drive_to_waypoint(target_pose=self.init_pose, t_insert_time=0.8)

    def go_home(self):
        self._drive_to_waypoint(target_pose=self.init_pose, t_insert_time=0.8)

    def disconnect(self):
        self.is_connected = False
        # self.go_home()
