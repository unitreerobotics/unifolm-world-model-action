import platform
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import IntEnum
from typing import Optional


class Robot_Num_Motors(IntEnum):
    Z1_6_Num_Motors = 6
    Z1_7_Num_Motors = 7
    Z1_12_Num_Motors = 12

    Dex1_Gripper_Num_Motors = 2
    G1_29_Num_Motors = 35


@dataclass
class MotorState:
    q: Optional[float] = None
    dq: Optional[float] = None
    tau: Optional[float] = None


class DataBuffer:
    def __init__(self) -> None:
        self.data = None
        self.lock = threading.Lock()

    def get_data(self):
        with self.lock:
            return self.data

    def set_data(self, data) -> None:
        with self.lock:
            self.data = data


class RobotDeviceNotConnectedError(Exception):
    """Exception raised when the robot device is not connected."""

    def __init__(
        self, message="This robot device is not connected. Try calling `robot_device.connect()` first."
    ):
        self.message = message
        super().__init__(self.message)


class RobotDeviceAlreadyConnectedError(Exception):
    """Exception raised when the robot device is already connected."""

    def __init__(
        self,
        message="This robot device is already connected. Try not calling `robot_device.connect()` twice.",
    ):
        self.message = message
        super().__init__(self.message)


def capture_timestamp_utc():
    return datetime.now(timezone.utc)


def busy_wait(seconds):
    if platform.system() == "Darwin":
        # On Mac, `time.sleep` is not accurate and we need to use this while loop trick,
        # but it consumes CPU cycles.
        end_time = time.perf_counter() + seconds
        while time.perf_counter() < end_time:
            pass
    else:
        # On Linux time.sleep is accurate
        if seconds > 0:
            time.sleep(seconds)


def precise_wait(t_end: float, slack_time: float = 0.001, time_func=time.monotonic):
    t_start = time_func()
    t_wait = t_end - t_start
    if t_wait > 0:
        t_sleep = t_wait - slack_time
        if t_sleep > 0:
            time.sleep(t_sleep)
        while time_func() < t_end:
            pass
    return
