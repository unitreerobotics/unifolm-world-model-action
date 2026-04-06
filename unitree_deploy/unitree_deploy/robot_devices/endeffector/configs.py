import abc
from dataclasses import dataclass

import draccus
import numpy as np


@dataclass
class EndEffectorConfig(draccus.ChoiceRegistry, abc.ABC):
    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@EndEffectorConfig.register_subclass("dex_1")
@dataclass
class Dex1_GripperConfig(EndEffectorConfig):
    motors: dict[str, tuple[int, str]]
    unit_test: bool = False
    init_pose: list | None = None
    control_dt: float = 1 / 200
    mock: bool = False
    max_pos_speed: float = 180 * (np.pi / 180) * 2
    topic_gripper_command: str = "rt/unitree_actuator/cmd"
    topic_gripper_state: str = "rt/unitree_actuator/state"

    def __post_init__(self):
        if self.control_dt < 0.002:
            raise ValueError(f"`control_dt` must > 1/500 (got {self.control_dt})")


@EndEffectorConfig.register_subclass("brainco_dual")
@dataclass
class BrancoDualHandConfig(EndEffectorConfig):
    """Config for a pair of Brainco Revo2 five-finger hands controlled via
    the brainco_bridge TCP server (see unitree_deploy/scripts/brainco_bridge.py).

    USB ports: left hand -> /dev/ttyUSB1, right hand -> /dev/ttyUSB2
    (as configured in ros2_stark_ws/src/ros2_stark_controller/config/params_v2_double.yaml)
    """

    bridge_host: str = "127.0.0.1"
    bridge_port: int = 9877
    # 12 DOFs total: 6 per hand (thumb, thumb_aux, index, middle, ring, pinky), range 0.0-1.0
    num_motors: int = 12
    init_pose: list | None = None  # 12 values; None => zeros (open hands)
    control_dt: float = 1 / 50.0  # stark_node timer runs at 20 Hz; 50 Hz is safe
