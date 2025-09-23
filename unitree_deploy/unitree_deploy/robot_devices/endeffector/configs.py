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
