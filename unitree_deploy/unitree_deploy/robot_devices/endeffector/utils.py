from typing import Protocol

from unitree_deploy.robot_devices.endeffector.configs import (
    Dex1_GripperConfig,
    EndEffectorConfig,
)


class EndEffector(Protocol):
    def connect(self): ...
    def disconnect(self): ...
    def motor_names(self): ...

    def read_current_endeffector_q(self): ...
    def read_current_endeffector_dq(self): ...
    def write_endeffector(self): ...

    def retarget_to_endeffector(self): ...
    def endeffector_ik(self): ...

    def go_start(self): ...
    def go_home(self): ...


def make_endeffector_motors_buses_from_configs(
    endeffector_configs: dict[str, EndEffectorConfig],
) -> list[EndEffectorConfig]:
    endeffector_motors_buses = {}

    for key, cfg in endeffector_configs.items():
        if cfg.type == "dex_1":
            from unitree_deploy.robot_devices.endeffector.gripper import Dex1_Gripper_Controller

            endeffector_motors_buses[key] = Dex1_Gripper_Controller(cfg)

        else:
            raise ValueError(f"The motor type '{cfg.type}' is not valid.")

    return endeffector_motors_buses


def make_endeffector_motors_bus(endeffector_type: str, **kwargs) -> EndEffectorConfig:
    if endeffector_type == "dex_1":
        from unitree_deploy.robot_devices.endeffector.gripper import Dex1_Gripper_Controller

        config = Dex1_GripperConfig(**kwargs)
        return Dex1_Gripper_Controller(config)

    else:
        raise ValueError(f"The motor type '{endeffector_type}' is not valid.")
