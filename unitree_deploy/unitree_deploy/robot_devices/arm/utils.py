from typing import Protocol

from unitree_deploy.robot_devices.arm.configs import ArmConfig, G1ArmConfig, Z1ArmConfig, Z1DualArmConfig


class Arm(Protocol):
    def connect(self): ...
    def disconnect(self): ...
    def motor_names(self): ...

    def read_current_motor_q(self): ...
    def read_current_arm_q(self): ...
    def read_current_arm_dq(self): ...
    def write_arm(self): ...

    def arm_ik(self): ...
    def arm_fk(self): ...
    def go_start(self): ...
    def go_home(self): ...


def make_arm_motors_buses_from_configs(armconfig: dict[str, ArmConfig]) -> list[Arm]:
    arm_motors_buses = {}

    for key, cfg in armconfig.items():
        if cfg.type == "z1":
            from unitree_deploy.robot_devices.arm.z1_arm import Z1ArmController

            arm_motors_buses[key] = Z1ArmController(cfg)
        elif cfg.type == "g1":
            from unitree_deploy.robot_devices.arm.g1_arm import G1_29_ArmController

            arm_motors_buses[key] = G1_29_ArmController(cfg)
        elif cfg.type == "z1_dual":
            from unitree_deploy.robot_devices.arm.z1_dual_arm import Z1_12_ArmController

            arm_motors_buses[key] = Z1_12_ArmController(cfg)
        else:
            raise ValueError(f"The motor type '{cfg.type}' is not valid.")

    return arm_motors_buses


def make_arm_motors_bus(arm_type: str, **kwargs) -> Arm:
    if arm_type == "z1":
        from unitree_deploy.robot_devices.arm.z1_arm import Z1ArmController

        config = Z1ArmConfig(**kwargs)
        return Z1ArmController(config)
    
    elif arm_type == "z1_dual":
        from unitree_deploy.robot_devices.arm.z1_dual_arm import Z1_12_ArmController

        config = Z1DualArmConfig(**kwargs)
        return Z1_12_ArmController(config)

    elif arm_type == "g1":
        from unitree_deploy.robot_devices.arm.g1_arm import G1_29_ArmController

        config = G1ArmConfig(**kwargs)
        return G1_29_ArmController(config)
    else:
        raise ValueError(f"The motor type '{arm_type}' is not valid.")
