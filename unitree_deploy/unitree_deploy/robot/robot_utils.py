from typing import Protocol

from unitree_deploy.robot.robot_configs import (
    G1_Dex1_Imageclint_RobotConfig,
    RobotConfig,
    Z1_Realsense_RobotConfig,
    Z1dual_Dex1_Opencv_RobotConfig,
    Z1dual_Dex1_Realsense_RobotConfig,
)


def get_arm_id(name, arm_type):
    return f"{name}_{arm_type}"


class Robot(Protocol):
    robot_type: str
    features: dict

    def connect(self): ...
    def capture_observation(self): ...
    def send_action(self, action): ...
    def disconnect(self): ...


def make_robot_config(robot_type: str, **kwargs) -> RobotConfig:
    if robot_type == "z1_realsense":
        return Z1_Realsense_RobotConfig(**kwargs)
    elif robot_type == "z1_dual_dex1_realsense":
        return Z1dual_Dex1_Realsense_RobotConfig(**kwargs)
    elif robot_type == "z1_dual_dex1_opencv":
        return Z1dual_Dex1_Opencv_RobotConfig(**kwargs)
    elif robot_type == "g1_dex1":
        return G1_Dex1_Imageclint_RobotConfig(**kwargs)
    else:
        raise ValueError(f"Robot type '{robot_type}' is not available.")


def make_robot_from_config(config: RobotConfig):
    from unitree_deploy.robot.robot import UnitreeRobot

    return UnitreeRobot(config)


def make_robot(robot_type: str, **kwargs) -> Robot:
    config = make_robot_config(robot_type, **kwargs)
    return make_robot_from_config(config)
