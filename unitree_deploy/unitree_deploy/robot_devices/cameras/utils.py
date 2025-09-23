"""
@misc{cadene2024lerobot,
  author       = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Wolf, Thomas},
  title        = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in PyTorch},
  howpublished = {Available at: https://github.com/huggingface/lerobot},
  year         = {2024},
}
"""

from typing import Protocol

import numpy as np

from unitree_deploy.robot_devices.cameras.configs import (
    CameraConfig,
    ImageClientCameraConfig,
    IntelRealSenseCameraConfig,
    OpenCVCameraConfig,
)


# Defines a camera type
class Camera(Protocol):
    def connect(self): ...
    def read(self, temporary_color: str | None = None) -> np.ndarray: ...
    def async_read(self) -> np.ndarray: ...
    def disconnect(self): ...


def make_cameras_from_configs(camera_configs: dict[str, CameraConfig]) -> list[Camera]:
    cameras = {}

    for key, cfg in camera_configs.items():
        if cfg.type == "opencv":
            from unitree_deploy.robot_devices.cameras.opencv import OpenCVCamera

            cameras[key] = OpenCVCamera(cfg)

        elif cfg.type == "intelrealsense":
            from unitree_deploy.robot_devices.cameras.intelrealsense import IntelRealSenseCamera

            cameras[key] = IntelRealSenseCamera(cfg)

        elif cfg.type == "imageclient":
            from unitree_deploy.robot_devices.cameras.imageclient import ImageClientCamera

            cameras[key] = ImageClientCamera(cfg)
        else:
            raise ValueError(f"The motor type '{cfg.type}' is not valid.")

    return cameras


def make_camera(camera_type, **kwargs) -> Camera:
    if camera_type == "opencv":
        from unitree_deploy.robot_devices.cameras.opencv import OpenCVCamera

        config = OpenCVCameraConfig(**kwargs)
        return OpenCVCamera(config)

    elif camera_type == "intelrealsense":
        from unitree_deploy.robot_devices.cameras.intelrealsense import IntelRealSenseCamera

        config = IntelRealSenseCameraConfig(**kwargs)
        return IntelRealSenseCamera(config)

    elif camera_type == "imageclient":
        from unitree_deploy.robot_devices.cameras.imageclient import ImageClientCamera

        config = ImageClientCameraConfig(**kwargs)
        return ImageClientCamera(config)

    else:
        raise ValueError(f"The camera type '{camera_type}' is not valid.")
