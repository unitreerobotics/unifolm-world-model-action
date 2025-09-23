import time

import cv2
import numpy as np
import tyro
from tqdm import tqdm

from unitree_deploy.robot_devices.cameras.configs import OpenCVCameraConfig
from unitree_deploy.robot_devices.cameras.utils import make_camera, make_cameras_from_configs
from unitree_deploy.utils.rich_logger import log_success


def usb_camera_default_factory():
    return {
        "cam_high": OpenCVCameraConfig(
            camera_index="/dev/video1",
            fps=30,
            width=640,
            height=480,
        ),
        "cam_left_wrist": OpenCVCameraConfig(
            camera_index="/dev/video3",
            fps=30,
            width=640,
            height=480,
        ),
        "cam_right_wrist": OpenCVCameraConfig(
            camera_index="/dev/video5",
            fps=30,
            width=640,
            height=480,
        ),
    }


def run_cameras(camera_style: int = 0):
    """
    Runs camera(s) based on the specified style.

    Args:
        camera_style (int):
            0 - Single camera (OpenCV).
            1 - Multiple cameras from config.
    """

    if camera_style == 0:
        # ========== Single camera ==========
        camera_kwargs = {"camera_type": "opencv", "camera_index": "/dev/video5", "mock": False}
        camera = make_camera(**camera_kwargs)
        camera.connect()
        log_success("Connecting camera.")

        while True:
            color_image = camera.read()
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            cv2.imshow("Camera", color_image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    elif camera_style == 1:
        # ========== Multi-camera from configs ==========
        cameras = make_cameras_from_configs(usb_camera_default_factory())

        for name in cameras:
            cameras[name].connect()
            log_success(f"Connecting {name} camera.")

        # Camera warm-up
        for _ in tqdm(range(20), desc="Camera warming up"):
            for name in cameras:
                cameras[name].async_read()
            time.sleep(1 / 30)

        while True:
            images = {}
            for name in cameras:
                images[name] = cameras[name].async_read()

            image_list = [
                np.stack([img.numpy()] * 3, axis=-1) if img.ndim == 2 else img.numpy() for img in images.values()
            ]

            stacked_image = np.hstack(image_list)
            cv2.imshow("Multi-Camera View", stacked_image)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                cv2.destroyAllWindows()
                break

    else:
        raise ValueError(f"Unsupported camera_style: {camera_style}")


if __name__ == "__main__":
    tyro.cli(run_cameras)
