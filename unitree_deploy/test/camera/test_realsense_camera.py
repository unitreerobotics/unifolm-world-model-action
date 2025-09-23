import time

import cv2
import numpy as np

from unitree_deploy.robot_devices.cameras.configs import IntelRealSenseCameraConfig
from unitree_deploy.robot_devices.cameras.utils import make_cameras_from_configs
from unitree_deploy.utils.rich_logger import log_success


def run_camera():
    # ===========================================
    def intelrealsense_camera_default_factory():
        return {
            "cam_high": IntelRealSenseCameraConfig(
                serial_number="044122071036",
                fps=30,
                width=640,
                height=480,
            ),
            "cam_wrist": IntelRealSenseCameraConfig(
                serial_number="419122270615",
                fps=30,
                width=640,
                height=480,
            ),
        }

    # ===========================================

    cameras = make_cameras_from_configs(intelrealsense_camera_default_factory())
    for name in cameras:
        cameras[name].connect()
        log_success(f"Connecting {name} cameras.")

    for _ in range(20):
        for name in cameras:
            cameras[name].async_read()
        time.sleep(1 / 30)

    while True:
        images = []
        for name in cameras:
            frame = cameras[name].async_read()
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.putText(frame, name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                images.append(frame)

        if images:
            rows = []
            for i in range(0, len(images), 2):
                row = np.hstack(images[i : i + 2])
                rows.append(row)
            canvas = np.vstack(rows)

            cv2.imshow("All Cameras", canvas)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_camera()
