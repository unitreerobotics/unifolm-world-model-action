import math

import numpy as np
import pinocchio as pin


def generate_rotation(step: int, rotation_speed: float, max_step: int = 240):
    """Generate rotation (quaternions) and translation deltas for left and right arm motions."""
    angle = rotation_speed * step if step <= max_step // 2 else rotation_speed * (max_step - step)

    # Create rotation quaternion for left arm (around Y-axis)
    l_quat = pin.Quaternion(np.cos(angle / 2), 0, np.sin(angle / 2), 0)

    # Create rotation quaternion for right arm (around Z-axis)
    r_quat = pin.Quaternion(np.cos(angle / 2), 0, 0, np.sin(angle / 2))

    # Define translation increments for left and right arm
    delta_l = np.array([0.001, 0.001, 0.001]) * 1.2
    delta_r = np.array([0.001, -0.001, 0.001]) * 1.2

    # Reverse direction in second half of cycle
    if step > max_step // 2:
        delta_l *= -1
        delta_r *= -1

    return l_quat, r_quat, delta_l, delta_r


def sinusoidal_single_gripper_motion(period: float, amplitude: float, current_time: float) -> np.ndarray:
    value = amplitude * (math.sin(2 * math.pi * current_time / period) + 1) / 2
    return np.array([value*5])


def sinusoidal_gripper_motion(period: float, amplitude: float, current_time: float) -> np.ndarray:
    value = amplitude * (math.sin(2 * math.pi * current_time / period) + 1) / 2
    return np.array([value]*5)
