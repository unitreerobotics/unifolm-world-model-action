"""The modification is derived from diffusion_policy/common/pose_trajectory_interpolator.py. Thank you for the outstanding contribution."""

import numbers
from typing import Union

import numpy as np
import scipy.interpolate as si


def joint_pose_distance(start_joint_angles, end_joint_angles):
    start_joint_angles = np.array(start_joint_angles)
    end_joint_angles = np.array(end_joint_angles)
    joint_angle_dist = np.linalg.norm(end_joint_angles - start_joint_angles)

    return joint_angle_dist


class JointTrajectoryInterpolator:
    def __init__(self, times: np.ndarray, joint_positions: np.ndarray):
        assert len(times) >= 1
        assert len(joint_positions) == len(times)
        self.num_joints = len(joint_positions[0])
        if not isinstance(times, np.ndarray):
            times = np.array(times)
        if not isinstance(joint_positions, np.ndarray):
            joint_positions = np.array(joint_positions)
        if len(times) == 1:
            self.single_step = True
            self._times = times
            self._joint_positions = joint_positions
        else:
            self.single_step = False
            assert np.all(times[1:] >= times[:-1])
            self.interpolators = si.interp1d(times, joint_positions, axis=0, assume_sorted=True)

    @property
    def times(self) -> np.ndarray:
        if self.single_step:
            return self._times
        else:
            return self.interpolators.x

    @property
    def joint_positions(self) -> np.ndarray:
        if self.single_step:
            return self._joint_positions
        else:
            n = len(self.times)
            joint_positions = np.zeros((n, self.num_joints))
            joint_positions = self.interpolators.y
            return joint_positions

    def trim(self, start_t: float, end_t: float) -> "JointTrajectoryInterpolator":
        assert start_t <= end_t
        times = self.times
        should_keep = (start_t < times) & (times < end_t)
        keep_times = times[should_keep]
        all_times = np.concatenate([[start_t], keep_times, [end_t]])
        all_times = np.unique(all_times)
        all_joint_positions = self(all_times)
        return JointTrajectoryInterpolator(times=all_times, joint_positions=all_joint_positions)

    def drive_to_waypoint(
        self,
        pose,
        time,
        curr_time,
        max_pos_speed=np.inf,
    ) -> "JointTrajectoryInterpolator":
        assert max_pos_speed > 0
        time = max(time, curr_time)

        curr_pose = self(curr_time)
        pos_dist = joint_pose_distance(curr_pose, pose)
        pos_min_duration = pos_dist / max_pos_speed
        duration = time - curr_time
        duration = max(duration, pos_min_duration)
        assert duration >= 0
        last_waypoint_time = curr_time + duration

        # insert new pose
        trimmed_interp = self.trim(curr_time, curr_time)
        times = np.append(trimmed_interp.times, [last_waypoint_time], axis=0)
        poses = np.append(trimmed_interp.joint_positions, [pose], axis=0)

        # create new interpolator
        final_interp = JointTrajectoryInterpolator(times, poses)
        return final_interp

    def schedule_waypoint(
        self, pose, time, max_pos_speed=np.inf, curr_time=None, last_waypoint_time=None
    ) -> "JointTrajectoryInterpolator":
        assert max_pos_speed > 0
        if last_waypoint_time is not None:
            assert curr_time is not None

        # trim current interpolator to between curr_time and last_waypoint_time
        start_time = self.times[0]
        end_time = self.times[-1]
        assert start_time <= end_time

        if curr_time is not None:
            if time <= curr_time:
                # if insert time is earlier than current time
                # no effect should be done to the interpolator
                return self
            # now, curr_time < time
            start_time = max(curr_time, start_time)

            if last_waypoint_time is not None:
                # if last_waypoint_time is earlier than start_time
                # use start_time
                end_time = curr_time if time <= last_waypoint_time else max(last_waypoint_time, curr_time)
            else:
                end_time = curr_time

        end_time = min(end_time, time)
        start_time = min(start_time, end_time)

        # end time should be the latest of all times except time after this we can assume order (proven by zhenjia, due to the 2 min operations)
        # Constraints:
        # start_time <= end_time <= time (proven by zhenjia)
        # curr_time <= start_time (proven by zhenjia)
        # curr_time <= time (proven by zhenjia)

        assert start_time <= end_time
        assert end_time <= time
        if last_waypoint_time is not None:
            if time <= last_waypoint_time:
                assert end_time == curr_time
            else:
                assert end_time == max(last_waypoint_time, curr_time)

        if curr_time is not None:
            assert curr_time <= start_time
            assert curr_time <= time

        trimmed_interp = self.trim(start_time, end_time)

        # determine speed
        duration = time - end_time
        end_pose = trimmed_interp(end_time)
        pos_dist = joint_pose_distance(pose, end_pose)

        joint_min_duration = pos_dist / max_pos_speed

        duration = max(duration, joint_min_duration)
        assert duration >= 0
        last_waypoint_time = end_time + duration

        # insert new pose
        times = np.append(trimmed_interp.times, [last_waypoint_time], axis=0)
        poses = np.append(trimmed_interp.joint_positions, [pose], axis=0)

        # create new interpolator
        final_interp = JointTrajectoryInterpolator(times, poses)
        return final_interp

    def __call__(self, t: Union[numbers.Number, np.ndarray]) -> np.ndarray:
        is_single = False
        if isinstance(t, numbers.Number):
            is_single = True
            t = np.array([t])

        joint_positions = np.zeros((len(t), self.num_joints))

        if self.single_step:
            joint_positions[:] = self._joint_positions[0]
        else:
            start_time = self.times[0]
            end_time = self.times[-1]
            t = np.clip(t, start_time, end_time)
            joint_positions[:, :] = self.interpolators(t)

        if is_single:
            joint_positions = joint_positions[0]
        return joint_positions


def generate_joint_positions(
    num_rows: int, num_cols: int, start: float = 0.0, step: float = 0.1, row_offset: float = 0.1
) -> np.ndarray:
    base_row = np.arange(start, start + step * num_cols, step)
    array = np.vstack([base_row + i * row_offset for i in range(num_rows)])
    return array


if __name__ == "__main__":
    # Example joint trajectory data (time in seconds, joint positions as an array of NUM_JOINTS joint angles)
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    joint_positions = generate_joint_positions(num_rows=5, num_cols=7, start=0.0, step=0.1, row_offset=0.1)
    interpolator = JointTrajectoryInterpolator(times, joint_positions)
    # Get joint positions at a specific time (e.g., t = 2.5 seconds)
    t = 0.1
    joint_pos_at_t = interpolator(t)
    print("Joint positions at time", t, ":", joint_pos_at_t)
