# Build your own robot

### Add your own config ((unitree_deploy/robot/robot_configs.py))

The base class of robot config is defined as **UnitreeRobotConfig**

```python
class UnitreeRobotConfig(RobotConfig):
    cameras: dict[str, CameraConfig] = field(default_factory=lambda: {})            # Corresponding to your own camera
    arm: dict[str, ArmConfig] = field(default_factory=lambda: {})                   # Corresponding to your own arm
    endeffector: dict[str, EndEffectorConfig] = field(default_factory=lambda: {})   # Corresponding to your own end-effector

    mock: bool = False                                                              # Simulation [To be implemented, for debugging, to check some class definitions and message type formats]
```

Specific example: separately fill in \[name\]:robot_devies → cameras,
arm, endeffector.\
If not provided, they default to empty and will not affect the system.\
(In principle, different robot_devies and different quantities can be
constructed.)

```python
class Z1dual_Dex1_Opencv_RobotConfig(UnitreeRobotConfig):

    # Troubleshooting: If one of your IntelRealSense cameras freezes during
    # data recording due to bandwidth limit, you might need to plug the camera
    # into another USB hub or PCIe card.
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {                         # Add corresponding configs for different cameras [name]:OpenCVCameraConfig + required parameters
            "cam_high": OpenCVCameraConfig(
                camera_index="/dev/video0",
                fps=30,
                width=640,
                height=480,
            ),
            "cam_left_wrist": OpenCVCameraConfig(
                camera_index="/dev/video2",
                fps=30,
                width=640,
                height=480,
            ),
            "cam_right_wrist": OpenCVCameraConfig(
                camera_index="/dev/video4",
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    arm: dict[str, ArmConfig] = field(
        default_factory=lambda: {
            "z1_dual": Z1DualArmConfig(                 # Add corresponding configs for different arms [name]:Z1DualArmConfig + required parameters
                unit_test = False,
                init_pose_left = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                init_pose_right = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                control_dt = 1/500.0,
                motors={
                    # name: (index, model)
                    "kLeftWaist":       [0, "z1-joint"],
                    "kLeftShoulder":    [1, "z1-joint"],
                    "kLeftElbow":       [2, "z1-joint"],
                    "kLeftForearmRoll": [3, "z1-joint"],
                    "kLeftWristAngle":  [4, "z1-joint"],
                    "kLeftWristRotate": [5, "z1-joint"],

                    "kRightWaist":          [7, "z1-joint"],
                    "kRightShoulder":       [8, "z1-joint"],
                    "kRightElbow":          [9, "z1-joint"],
                    "kRightForearmRoll":    [10, "z1-joint"],
                    "kRightWristAngle":     [11, "z1-joint"],
                    "kRightWristRotate":    [12, "z1-joint"],
                    },
                ),
        }
    )

    endeffector: dict[str, EndEffectorConfig] = field(
        default_factory=lambda: {
            "gripper": GripperConfig(                   # Add corresponding configs for different end-effectors [name]:GripperConfig + required parameters
                unit_test = False,
                unit_test = True,
                control_dt = 1/250,
                motors={
                    # name: (index, model)
                    "kLeftGripper":  [0, "z1_gripper-joint"],
                    "kRightGripper": [1, "z1_gripper-joint"],
                },
            ),
        }
    )

    mock: bool = False
```

---

### robot utils ((unitree_deploy/robot/utils.py))

```python
Implementation of the Robot base class

class Robot(Protocol):
    robot_type: str
    features: dict

    def connect(self): ...                              # Connect devices (including cameras, arms, end-effectors of robot_devies)
    def capture_observation(self): ...                  # capture_observation (Get current state, including data from camera + arm + end-effector)
    def send_action(self, action): ...                  # send_action (Send action to arm + end-effector actuators, can be used for model inference and data replay)
    def disconnect(self): ...                           # Disconnect devices
```

External calls **make_robot_from_config** and **make_robot** are used in
**control_robot**, to initialize the robot and implement specific
functions.

---

### manipulator ((unitree_deploy/robot/manipulator.py))

UnitreeRobot implements initialization by calling
**UnitreeRobotConfig**.

```python
    Several important parts of the implementation

    def capture_observation(self):                                             # Get current obs, return { observation.state, observation.images}

    def send_action(                                                           # Model inference and data replay, receives action + time
            self, action: torch.Tensor, t_command_target:float|None = None
        ) -> torch.Tensor:

    # Here we input device data
    # Output (arm + end-effector) joint angle positions, end-effector positions, or other data conversion (IK is implemented here!)
    # Output is uniformly converted into joint angle positions {"left":arm_joint_points, "roght":arm_joint_points} + {"left":endeffector_joint_points, "roght":endeffector_joint_points}
    # Why consider left and right? Because this separates single-arm cases, and different arms and different end-effectors.
    # This way, the implementation can work properly.
    def convert_data_based_on_robot_type(self, robot_type: str, leader_pos: dict[str, np.ndarray]
        ) -> None | tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
```
