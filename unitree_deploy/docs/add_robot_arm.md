# How to Build Your Own Arm

### Define your own config for the robot arm (unitree_deploy/robot_devices/arm/config.py)

```python
@ArmConfig.register_subclass("z1")     # Register your custom arm wrapper. Here use def __init__(self, config: Z1DualArmConfig):
@dataclass
class Z1ArmConfig(ArmConfig):
    port: str
    motors: dict[str, tuple[int, str]]
    mock: bool = False
    init_pose_left: list = None
    init_pose_right: list = None
    control_dt: float = 1/500.0

# Default parameters go first [parameters that may need to be customized],
# Non-default parameters go later [fixed parameters]
```

### Description of methods in your arm class (unitree_deploy/robot_devices/arm/utils.py)

```python
# Base class for Arm, extensible with required methods

class Arm(Protocol):
    def connect(self): ...
    def disconnect(self): ...
    def motor_names(self): ...

    def read_current_motor_q(self): ...
    def read_current_arm_q(self): ...
    def read_current_arm_dq(self): ...
    def write_arm(self): ...

    def arm_ik(self): ...
```

How to implement external calls?
Use make_arm_motors_buses_from_configs [based on the config file] to construct the UnitreeRobot class.
Use make_arm_motors_bus [based on arm_type] which is generally used for external module loading.

### Implementation of the arm class (unitree_deploy/robot_devices/arm/.../....py)

```python
    # These methods need to be implemented and completed
    def connect(self): ...
    def disconnect(self): ...
    def motor_names(self): ...
    # connect() and disconnect() should handle initialization and homing respectively

    def read_current_motor_q(self): ...
    def read_current_arm_q(self): ...
    def read_current_arm_dq(self): ...
    # Outputs should be unified as np.ndarray

    def write_arm(self): ...
    # Write control commands here

    def arm_ik(self): ...
    # Wrap IK into your own arm class for external calling

    # Private/protected properties [for reading motor names, IDs, etc.]
    @property
    def motor_names(self) -> list[str]:
        return list(self.motors.keys())

    @property
    def motor_models(self) -> list[str]:
        return [model for _, model in self.motors.values()]

    @property
    def motor_indices(self) -> list[int]:
        return [idx for idx, _ in self.motors.values()]
```

All arms use threading to implement \_subscribe_motor_state and \_ctrl_motor_state threads for internal reading and writing within the class.
