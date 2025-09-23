# How to Build Your Own End-Effector [Currently dex_1 and dex_3 are available]

### Define your own config for the end-effector (unitree_deploy/robot_devices/endeffector/config.py)

```python
@EndEffectorConfig.register_subclass("gripper")  # Register your custom end-effector wrapper. Here it uses def __init__(self, config: GripperConfig):
@dataclass
class GripperConfig(EndEffectorConfig):
    motors: dict[str, tuple[int, str]]
    unit_test: bool = False
    control_dt: float = 1/200
    mock: bool = False

    def __post_init__(self):
        if self.control_dt < 0.002:
            raise ValueError(f"`control_dt` must > 1/500 (got {self.control_dt})")

# Default arguments should be placed first [parameters that may need to be customized],
# Non-default arguments should be placed later [fixed or less important parameters].
```

### Description of methods in your end-effector class (unitree_deploy/robot_devices/endeffector/utils.py)

```python
# Base class for EndEffector, extend with required methods

class EndEffector(Protocol):
    def connect(self): ...
    def disconnect(self): ...
    def motor_names(self): ...

    def read_current_endeffector_q(self): ...
    def read_current_endeffector_dq(self): ...
    def write_endeffector(self): ...

    def endeffector_ik(self): ...
```

How to call externally?
Use make_endeffector_motors_buses_from_configs → Construct the UnitreeRobot class based on the config file
Use make_endeffector_motors_bus → Construct based on endeffector_type (typically for external module loading)

### Implementation of your end-effector class (unitree_deploy/robot_devices/endeffector/.../....py)

```python
    # These methods need to be implemented and completed
    def connect(self): ...
    def disconnect(self): ...
    def motor_names(self): ...
    # connect() and disconnect() should handle initialization and homing respectively

    def read_current_endeffector_q(self): ...
    def read_current_endeffector_dq(self): ...
    # Outputs should be unified as np.ndarray

    def write_endeffector(self): ...
    # Write control commands here

    def arm_ik(self): ...
    # Wrap IK into your own arm class, to be called externally

    # Private/protected properties
    # (for reading motor names, IDs, etc. These will be used in UnitreeRobot class for dataset encapsulation)
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

For arms, use threading to implement \_subscribe_gripper_motor_state (thread for reading motor states),\_ctrl_gripper_motor (thread for motor control),Both threads should run internally within the class.
