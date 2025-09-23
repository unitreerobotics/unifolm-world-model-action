# How to build your own cameras

### Define your own config for cameras (unitree_deploy/robot_devices/cameras/config.py)

```python
@CameraConfig.register_subclass("opencv") # Define and wrap your own cameras. Here use def __init__(self, config: OpenCVCameraConfig):
@dataclass
class OpenCVCameraConfig(CameraConfig):
    """
    Example of tested options for Intel Real Sense D405:

    OpenCVCameraConfig(0, 30, 640, 480)
    OpenCVCameraConfig(0, 60, 640, 480)
    OpenCVCameraConfig(0, 90, 640, 480)
    OpenCVCameraConfig(0, 30, 1280, 720)

    """
    # Define the required camera parameters
    camera_index: int
    fps: int | None = None
    width: int | None = None
    height: int | None = None
    color_mode: str = "rgb"
    channels: int | None = None
    rotation: int | None = None
    mock: bool = False

    def __post_init__(self):
        if self.color_mode not in ["rgb", "bgr"]:
            raise ValueError(
                f"`color_mode` is expected to be 'rgb' or 'bgr', but {self.color_mode} is provided."
            )

        self.channels = 3

        if self.rotation not in [-90, None, 90, 180]:
            raise ValueError(f"`rotation` must be in [-90, None, 90, 180] (got {self.rotation})")

# Default parameters go first [parameters that need to be customized],
# Non-default parameters go later [fixed parameters]
```

### Description of methods in your cameras class (unitree_deploy/robot_devices/cameras/utils.py)

```python
# Base class for cameras, extensible with required methods

class Camera(Protocol):
    def connect(self): ...
    def read(self, temporary_color: str | None = None) -> np.ndarray: ...   # Single-threaded reading
    def async_read(self) -> np.ndarray: ...                                 # Multi-threaded
    def disconnect(self): ...
```

How can external modules implement calls? Use **make_cameras_from_configs [based on configuration files]** to construct the `UnitreeRobot` class.  
**make_camera [based on camera_type]** is generally used for external module loading.

### Implementation of the `camera` class (unitree_deploy/robot_devices/camera/.../....py)

```python
    # These need to be completed, focusing on implementing these two parts
    def read(self, temporary_color: str | None = None) -> np.ndarray: ...   # Single-threaded reading
    def async_read(self) -> np.ndarray: ...                                 # Multi-threaded
```

All cameras use threading to implement `async_read` for internal read and write operations.
