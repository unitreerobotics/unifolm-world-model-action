"""Brainco Revo2 dual-hand controller for unitree_deploy.

Communicates with the brainco_bridge TCP server (running in the g1brainco
ROS2 env) via newline-delimited JSON over a localhost TCP socket.

Controls both hands as a single 12-DOF end-effector:
  - positions 0–5:  left hand  (thumb, thumb_aux, index, middle, ring, pinky)
  - positions 6–11: right hand (thumb, thumb_aux, index, middle, ring, pinky)
  - all positions are in range 0.0 (open) to 1.0 (closed)

Before connecting, start the bridge in the g1brainco env:
    conda activate g1brainco
    source ~/unitree_ros2/setup.sh
    source ~/unitree-g1-brainco-hand/ros2_stark_ws/install/setup.bash
    python ~/unifolm-world-model-action/unitree_deploy/scripts/brainco_bridge.py
"""
import json
import threading

import numpy as np

from unitree_deploy.robot_devices.endeffector.configs import BrancoDualHandConfig
from unitree_deploy.utils.rich_logger import log_error, log_success, log_warning


class Brainco_DualHand_Controller:
    JOINT_NAMES = [
        "left_thumb", "left_thumb_aux", "left_index",
        "left_middle", "left_ring", "left_pinky",
        "right_thumb", "right_thumb_aux", "right_index",
        "right_middle", "right_ring", "right_pinky",
    ]

    def __init__(self, config: BrancoDualHandConfig):
        self.config = config
        self.bridge_host = config.bridge_host
        self.bridge_port = config.bridge_port
        self.init_pose = np.array(config.init_pose) if config.init_pose else np.zeros(12)
        self._sock = None
        self._lock = threading.Lock()
        self._buf = ""
        self.is_connected = False

    @property
    def motor_names(self) -> list[str]:
        return self.JOINT_NAMES

    def connect(self):
        import socket
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.connect((self.bridge_host, self.bridge_port))
            self._sock.settimeout(2.0)
            self.is_connected = True
            log_success("[Brainco_DualHand_Controller] Connected to bridge at "
                        f"{self.bridge_host}:{self.bridge_port}")
        except Exception as e:
            log_error(f"❌ Brainco bridge connect failed: {e}\n"
                      "  Make sure brainco_bridge.py is running in the g1brainco env.")

    def _send_recv(self, payload: dict) -> dict:
        with self._lock:
            msg = json.dumps(payload) + "\n"
            self._sock.sendall(msg.encode("utf-8"))
            while "\n" not in self._buf:
                chunk = self._sock.recv(1024).decode("utf-8")
                self._buf += chunk
            line, self._buf = self._buf.split("\n", 1)
            return json.loads(line)

    def read_current_endeffector_q(self) -> np.ndarray:
        try:
            resp = self._send_recv({"cmd": "get"})
            return np.array(resp["left"] + resp["right"], dtype=np.float64)
        except Exception as e:
            log_warning(f"[Brainco] read error: {e}")
            return np.zeros(12)

    def read_current_endeffector_dq(self) -> np.ndarray:
        # Brainco stark_node does not expose velocities via motor_status
        return np.zeros(12)

    def write_endeffector(
        self,
        q_target,
        tauff_target=None,
        time_target=None,
        cmd_target=None,
    ):
        try:
            positions = np.clip(np.asarray(q_target, dtype=float), 0.0, 1.0)
            self._send_recv({
                "cmd": "set",
                "left": positions[:6].tolist(),
                "right": positions[6:].tolist(),
            })
        except Exception as e:
            log_warning(f"[Brainco] write error: {e}")

    def go_start(self):
        self.write_endeffector(self.init_pose)

    def go_home(self):
        # Open both hands fully
        self.write_endeffector(np.zeros(12))

    def disconnect(self):
        self.is_connected = False
        if self._sock:
            try:
                self.go_home()
                self._sock.close()
            except Exception:
                pass
