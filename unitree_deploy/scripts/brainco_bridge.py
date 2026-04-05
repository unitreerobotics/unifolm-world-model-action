#!/usr/bin/env python3
"""
Bridge: ROS2 stark_node (Brainco Revo2) <-> TCP socket for unitree_deploy.

Run in the g1brainco conda env AFTER the stark_node is already running:

    conda activate g1brainco
    source ~/unitree_ros2/setup.sh
    source ~/unitree-g1-brainco-hand/ros2_stark_ws/install/setup.bash
    python ~/unifolm-world-model-action/unitree_deploy/scripts/brainco_bridge.py

Protocol: newline-delimited JSON over TCP on 127.0.0.1:9877
  Read state:   {"cmd": "get"}
    -> {"left": [6 floats 0-1], "right": [6 floats 0-1]}
  Send command: {"cmd": "set", "left": [6 floats 0-1], "right": [6 floats 0-1]}
    -> {"ok": true}

Finger order (both hands): thumb, thumb_aux, index, middle, ring, pinky
"""
import json
import socket
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from ros2_stark_interfaces.msg import MotorStatus

HOST = "127.0.0.1"
PORT = 9877
JOINT_NAMES = ["thumb", "thumb_aux", "index", "middle", "ring", "pinky"]


class BrancoBridge(Node):
    def __init__(self):
        super().__init__("brainco_bridge")
        self._lock = threading.Lock()
        self._positions_left = [0.0] * 6
        self._positions_right = [0.0] * 6

        self.sub_left = self.create_subscription(
            MotorStatus, "motor_status", self._cb_left, 10
        )
        self.sub_right = self.create_subscription(
            MotorStatus, "motor_status_r", self._cb_right, 10
        )
        self.pub_left = self.create_publisher(JointState, "joint_commands_left", 10)
        self.pub_right = self.create_publisher(JointState, "joint_commands_right", 10)
        self.get_logger().info("BrancoBridge node ready.")

    def _cb_left(self, msg):
        with self._lock:
            # MotorStatus.positions is uint8[6], range 0-255; map to 0.0-1.0
            self._positions_left = [p / 255.0 for p in msg.positions]

    def _cb_right(self, msg):
        with self._lock:
            self._positions_right = [p / 255.0 for p in msg.positions]

    def get_state(self):
        with self._lock:
            return list(self._positions_left), list(self._positions_right)

    def send_cmd(self, left_positions, right_positions):
        now = self.get_clock().now().to_msg()
        for positions, pub in [
            (left_positions, self.pub_left),
            (right_positions, self.pub_right),
        ]:
            msg = JointState()
            msg.header.stamp = now
            msg.name = JOINT_NAMES
            msg.position = [float(p) for p in positions]
            pub.publish(msg)


def _handle_client(conn, bridge):
    buf = ""
    try:
        while True:
            data = conn.recv(1024)
            if not data:
                break
            buf += data.decode("utf-8")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                req = json.loads(line)
                if req["cmd"] == "get":
                    left, right = bridge.get_state()
                    resp = json.dumps({"left": left, "right": right}) + "\n"
                    conn.sendall(resp.encode("utf-8"))
                elif req["cmd"] == "set":
                    bridge.send_cmd(req["left"], req["right"])
                    conn.sendall(json.dumps({"ok": True}).encode("utf-8") + b"\n")
    except Exception as e:
        bridge.get_logger().warn(f"[bridge] client error: {e}")
    finally:
        conn.close()


def _serve(bridge):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(5)
    bridge.get_logger().info(f"[bridge] TCP server listening on {HOST}:{PORT}")
    while True:
        conn, addr = server.accept()
        bridge.get_logger().info(f"[bridge] connection from {addr}")
        t = threading.Thread(target=_handle_client, args=(conn, bridge), daemon=True)
        t.start()


def main():
    rclpy.init()
    bridge = BrancoBridge()

    server_thread = threading.Thread(target=_serve, args=(bridge,), daemon=True)
    server_thread.start()

    rclpy.spin(bridge)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
