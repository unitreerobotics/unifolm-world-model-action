# Unitree Deploy 

<div align="center">
  <p align="right">
    <span> 🌎English </span> | <a href="./docs/README_cn.md"> 🇨🇳中文 </a>
  </p>
</div>



This document provides instructions for setting up the deployment environment for Unitree G1 (with gripper) and Z1 platforms, including dependency installation, image service startup, and gripper control.

# 0. 📖 Introduction

This repository is used for model deployment with Unitree robots.

---

# 1. 🛠️ Environment Setup 

```bash
conda create -n unitree_deploy python=3.10 && conda activate unitree_deploy

conda install pinocchio -c conda-forge
pip install -e .

# Optional: Install lerobot dependencies
pip install -e ".[lerobot]"

git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python && pip install -e . && cd ..
```

---
# 2. 🚀 Start 

**Tip: Keep all devices on the same LAN**

## 2.1 🤖 Run G1 with Brainco Revo2 Five-Finger Hands

### 2.1.0 🧠 Architecture Overview

The Brainco Revo2 hands are driven by a ROS2 node (`stark_node`) that runs in a separate conda environment (`g1brainco`, Python 3.8, ROS2 Foxy). The `unitree_deploy` client runs in its own conda env (`unitree_deploy`, Python 3.10). Because these two environments cannot share the same Python process, a lightweight **TCP bridge** (`brainco_bridge.py`) mediates between them over `localhost:9877` using newline-delimited JSON.

```
stark_node (ROS2)  <-- /motor_status{,_r} / joint_commands_{left,right} -->
brainco_bridge.py (g1brainco env, TCP 127.0.0.1:9877)  <-->
Brainco_DualHand_Controller (unitree_deploy env, robot_client.py)
```

**Robot specifics (G1 23-DOF + Brainco):**
- Waist: 1 DOF (yaw/twist, index 12)
- Arms: 5 real DOFs each (shoulder ×3, elbow, forearm roll); **no wrist pitch/yaw**
- Hands: Brainco Revo2 five-finger, USB ports `/dev/ttyUSB1` (left) and `/dev/ttyUSB2` (right)
- Action space: 14 arm DOFs (4 phantom wrist slots kept for SDK compatibility) + 12 hand DOFs = **26 total**
- Hand DOF range: 0.0 = open, 1.0 = closed
- Finger order (both hands): thumb, thumb_aux, index, middle, ring, pinky

---

### 2.1.1 🛠️ Prerequisites

In addition to the base `unitree_deploy` env setup (Section 1), you need the `ros2_stark_ws` workspace built and the `g1brainco` conda env set up on the robot. Follow the [unitree-g1-brainco-hand](https://github.com/unitreerobotics/unitree-g1-brainco-hand) setup guide for this.

**Python version fix:** The Unitree G1 robot runs Python 3.10.12. If `pip install -e .` inside `unitree_deploy/` fails with `requires-python == 3.10.18`, edit `unitree_deploy/pyproject.toml` on the robot:

```toml
# Change:
requires-python = "==3.10.18"
# To:
requires-python = ">=3.10"
```

**pinocchio / libgomp fix (aarch64):** pinocchio ships a bundled `libgomp.so.1` that conflicts with the system GCC one on ARM64. Add this to `~/.bashrc` or set it each session before running the client:

```bash
export LD_PRELOAD=/home/unitree/miniconda3/envs/unitree_deploy/lib/libgomp.so.1
```

---

### 2.1.2 📷 Image Capture Service Setup (G1 Board)

[To open the image_server, follow these steps](https://github.com/unitreerobotics/xr_teleoperate?tab=readme-ov-file#31-%EF%B8%8F-image-service)

1. Connect to the G1 board:
    ```bash
    ssh unitree@192.168.123.164  # Password: 123
    ```

2. Activate the environment and start the image server:
    ```bash
    conda activate tv
    cd ~/image_server
    python image_server.py
    ```

---

### 2.1.3 🤚 Brainco Hand Setup (Three Terminals, All on Robot)

Open three SSH sessions to the robot (`ssh unitree@192.168.123.164`).

**Terminal 1 — stark_node (ROS2 hand driver):**

```bash
conda activate g1brainco
source ~/unitree_ros2/setup.sh
source ~/unitree-g1-brainco-hand/ros2_stark_ws/install/setup.bash
ros2 launch ros2_stark_controller stark_launch.py
```

Wait for the stark_node to connect to both hands via `/dev/ttyUSB1` and `/dev/ttyUSB2` before proceeding.

**Terminal 2 — brainco_bridge.py (ROS2 ↔ TCP bridge):**

```bash
conda activate g1brainco
source ~/unitree_ros2/setup.sh
source ~/unitree-g1-brainco-hand/ros2_stark_ws/install/setup.bash
python ~/unifolm-world-model-action/unitree_deploy/scripts/brainco_bridge.py
```

You should see: `[bridge] TCP server listening on 127.0.0.1:9877`

**Terminal 3 — robot client:**

```bash
conda activate unitree_deploy
cd ~/unifolm-world-model-action/unitree_deploy

# Required: preload libgomp to prevent pinocchio TLS error on aarch64
export LD_PRELOAD=/home/unitree/miniconda3/envs/unitree_deploy/lib/libgomp.so.1

# Required: point the client at the inference server (change IP as needed)
export INFERENCE_SERVER_HOST=192.168.123.222

python scripts/robot_client.py \
    --robot_type g1_brainco \
    --language_instruction "your task here"
```

The robot client will:
1. Connect to the G1 arm (DDS), Brainco hands (bridge), and image client
2. Move arms to Ready Mode position (`go_start()`)
3. Open both hands (all zeros)
4. Begin polling the inference server for actions

---

### 2.1.4 ✅ Testing

- **Brainco Bridge Test** (run in `g1brainco` env while bridge is running):
  ```bash
  python -c "
  import socket, json
  s = socket.socket(); s.connect(('127.0.0.1', 9877))
  s.sendall(b'{\"cmd\": \"get\"}\n')
  print(s.recv(1024).decode())
  s.close()
  "
  ```

- **G1 Arm Test:**
  ```bash
  python test/arm/g1/test_g1_arm.py
  ```

- **Image Client Camera Test:**
  ```bash
  python test/camera/test_image_client_camera.py
  ```

---

## 2.2 🤖 Run G1 with Dex_1 Gripper

### 2.2.1 📷 Image Capture Service Setup (G1 Board) 

[To open the image_server, follow these steps](https://github.com/unitreerobotics/xr_teleoperate?tab=readme-ov-file#31-%EF%B8%8F-image-service)
1. Connect to the G1 board:
    ```bash
    ssh unitree@192.168.123.164  # Password: 123
    ```

2. Activate the environment and start the image server:
    ```bash
    conda activate tv
    cd ~/image_server
    python image_server.py
    ```

---

### 2.2.2 🤏 Dex_1 Gripper Service Setup (Development PC2)

Refer to the [Dex_1 Gripper Installation Guide](https://github.com/unitreerobotics/dex1_1_service?tab=readme-ov-file#1--installation) for detailed setup instructions.

1. Navigate to the service directory:
    ```bash
    cd ~/dex1_1_service/build
    ```

2. Start the gripper service, **ifconfig examines its own dds networkInterface**:
    ```bash
    sudo ./dex1_1_gripper_server --network eth0 -l -r
    ```

3. Verify communication with the gripper service:
    ```bash
    ./test_dex1_1_gripper_server --network eth0 -l -r
    ```

---

### 2.2.3 ✅Testing 

Perform the following tests to ensure proper functionality:

- **Dex1 Gripper Test**:
  ```bash
  python test/endeffector/test_dex1.py
  ```

- **G1 Arm Test**:
  ```bash
  python test/arm/g1/test_g1_arm.py
  ```

- **Image Client Camera Test**:
  ```bash
  python test/camera/test_image_client_camera.py
  ```

- **G1 Datasets Replay**:
  ```bash
  # --repo-id     Your unique repo ID on Hugging Face Hub 
  # --robot_type     The type of the robot e.g., z1_dual_dex1_realsense, z1_realsense, g1_dex1, 
  
  python test/test_replay.py --repo-id unitreerobotics/G1_CameraPackaging_NewDataset --robot_type g1_dex1
  ```
---

## 2.2 🦿 Run Z1 

### 2.2.1 🦿 Z1 Setup
Clone and build the required repositories:

1. Download [z1_controller](https://github.com/unitreerobotics/z1_controller.git) and [z1_sdk](https://github.com/unitreerobotics/z1_sdk.git).

2. Build the repositories:
    ```bash
    mkdir build && cd build
    cmake .. && make -j
    ```

3. Copy the `unitree_arm_interface` library: [Modify according to your own path]
    ```bash
    cp z1_sdk/lib/unitree_arm_interface.cpython-310-x86_64-linux-gnu.so ./unitree_deploy/robot_devices/arm
    ```

4. Start the Z1 controller [Modify according to your own path]:
    ```bash
    cd z1_controller/build && ./z1_ctrl
    ```

---

### 2.2.2 Testing ✅

Run the following tests:

- **Realsense Camera Test**:
  ```bash
  python test/camera/test_realsense_camera.py # Modify the corresponding serial number according to your realsense
  ```

- **Z1 Arm Test**:
  ```bash
  python test/arm/z1/test_z1_arm.py
  ```

- **Z1 Environment Test**:
  ```bash
  python test/arm/z1/test_z1_env.py
  ```

- **Z1 Datasets Replay**:
  ```bash
  # --repo-id     Your unique repo ID on Hugging Face Hub 
  # --robot_type     The type of the robot e.g., z1_dual_dex1_realsense, z1_realsense, g1_dex1, 

  python test/test_replay.py --repo-id unitreerobotics/Z1_StackBox_Dataset --robot_type z1_realsense
  ```
---

## 2.3 🦿 Run Z1_Dual

### 2.3.1 🦿 Z1 Setup and Dex1 Setup
Clone and build the required repositories:

1. Download and compile the corresponding code according to the above z1 steps and Download the gripper program to start locally

2. [Modify the multi-machine control according to the document](https://support.unitree.com/home/zh/Z1_developer/sdk_operation)

3. [Download the modified z1_sdk_1 and then compile it](https://github.com/unitreerobotics/z1_sdk/tree/z1_dual), Copy the `unitree_arm_interface` library: [Modify according to your own path]
    ```bash
    cp z1_sdk/lib/unitree_arm_interface.cpython-310-x86_64-linux-gnu.so ./unitree_deploy/robot_devices/arm
    ```

4. Start the Z1 controller [Modify according to your own path]:
    ```bash
    cd z1_controller/builb && ./z1_ctrl
    cd z1_controller_1/builb && ./z1_ctrl
    ```
5. Start the gripper service, **ifconfig examines its own dds networkInterface**:
    ```
    sudo ./dex1_1_gripper_server --network eth0 -l -r
    ```
---

### 2.3.2 Testing ✅

Run the following tests:

- **Z1_Dual Arm Test**:
  ```bash
  python test/arm/z1/test_z1_arm_dual.py
  ```

- **Z1_Dual Datasets Replay**:
  ```bash
  # --repo-id     Your unique repo ID on Hugging Face Hub 
  # --robot_type     The type of the robot e.g., z1_dual_dex1_realsense, z1_realsense, g1_dex1, 

  python test/test_replay.py --repo-id unitreerobotics/Z1_Dual_Dex1_StackBox_Dataset_V2 --robot_type z1_dual_dex1_realsense
  ```
---


# 3.🧠 Inference and Deploy
1. [Modify the corresponding parameters according to your configuration](./unitree_deploy/robot/robot_configs.py)
2. Go back the **step-2 of Client Setup** under the [Inference and Deployment under Decision-Making Mode](https://github.com/unitreerobotics/unifolm-world-model-action/blob/main/README.md).

# 4.🏗️ Code structure

[If you want to add your own robot equipment, you can build it according to this document](./docs/GettingStarted.md)


# 5. 🤔 Troubleshooting

## `unitree_sdk2_python` install fails with "Could not locate cyclonedds"

The `cyclonedds` Python package builds from source and needs to find the CycloneDDS C library at build time.
If you see:

```
Could not locate cyclonedds. Try to set CYCLONEDDS_HOME or CMAKE_PREFIX_PATH
```

Set `CYCLONEDDS_HOME` to your CycloneDDS install prefix before running `pip install`:

```bash
# Unitree G1 ships with CycloneDDS pre-built in ~/cyclonedds_ws:
export CYCLONEDDS_HOME=~/cyclonedds_ws/install/cyclonedds

cd unitree_sdk2_python && pip install -e . && cd ..
```

You can persist this in your shell config so it applies to future installs:

```bash
echo 'export CYCLONEDDS_HOME=~/cyclonedds_ws/install/cyclonedds' >> ~/.bashrc
```

For assistance with other issues, contact the project maintainer or refer to the respective GitHub repository documentation. 📖

---

## `pinocchio` import fails: "cannot allocate memory in static TLS block" (aarch64)

On ARM64 (G1 robot board), pinocchio ships a bundled `libgomp.so.1` that cannot be loaded via normal `dlopen` after the TLS block is full:

```
ImportError: /home/unitree/miniconda3/envs/unitree_deploy/lib/python3.10/site-packages/
  pinocchio/../../libgomp.so.1: cannot allocate memory in static TLS block
```

Fix — preload the library before Python starts:

```bash
export LD_PRELOAD=/home/unitree/miniconda3/envs/unitree_deploy/lib/libgomp.so.1
```

Add this to `~/.bashrc` for persistence, or prefix every `python ...` invocation.
Note: the bare assignment `LD_PRELOAD=...` (without `export`) does **not** propagate to subprocess calls from Python.

---

## Brainco bridge connection refused

If the robot client prints `❌ Brainco bridge connect failed: [Errno 111] Connection refused`, the TCP bridge is not running. Make sure Terminal 2 (`brainco_bridge.py`) is running in the `g1brainco` env and that `stark_node` (Terminal 1) started successfully first.

---

## Arms jerking and whirring on startup

**Root cause: robot in Ready Mode (Locked Standing)**

If the robot is in Ready Mode (Locked Standing) when the robot client starts, the G1's built-in motion controller and `unitree_deploy`'s arm controller send competing control signals to the same joints. This produces constant jerking and whirring even before any inference action is executed.

**Fix: switch the robot to Development Mode before running the client.**

On the robot's App (or via the Unitree controller), select **Development Mode** before starting `robot_client.py`. Development Mode disables the built-in standing controller, giving `unitree_deploy` exclusive control over the joints. The jerking and whirring stop immediately.

---

**Secondary cause: competing init poses inside `unitree_deploy`**

Even in Development Mode, two sources inside the code can fight over the arm target position at startup:
1. `go_start()` in `G1_29_ArmController` drives to `G1ArmConfig.init_pose`
2. `env.step(INIT_POSE[...])` in `robot_client.py` sends a target position immediately after

These are aligned to the same Ready Mode values so they do not compete. If you modify either, make sure both `INIT_POSE['g1_brainco']` in `scripts/robot_client.py` and `ready_mode_pose` in `brainco_dual_arm_default_factory()` in `robot/robot_configs.py` use the same values.

The `G1_29_ArmController` also initialises `self.q_target` to `init_pose` (not zeros) so the background control loop holds the start position instead of driving back to the zero pose.

---

## Inference server normalizer dimension mismatch (26 vs 12)

The `g1_brainco` robot type uses a 26-DOF observation state (14 arm + 12 hand). If the trained model's dataset was recorded with a different state representation (e.g., 12 DOFs only), you will see:

```
RuntimeError: The size of tensor a (26) must match the size of tensor b (12) at non-singleton dimension 1
```

This means the model checkpoint and the `--dataset_name` passed to the inference server do not match the 26-DOF action space. Use a model checkpoint trained on `g1_brainco` data, or check `configs/inference/world_model_decision_making.yaml` and verify the `dataset_name` matches a dataset with 26-DOF state/action dimensions.


# 6. 🙏 Acknowledgement

This code builds upon following open-source code-bases. Please visit the URLs to see the respective LICENSES (If you find these projects valuable, it would be greatly appreciated if you could give them a star rating.):

1. https://github.com/huggingface/lerobot
2. https://github.com/unitreerobotics/unitree_sdk2_python
