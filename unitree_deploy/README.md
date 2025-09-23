# Unitree Deploy 

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
cd unitree_sdk2_python  && pip install -e . && cd ..
```

---
# 2. 🚀 Start 

**Tip: Keep all devices on the same LAN**

## 2.1 🤖 Run G1 with Dex_1 Gripper 

### 2.1.1 📷 Image Capture Service Setup (G1 Board) 

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

### 2.1.2 🤏 Dex_1 Gripper Service Setup (Development PC2)

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

### 2.1.2 ✅Testing 

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
    cd z1_controller/build
    ./z1_ctrl
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
  python test/test_replay.py --repo-id unitreerobotics/Z1_Dual_Dex1_StackBox_Dataset_V2 --robot_type z1_dual_dex1_realsense
  ```
---


# 3.🧠 Inference and Deploy
1. [Modify the corresponding parameters according to your configuration](./unitree_deploy/robot/robot_configs.py)
2. Go back the **step-2 of Client Setup** under the [Inference and Deployment under Decision-Making Mode](https://github.com/unitreerobotics/unifolm-world-model-action/blob/main/README.md).

# 4.🏗️ Code structure

[If you want to add your own robot equipment, you can build it according to this document](./docs/GettingStarted.md)


# 5. 🤔 Troubleshooting

For assistance, contact the project maintainer or refer to the respective GitHub repository documentation. 📖


# 6. 🙏 Acknowledgement

This code builds upon following open-source code-bases. Please visit the URLs to see the respective LICENSES (If you find these projects valuable, it would be greatly appreciated if you could give them a star rating.):

1. https://github.com/huggingface/lerobot
2. https://github.com/unitreerobotics/unitree_sdk2_python
