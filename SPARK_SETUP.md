# NVIDIA DGX Spark Setup & CUDA Fixes

This document outlines the fixes and optimizations applied to run the UnifoLM-WMA-0 model on an **NVIDIA DGX Spark** with **Blackwell (GB10)** GPUs.

## 1. CUDA & GPU Fixes

### Problem: Broken Driver/Kernel Mismatch
On the DGX Spark, `nvidia-smi` may fail with:
`NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.`

This occurs when the NVIDIA kernel module is not compiled for the current kernel (e.g., `6.14.0-1015-nvidia`).

**Fix:**
Ensure the driver is correctly installed and loaded for the active kernel.
```bash
sudo apt-get install --reinstall nvidia-driver-580
# Or rebuild via DKMS
sudo dkms autoinstall
```

### Problem: CPU-only PyTorch
The default environment may have a CPU-only build of PyTorch.

**Fix:**
Force reinstall PyTorch with CUDA 13.0 support (required for Blackwell GPUs):
```bash
pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

### Problem: Blackwell Compatibility Warning
PyTorch may show a warning:
`Found GPU0 NVIDIA GB10 which is of cuda capability 12.1. Minimum and Maximum cuda capability supported by this version of PyTorch is (8.0) - (12.0)`

**Optimization:**
Guide PyTorch to target the Blackwell architecture:
```bash
export TORCH_CUDA_ARCH_LIST="12.1"
```

## 2. Codebase Adaptations for Robustness

Several hardcoded `cuda` calls and strict loading requirements were modified to allow for device-agnostic execution and loading of base models:

- **Checkpoint Loading:** Updated `scripts/evaluation/world_model_interaction.py` to use `strict=False` in `load_state_dict`. This allows the base model (which lacks specific action/state heads) to load correctly.
- **Device Agnostic Encoders:** Updated all encoders in `src/unifolm_wma/modules/encoders/condition.py` to check for CUDA availability instead of defaulting to `"cuda"`.
- **Vision Backbone:** Updated `src/unifolm_wma/modules/vision/dinosiglip_vit.py` to move tensors to the device of the model parameters.
- **Sampler Fix:** Fixed `src/unifolm_wma/models/samplers/ddim.py` to use `self.model.device` in `register_buffer`.
- **Xformers Assertion:** Removed a hardcoded assertion in `src/unifolm_wma/modules/attention.py` that forced `xformers` usage, allowing a fallback to standard PyTorch attention.

## 3. Performance Bottlenecks & Missing Wheels

Performance is currently limited by the absence of specialized Blackwell (SM 12.1) binaries for libraries like `xformers`.

### Missing: `xformers` for CUDA 13.0
The standard `pip install xformers` fails for CUDA 13.0 because pre-built wheels are not yet available on the PyTorch index.

### Performance Impact
Without `xformers`, the model uses standard PyTorch attention. While functional, it is significantly slower than the optimized Memory Efficient Attention provided by `xformers`.

## 4. Building Missing Wheels Manually

To achieve peak performance on the DGX Spark, you must build `xformers` from source.

### Requirements
- CUDA Toolkit 13.0
- `nvcc` in your PATH
- `ninja` build system

### Build Instructions
```bash
conda activate unifolm-wma
pip install ninja
git clone https://github.com/facebookresearch/xformers.git
cd xformers
git submodule update --init --recursive
export TORCH_CUDA_ARCH_LIST="12.1"
pip install -v -e .
```
*Note: The build process can take 30-60 minutes depending on CPU performance.*

## 5. Running the Model

Always ensure the correct environment and architecture list are set:
```bash
export TORCH_CUDA_ARCH_LIST="12.1"
conda run -n unifolm-wma bash scripts/run_world_model_interaction.sh
```
