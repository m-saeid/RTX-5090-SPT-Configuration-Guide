# 🚀 RTX 5090 Configuration Guide for Superpoint Transformer (SPT)

Welcome to the definitive guide for configuring and running the **[Superpoint Transformer (SPT)](https://github.com/drprojects/superpoint_transformer)** on bleeding-edge NVIDIA hardware, specifically the **RTX 5090 (Blackwell architecture, sm_120)**.

As of early 2026, standard pre-compiled binaries for PyTorch and Torch Geometric (PyG) do not natively include the `sm_120` compute kernels. Furthermore, recent updates to NumPy (v2.x) and PyG (v2.6+) introduce breaking API changes and metaclass conflicts with older C++ point cloud extensions.

This repository provides a robust, tested installation pipeline to bypass compiler limitations, resolve dependency conflicts, and get SPT training and evaluation running smoothly on an RTX 5090.

---

## 🛑 The Core Challenges on Blackwell

If you attempt a standard installation on an RTX 5090, you will likely encounter one or more of the following fatal errors:

* `CUDA error: no kernel image is available for execution`
  Standard PyTorch cu124 or cu126 wheels lack `sm_120` binaries.

* `nvcc fatal: Unsupported gpu architecture 'compute_120'`
  The local CUDA toolkit's nvcc compiler does not yet recognize the Blackwell architecture flags during source compilation.

* `AttributeError: _ARRAY_API not found`
  or
  `ImportError: numpy.core.multiarray failed to import`
  SPT C++ dependencies (pycut-pursuit, pgeof) are compiled against the NumPy 1.x ABI, which breaks when PyTorch Nightly auto-installs NumPy 2.x.

* `TypeError: metaclass conflict: the metaclass of a derived class must be a (non-strict) subclass...`
  A conflict between SPT's custom Batch class and the latest PyG 2.6+ internal metaclasses.

This guide resolves all of these issues.

---

## 🛠️ Prerequisites

* **OS:** Linux (Ubuntu 20.04 / 22.04 / 24.04 recommended)
* **Driver:** NVIDIA Driver 590.xx or newer
* **Environment Manager:** Conda or Miniforge installed
* **Target Project:** You have already cloned the original SPT repository

---

## 🚀 Step-by-Step Installation

Navigate to the root directory of your cloned SPT repository and follow these steps.

You can run the commands manually or use the script below.

---

## 📦 The "Perfect" Install Script

Save the following as `install_5090.sh` in the root of your SPT project, make it executable, and run it:

```bash
chmod +x install_5090.sh
./install_5090.sh
```

```bash
#!/bin/bash

# Configuration
ENV_NAME="spt_5090"
PYTHON_VER="3.11"
PYTORCH_NIGHTLY_URL="https://download.pytorch.org/whl/nightly/cu128"

echo "_________________________________________________"
echo "🚀 RTX 5090 (Blackwell) Optimized Installer 🚀"
echo "_________________________________________________"

# 1. Create and Activate Environment
conda create --name $ENV_NAME python=$PYTHON_VER -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# 2. Install PyTorch Nightly
# Nightly is required for the latest CUDA 12.8 support compatible with 590+ drivers
pip install --pre torch torchvision torchaudio --index-url $PYTORCH_NIGHTLY_URL

# 3. Compile Environment Variables
# CRITICAL: We target 'sm_90' (Hopper) because nvcc might not recognize 'sm_120' yet.
# The RTX 5090 will run sm_90 compiled code via forward compatibility.
export TORCH_CUDA_ARCH_LIST="9.0"
export FORCE_CUDA=1
export MAX_JOBS=8 # Adjust based on your CPU cores

# 4. Fix ABI and Build Tools
pip install "numpy<2.0.0"
pip install "setuptools<70.0.0" wheel pytest-runner

# 5. Build PyG Dependencies from Source
# --no-build-isolation is REQUIRED so the subprocess compiler finds torch/numpy headers
echo "📐 Building Torch Geometric from source..."
pip install git+https://github.com/pyg-team/pyg-lib.git --no-build-isolation
pip install git+https://github.com/rusty1s/pytorch_scatter.git --no-build-isolation
pip install git+https://github.com/rusty1s/pytorch_cluster.git --no-build-isolation
pip install git+https://github.com/rusty1s/pytorch_spline_conv.git --no-build-isolation

# Pin torch-geometric to avoid metaclass conflicts with SPT's Data/Batch definitions
pip install "torch-geometric==2.5.3"

# 6. Build SPT Native Extensions
echo "🛠️ Compiling SPT Extensions (pgeof, cut-pursuit)..."
pip install --force-reinstall --no-build-isolation pgeof pycut-pursuit pygrid-graph torch-graph-components torch-ransac3d

# 7. Install Local FRNN
echo "🧩 Installing FRNN..."
if [ -d "src/dependencies/FRNN" ]; then
    export TORCH_CUDA_ARCH_LIST="9.0"

    cd src/dependencies/FRNN/external/prefix_sum
    rm -rf build/
    pip install . --no-build-isolation

    cd ../../
    rm -rf build/
    pip install . --no-build-isolation
    cd ../../../
else
    echo "⚠️ FRNN directory not found! Make sure you are in the SPT root."
fi

# 8. Install remaining standard dependencies
pip install matplotlib plotly jupyterlab ipywidgets torchmetrics==0.11.4 plyfile h5py colorhash seaborn numba pytorch-lightning pyrootutils hydra-core hydra-colorlog hydra-submitit-launcher "rich<=14.0" torch_tb_profiler wandb open3d gdown

echo "✅ Success! Your RTX 5090 environment is ready."
```

---

## 🧠 Why This Works (Technical Details)

For developers adapting this guide for other RTX 5090 projects:

### `TORCH_CUDA_ARCH_LIST="9.0"`

The most important trick. While the RTX 5090 is `sm_120`, passing `12.0` or `120` to nvcc fails if your local CUDA compiler is 12.6 or older. Compiling for `sm_90` allows builds to succeed, and NVIDIA forward compatibility allows the 5090 to execute Hopper-optimized PTX correctly.

### `--no-build-isolation`

When compiling libraries like `pyg-lib` or `prefix_sum`, pip normally creates an isolated environment. This hides installed torch and numpy headers from the compiler, causing:

```
ModuleNotFoundError: No module named 'torch'
```

Disabling isolation forces builds to use your active conda environment.

### `numpy<2.0.0`

PyTorch Nightly upgrades NumPy to 2.x. NumPy 2.0 introduced major ABI changes. Mixing NumPy 1.x and 2.x compiled extensions causes fatal `multiarray` import errors.

---

## 🎉 Verification

To verify your environment, run an evaluation job (after downloading weights and datasets per the main SPT repo):

```bash
python src/eval.py experiment=semantic/s3dis datamodule.fold=5 ckpt_path=/path/to/your/checkpoint.ckpt
```

If evaluation starts without CUDA kernel or metaclass errors, your RTX 5090 setup is complete.

---

## 🤝 Acknowledgments

* Original authors of the [Superpoint Transformer](https://github.com/drprojects/superpoint_transformer)
* PyTorch and PyG communities for bleeding-edge hardware support

