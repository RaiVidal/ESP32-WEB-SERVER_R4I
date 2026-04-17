# ESP32 Neural Network Deployment Guide (Multi-Board)
### PyTorch → INT8 Quantisation → ESP-DL → ESP32 (S3 / WROOM-32)

For board-specific workflows (S3 / WROOM-32 / C3 sysinfo-only), see `GUIDE.md`.

---

## Table of Contents
1. [Chip Selection: Why ESP32-S3](#1-chip-selection)
2. [Hardware Acceleration Explained](#2-hardware-acceleration)
3. [Project Structure](#3-project-structure)
4. [Prerequisites](#4-prerequisites)
5. [Python Environment Setup](#5-python-environment-setup)
6. [ESP-IDF Setup](#6-esp-idf-setup)
7. [Layer Definitions: what's available](#7-layer-definitions)
8. [Benchmarking Layers on PC](#8-benchmarking-layers-on-pc)
9. [Training TinyCNN (toy example: MNIST)](#9-training-tinycnn)
10. [Quantisation & Export with esp-ppq](#10-quantisation--export)
11. [Building the Firmware](#11-building-the-firmware)
12. [Flashing to Hardware](#12-flashing-to-hardware)
13. [Reading Results over UART](#13-reading-results-over-uart)
14. [Bring Your Own Model](#14-bring-your-own-model)
15. [Tuning & Troubleshooting](#15-tuning--troubleshooting)
16. [Expected Performance Numbers](#16-expected-performance-numbers)

---

## 1. Chip Selection

### Winner: ESP32-S3

| Feature | ESP32-WROOM-32 | ESP32-C3 | **ESP32-S3** |
|---|---|---|---|
| CPU cores | 2× Xtensa LX6 | 1× RISC-V | **2× Xtensa LX7** |
| Clock | 240 MHz | 160 MHz | **240 MHz** |
| SIMD / PIE | ❌ | ❌ | ✅ **128-bit Q-registers** |
| AI acceleration | ❌ | ❌ | ✅ **~4–6× speedup** |
| PSRAM support | Up to 4 MB | ❌ | **Up to 8 MB octal SPI** |
| USB OTG | ❌ | ❌ | ✅ |
| Dual-core DL scheduling | ❌ | N/A | ✅ Conv2D, DepthwiseConv |

**Why ESP32-S3 wins**: it is the only chip in this set that carries Espressif's PIE (Processor Instruction Extensions) — 128-bit SIMD vector registers that map 16 INT8 values into a single multiply-accumulate cycle. ESP-DL's quantised operators (Conv, DepthwiseConv, GEMM) are hand-written to target these registers. The measured speedup for INT8 models is **4.5–6.25× vs the original ESP32**, making it the only practical choice for on-device neural network inference.

> **Board recommendation**: `ESP32-S3-DevKitC-1` (official Espressif) or any S3 board with at least 8 MB flash and 8 MB octal PSRAM (labelled N16R8 or N8R8). The PSRAM is important — large INT8 weight tensors may not fit in the 512 KB internal SRAM alone.

---

## 2. Hardware Acceleration Explained

### PIE — Processor Instruction Extensions

The ESP32-S3's Xtensa LX7 core includes dedicated SIMD Q-registers (128-bit wide). In INT8 mode, each Q-register holds **16 INT8 values** simultaneously.

```
FP32 (no SIMD):  1 multiply per cycle
INT8 (PIE SIMD): 16 multiplies per cycle  →  ~16× theoretical throughput
```

ESP-DL further compounds this with **dual-core scheduling**: Conv2D and DepthwiseConv2D split their output channels across both cores automatically, yielding close to a 2× additional speedup for large layers.

### What ESP-DL Accelerates (Hardware)

| Layer | PIE SIMD | Dual-core |
|---|---|---|
| Conv2D | ✅ | ✅ |
| DepthwiseConv2D | ✅ | ✅ |
| GEMM / Linear | ✅ | ❌ |
| Add | ✅ | ❌ |
| Mul | ✅ | ❌ |
| GlobalAveragePool | ✅ (NHWC) | ❌ |
| MaxPool / AvgPool | ✅ (NHWC) | ❌ |
| Any activation | 8-bit LUT | ❌ |
| BatchNorm | fused into Conv | N/A |

### Quantisation Strategy

ESP-DL uses **symmetric power-of-two quantisation**:

```
float_value ≈ int8_value × 2^exponent
```

This lets the hardware replace division with a bit-shift — no floating-point unit required at inference time. The esp-ppq tool handles all calibration and exponent computation during the export step.

---

## 3. Project Structure

```
./
├── setup_board.py             # Generate sdkconfig.defaults + partitions.csv per board
├── train.py                   # Compatibility wrapper (calls pytorch_layers/train.py)
├── deploy.py                  # Compatibility wrapper (calls deployment/deploy.py)
├── pytorch_layers/
│   ├── __init__.py
│   ├── esp32_layers.py        # Layer definitions + BOARD_PROFILES + CLI benchmark
│   └── train.py               # Training script (MNIST + custom datasets, --board)
├── deployment/
│   ├── __init__.py
│   └── deploy.py              # Quantise + export to .espdl / .onnx (--board)
├── esp32_firmware/
│   ├── CMakeLists.txt         # Top-level ESP-IDF project file
│   ├── partitions.csv         # Generated partition table (setup_board.py)
│   ├── sdkconfig.defaults     # Generated build defaults (setup_board.py)
│   ├── spiffs/                # Files flashed to the SPIFFS partition
│   │   └── tiny_cnn.espdl     # Model artifact (S3/WROOM only)
│   ├── components/
│   │   └── espressif__esp-dl/ # Vendored esp-dl component (v3.3.0)
│   └── main/
│       ├── main.cpp           # Firmware: loads model, runs inference, prints latency
│       ├── CMakeLists.txt     # Component registration
│       ├── idf_component.yml  # Component manager deps (esp-dl, etc.)
│       └── Kconfig.projbuild  # menuconfig options
├── checkpoints/               # Training outputs (.pth)
├── deployment_output/         # Export outputs (.espdl/.onnx)
├── data/                      # Datasets (MNIST downloads here by default)
├── esp-idf/                   # Optional: local ESP-IDF checkout
├── pyproject.toml             # Python dependencies (PEP 621)
├── README.md                  # Overview
└── GUIDE.md                   # Multi-board workflows
```

---

## 4. Prerequisites

### Hardware
- ESP32-S3 board (recommended for real inference)
- ESP32-WROOM-32 board (supported, slower inference)
- ESP32-C3 board (sysinfo-only firmware; no ESP-DL inference)
- USB-C cable (for programming + UART monitor)
- No external components required for this demo

### Software (Host)
- **OS**: Linux, macOS, or Windows (WSL2 recommended on Windows)
- **Python**: >= 3.10
- **Git**: any recent version
- **ESP-IDF**: v5.4 or higher

---

## 5. Python Environment Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install Python dependencies
pip install -e .
```

> **Note**: If your virtualenv was created without `pip`, run: `python -m ensurepip --upgrade`.

> **Note**: `esp-ppq` requires a C++ compiler on PATH (GCC/Clang on Linux/macOS, MSVC on Windows). On Ubuntu: `sudo apt install build-essential ninja-build`.

### Verify the installation

```bash
python -c "import torch; print(torch.__version__)"
python -c "from esp_ppq.api import espdl_quantize_torch; print('esp-ppq OK')"
```

---

## 6. ESP-IDF Setup

### 6.1 Install ESP-IDF v5.4+

If you already have ESP-IDF installed, you only need to source `export.sh`.

This repository may also include an `esp-idf/` checkout — in that case you can source it directly:

```bash
. ./esp-idf/export.sh
```

```bash
# Install prerequisites (Ubuntu/Debian)
sudo apt update
sudo apt install -y git wget flex bison gperf python3 python3-pip \
    python3-venv cmake ninja-build ccache libffi-dev libssl-dev \
    dfu-util libusb-1.0-0

# Clone and install ESP-IDF
mkdir -p ~/esp
cd ~/esp
git clone --recursive https://github.com/espressif/esp-idf.git
cd esp-idf
git checkout v5.4
./install.sh esp32s3 esp32 esp32c3

# Source the environment (add to ~/.bashrc for permanent effect)
. ~/esp/esp-idf/export.sh
```

### 6.2 Verify

```bash
idf.py --version
# Expected: ESP-IDF v5.4.x
```

---

## 7. Layer Definitions

All layers are defined in `pytorch_layers/esp32_layers.py`. Each maps 1:1 to an ESP-DL operator:

| Python class | ESP-DL operator | ONNX op | Accelerated |
|---|---|---|---|
| `ESP32Conv2d` | Conv2D | Conv | PIE + dual-core |
| `ESP32DepthwiseConv2d` | DepthwiseConv2D | Conv (groups=C) | PIE + dual-core |
| `ESP32Linear` | GEMM | Gemm | PIE SIMD |
| `ESP32BatchNorm2d` | *(fused)* | *(removed at export)* | zero cost |
| `ESP32Add` | Add | Add | PIE SIMD |
| `ESP32GlobalAvgPool` | GlobalAveragePool | GlobalAveragePool | PIE SIMD |
| `ESP32Pool2d` | MaxPool / AvgPool | MaxPool / AveragePool | PIE SIMD |
| `ESP32InvertedResidual` | Conv ×3 + Add | *(composite)* | all PIE |
| `TinyCNN` | all of the above | *(full model)* | end-to-end |

---

## 8. Benchmarking Layers on PC

This gives you a PC-relative FP32 reference. On-chip INT8 will be significantly faster.

```bash
cd pytorch_layers

# Benchmark a specific layer
python esp32_layers.py --board esp32s3_mini --layer conv2d
python esp32_layers.py --board esp32s3_mini --layer depthwise
python esp32_layers.py --board esp32s3_mini --layer linear
python esp32_layers.py --board esp32s3_mini --layer mobilenet_block
python esp32_layers.py --board esp32s3_mini --layer tiny_cnn

# Benchmark ALL layers at once
python esp32_layers.py --board esp32s3_mini --layer all --runs 500
```

Sample output:
```
Board         : esp32s3_mini
SIMD          : YES (PIE)
est. speedup  : ~5x vs FP32 reference

============================================================
  Layer         : conv2d  [Standard Conv2d — PIE dual-core accelerated]
  Parameters    : 448
  Avg latency   : 0.0321 ms  (PC/CPU reference)
============================================================
```

---

## 9. Training TinyCNN

### 9.1 MNIST (toy example — runs in ~5 minutes)

```bash
python train.py \
    --board esp32s3_mini \
    --dataset mnist \
    --epochs 10 \
    --batch-size 128 \
    --lr 1e-3 \
    --output-dir ./checkpoints
```

Expected final accuracy: **~99%** on MNIST validation set.

### 9.2 Custom Dataset

Organise your data as an `ImageFolder`:

```
my_dataset/
├── train/
│   ├── cat/   img001.jpg  img002.jpg …
│   └── dog/   img003.jpg  img004.jpg …
└── val/
    ├── cat/   …
    └── dog/   …
```

Then train:

```bash
python train.py \
    --board esp32s3_mini \
    --dataset custom \
    --data-dir ./my_dataset \
    --in-channels 3 \
    --n-classes 2 \
    --img-size 32 \
    --base-ch 16 \
    --epochs 30 \
    --output-dir ./checkpoints
```

> **Tip**: Keep `--img-size` ≤ 64 for real-time inference on ESP32-S3. Images are scaled down during preprocessing.

---

## 10. Quantisation & Export

This step converts the trained FP32 model to INT8 `.espdl` format using post-training quantisation (PTQ).

### 10.1 Export a single layer (benchmark mode)

```bash
# Export conv2d layer to .espdl
python deploy.py --mode layer --layer conv2d --board esp32s3_mini --bits 8

# Export all layers one by one
for layer in conv2d depthwise linear batchnorm pool mobilenet_block tiny_cnn; do
    python deploy.py --mode layer --layer $layer --board esp32s3_mini --bits 8
done
```

### 10.2 Export full TinyCNN model

```bash
# MNIST model
python deploy.py \
    --mode model \
    --checkpoint ./checkpoints/tiny_cnn_best.pth \
    --dataset mnist \
    --board esp32s3_mini \
    --bits 8 \
    --calib-samples 256 \
    --output-dir ./deployment_output

# Custom model
python deploy.py \
    --mode model \
    --checkpoint ./checkpoints/tiny_cnn_best.pth \
    --dataset custom \
    --data-dir ./my_dataset \
    --board esp32s3_mini \
    --in-channels 3 \
    --img-size 32 \
    --bits 8 \
    --output-dir ./deployment_output
```

### 10.3 Output files

After running deploy.py, your `deployment_output/` folder contains board-tagged artifacts.

Examples:

```
deployment_output/
├── tiny_cnn_s3.espdl    ← ESP32-S3 (fast, PIE SIMD)
└── tiny_cnn_s3.onnx     ← Inspect this in Netron (https://netron.app)

# WROOM-32:
#   tiny_cnn_wroom.espdl + tiny_cnn_wroom.onnx
# C3:
#   tiny_cnn_c3.onnx only (no .espdl)
```

The `.espdl` file contains:
- Quantised INT8 weights (power-of-two exponents)
- Model graph (FlatBuffers, zero-copy)
- Test input/output values (for board-side validation)

### 10.4 Inspect the model (optional)

Open the exported `.onnx` (e.g. `tiny_cnn_s3.onnx`) at [https://netron.app](https://netron.app) to visually verify every layer before flashing.

---

## 11. Building the Firmware

### 11.1 Copy the model to SPIFFS data folder

```bash
mkdir -p esp32_firmware/spiffs
cp deployment_output/tiny_cnn_s3.espdl esp32_firmware/spiffs/tiny_cnn.espdl

# WROOM-32: copy tiny_cnn_wroom.espdl instead
# C3: no .espdl is produced (sysinfo-only firmware)
```

### 11.2 Set target and fetch components

```bash
python setup_board.py --board esp32s3_mini

cd esp32_firmware

# Source ESP-IDF (choose one)
. ../esp-idf/export.sh
# or: . ~/esp/esp-idf/export.sh

# Set target chip
idf.py set-target esp32s3

# Download managed component dependencies
idf.py update-dependencies
```

This repo vendors `esp-dl` as a local component at `esp32_firmware/components/espressif__esp-dl` (v3.3.0)
to avoid IDF v5.4 compatibility issues.

If you remove the local override and instead use a managed `esp-dl` component (especially <= 3.2.x),
see `GUIDE.md` for the optional `MALLOC_CAP_SIMD` compatibility patch.

### 11.3 (Optional) Customise via menuconfig

```bash
idf.py menuconfig
# Navigate to: ESP32 Neural Network Config
#   - MODEL_PATH       → file path in SPIFFS
#   - INTERNAL_RAM_MB  → SRAM budget for model ops (1–4)
#   - N_RUNS           → benchmark repetition count
#   - BOARD_TARGET      → board profile string (optional)
```

### 11.4 Build

```bash
idf.py build
```

A successful build prints:
```
Project build complete.
```

---

## 12. Flashing to Hardware

### 12.1 Find the serial port

```bash
# Linux
ls /dev/ttyUSB* /dev/ttyACM*

# macOS
ls /dev/cu.*
```

### 12.2 Flash firmware + SPIFFS in one command

```bash
# Replace /dev/ttyUSB0 with your actual port
idf.py -p /dev/ttyUSB0 flash monitor
```

> **ESP32-S3 with native USB**: if your board uses the native USB port (not a USB-UART bridge), use `--before default_reset --after hard_reset` or put the board into download mode manually by holding BOOT, pressing RESET, releasing BOOT.

### 12.3 Flash SPIFFS partition (model file) separately

If you change only the model without rebuilding the firmware:

```bash
# Build SPIFFS image
idf.py build  # only needed once to generate spiffs.bin

# Flash only the SPIFFS partition
parttool.py -p /dev/ttyUSB0 write_partition \
    --partition-name=spiffs \
    --input build/spiffs.bin
```

---

## 13. Reading Results over UART

After flashing, the UART monitor (115200 baud) prints:

```
I (312)  NN_ESP32S3: ESP32 Neural Network Inference Demo
I (318)  NN_ESP32S3: Model path : /spiffs/tiny_cnn.espdl
I (320)  NN_ESP32S3: Internal RAM for model ops: 2 MB
I (410)  NN_ESP32S3: SPIFFS: 49152 / 1048576 bytes used
I (415)  NN_ESP32S3: [MEM] after SPIFFS mount: internal=287432 B  PSRAM=not present
I (610)  NN_ESP32S3: Loading model from /spiffs/tiny_cnn.espdl …
I (622)  NN_ESP32S3: Model file size: 43892 bytes
I (780)  NN_ESP32S3: Model inputs (1):
I (780)  NN_ESP32S3:   'input'  shape=[1,28,28,1]  dtype=2
I (781)  NN_ESP32S3: Model outputs (1):
I (781)  NN_ESP32S3:   'output'  shape=[1,10]  dtype=2
I (900)  NN_ESP32S3: Running single inference …
I (901)  NN_ESP32S3: Predicted class : 0  (score=3.1250)
I (1200) NN_ESP32S3: ──────────────────────────────────────
I (1200) NN_ESP32S3: Benchmark (50 runs):
I (1201) NN_ESP32S3:   Total   : 287.50 ms
I (1201) NN_ESP32S3:   Average : 5.750 ms / inference
I (1201) NN_ESP32S3:   FPS     : 173.9
I (1201) NN_ESP32S3: ──────────────────────────────────────
```

To exit the monitor: `Ctrl+]`

---

## 14. Bring Your Own Model

### Requirements checklist

1. **All operators in the model must be in the ESP-DL supported list**  
   Check [operator_support_state.md](https://github.com/espressif/esp-dl/blob/master/operator_support_state.md) before designing your architecture.

2. **No dynamic shapes** — ESP-DL requires static shapes throughout.

3. **Batch size = 1** — ESP-DL does not support batched inference.

4. **ONNX opset 18** — export with `torch.onnx.export(..., opset_version=18)`.

5. **Model size** — target ≤ 1 MB INT8 weights for reliable SRAM + PSRAM fit.

### Workflow for a custom PyTorch model

```python
# my_model.py — your model must only use ESP-DL-supported ops
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Use only: Conv2d, BatchNorm2d, ReLU/ReLU6,
        #           Linear, Add (+), MaxPool2d, AvgPool2d
        ...

    def forward(self, x):
        ...
```

Then deploy exactly like TinyCNN:

```bash
python deploy.py \
    --mode model \
    --checkpoint ./my_checkpoint.pth \
    --dataset custom \
    --data-dir ./my_data \
    --bits 8 \
    --output-dir ./deployment_output
```

And copy the `.espdl` into `esp32_firmware/spiffs/`, rebuild, and reflash.

---

## 15. Tuning & Troubleshooting

### Accuracy drop after quantisation

- **Increase calibration samples**: `--calib-samples 512` or more.
- **Switch to INT16**: `--bits 16` (slower but ~2× better accuracy).
- **Per-channel quantisation** (default INT8 in esp-ppq) is already used for Conv — ensure you don't override it.
- **Check outlier activations**: add `BatchNorm2d` after every Conv. Unconstrained activation ranges destroy INT8 accuracy.

### Allocation failure / model won't load

```
E (xxx) NN_ESP32S3: Failed to allocate model
```

- Decrease `CONFIG_INTERNAL_RAM_MB` (frees internal heap). On non-PSRAM boards, keep this at `1`.
- Check your board has PSRAM (labelled R8 or R16 in the part number).
- Shrink the model: reduce `--base-ch` from 16 to 8 in train.py.

### SPIFFS model not found

```
E (xxx) NN_ESP32S3: Model file not found: /spiffs/tiny_cnn.espdl
```

- Verify `esp32_firmware/spiffs/tiny_cnn.espdl` exists before building.
- Make sure `CONFIG_MODEL_PATH` in menuconfig matches the filename you copied.
- Re-run `idf.py build` after adding the file.

### esp-ppq ImportError

```
ModuleNotFoundError: No module named 'esp_ppq'
```

```bash
pip install esp-ppq
# or from source:
pip install git+https://github.com/espressif/esp-ppq.git
```

### Build error: component esp-dl not found

```bash
idf.py update-dependencies
```

This fetches managed dependencies and resolves the `esp-dl` component. In this repo, `esp-dl` is
vendored as a local component under `esp32_firmware/components/espressif__esp-dl`.

### ESP32-S3 won't enter download mode

Hold **BOOT (GPIO0)** → press **RESET (EN)** → release **BOOT** → then run `idf.py flash`.

---

## 16. Expected Performance Numbers

Measured on ESP32-S3 @ 240 MHz, ESP-DL v3.x, INT8:

| Model / Layer | Input | Params | INT8 Size | Avg Latency | FPS |
|---|---|---|---|---|---|
| TinyCNN (base=16) | 28×28×1 | ~18 K | ~25 KB | ~6 ms | ~170 |
| TinyCNN (base=32) | 32×32×3 | ~70 K | ~80 KB | ~15 ms | ~65 |
| Single Conv2d (3→16, 3×3) | 32×32 | 448 | — | ~1.5 ms | — |
| Single DepthwiseSep (16→32) | 16×16 | ~600 | — | ~0.8 ms | — |
| Single Linear (256→64) | — | 16 K | — | ~0.3 ms | — |

> Latency scales roughly linearly with spatial resolution and channel count.  
> Dual-core scheduling halves Conv2D latency for large feature maps.

---

## Quick Reference: Full Workflow

```bash
# ─── One-time setup ───────────────────────────────────────────────────────
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# Source ESP-IDF (choose one)
. ./esp-idf/export.sh          # repo-local checkout
# or: . ~/esp/esp-idf/export.sh

# ─── Select board (generates sdkconfig.defaults + partitions.csv) ──────────
python setup_board.py --board esp32s3_mini

# ─── Train ────────────────────────────────────────────────────────────────
python train.py --board esp32s3_mini --dataset mnist --epochs 10

# ─── Quantise & export ────────────────────────────────────────────────────
python deploy.py --mode model \
    --board esp32s3_mini \
    --checkpoint ./checkpoints/tiny_cnn_best.pth \
    --dataset mnist --bits 8 \
    --output-dir ./deployment_output

# ─── Copy model to firmware ───────────────────────────────────────────────
mkdir -p esp32_firmware/spiffs
cp deployment_output/tiny_cnn_s3.espdl esp32_firmware/spiffs/tiny_cnn.espdl

# ─── Build & flash ────────────────────────────────────────────────────────
cd esp32_firmware
idf.py set-target esp32s3
idf.py update-dependencies

idf.py build
idf.py -p /dev/ttyUSB0 flash monitor
```

---