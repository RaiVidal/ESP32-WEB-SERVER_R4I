# ESP32 Multi-Board Neural Network Deployment Guide
### PyTorch → INT8 Quantisation → ESP-DL → ESP32 (S3 / WROOM-32)

This project supports multiple ESP32-family boards with a board-aware Python
pipeline and a single firmware codebase.

## 1. Chip Selection

**Recommended for real inference**: ESP32-S3.

Why:
- PIE SIMD (128-bit Q-registers) accelerates INT8 ops in ESP-DL.
- Dual-core scheduling improves Conv/Depthwise throughput.

## 2. Multi-Board Support

### 2.1 Board Constraints

| Board           | idf.py target | esp-ppq target | PIE SIMD | PSRAM     | Flash   | ESP-DL inference |
|-----------------|---------------|----------------|----------|-----------|---------|-----------------|
| esp32s3_mini    | esp32s3       | esp32s3        | YES      | None      | 4 MB    | FULL            |
| esp32_wroom32   | esp32         | c              | NO       | Optional  | 4/8 MB  | SLOW (C ops)    |
| esp32c3_mini    | esp32c3       | N/A            | NO       | None      | 4 MB    | NOT SUPPORTED   |

Key facts:
- ESP32-S3: Xtensa LX7, 240 MHz, PIE SIMD, 512 KB SRAM
- ESP32 (WROOM-32): Xtensa LX6, 240 MHz, no SIMD, 520 KB SRAM
- ESP32-C3: RISC-V, 160 MHz, no SIMD, 400 KB SRAM, ESP-DL inference not supported

### 2.2 Workflow (per board)

The board configuration files are generated (not committed) by:

```bash
python setup_board.py --board <BOARD>
```

This writes:
- `esp32_firmware/sdkconfig.defaults`
- `esp32_firmware/partitions.csv`

and prints the exact `idf.py` commands to run next.

#### esp32s3_mini

```bash
python setup_board.py --board esp32s3_mini

python pytorch_layers/train.py --board esp32s3_mini --dataset mnist --epochs 10

python deployment/deploy.py --mode model \
  --checkpoint checkpoints/tiny_cnn_best.pth \
  --dataset mnist --board esp32s3_mini --bits 8 --output-dir deployment_output

# Copy (and rename) the exported artifact to the firmware SPIFFS folder
mkdir -p esp32_firmware/spiffs
cp deployment_output/tiny_cnn_s3.espdl esp32_firmware/spiffs/tiny_cnn.espdl

cd esp32_firmware

# Source ESP-IDF (choose one)
. ../esp-idf/export.sh
# or: . ~/esp/esp-idf/export.sh

idf.py set-target esp32s3
idf.py update-dependencies

idf.py build
idf.py -p <PORT> flash monitor
```

#### esp32_wroom32 (4 MB / 8 MB)

```bash
python setup_board.py --board esp32_wroom32              # 4 MB default
python setup_board.py --board esp32_wroom32 --flash-size 8mb

python pytorch_layers/train.py --board esp32_wroom32 --dataset mnist --epochs 10

python deployment/deploy.py --mode model \
  --checkpoint checkpoints/tiny_cnn_best.pth \
  --dataset mnist --board esp32_wroom32 --bits 8 --output-dir deployment_output

mkdir -p esp32_firmware/spiffs
cp deployment_output/tiny_cnn_wroom.espdl esp32_firmware/spiffs/tiny_cnn.espdl

cd esp32_firmware

# Source ESP-IDF (choose one)
. ../esp-idf/export.sh
# or: . ~/esp/esp-idf/export.sh

idf.py set-target esp32
idf.py update-dependencies

idf.py build
idf.py -p <PORT> flash monitor
```

Notes:
- WROOM-32 runs ESP-DL with C operators (no SIMD). Expect ~4–6× slower inference.
- Prefer `--base-ch 8` or smaller models.

#### esp32c3_mini (PC benchmark + sysinfo firmware)

```bash
python setup_board.py --board esp32c3_mini

# Training is allowed for PC-side benchmarking; the model will not deploy.
python pytorch_layers/train.py --board esp32c3_mini --dataset mnist --epochs 10

# Export is ONNX-only on C3
python deployment/deploy.py --mode model \
  --checkpoint checkpoints/tiny_cnn_best.pth \
  --dataset mnist --board esp32c3_mini --output-dir deployment_output

cd esp32_firmware

# Source ESP-IDF (choose one)
. ../esp-idf/export.sh
# or: . ~/esp/esp-idf/export.sh

idf.py set-target esp32c3
idf.py update-dependencies

idf.py build
idf.py -p <PORT> flash monitor
```

C3 notes:
- Deploy produces `tiny_cnn_c3.onnx` only (no `.espdl`).
- Firmware prints `[SYSINFO]` and idles (no model loading / inference).

### 2.3 ESP-DL + IDF v5.4 note (MALLOC_CAP_SIMD)

ESP-IDF v5.4 does not define `MALLOC_CAP_SIMD`. Some older `esp-dl` releases
used it unconditionally in `dl_image_process.cpp`.

This repository vendors `esp-dl` as a local component at:

- `esp32_firmware/components/espressif__esp-dl` (v3.3.0)

and that copy already guards `MALLOC_CAP_SIMD`, so **no patch is required** for
the default repo setup.

If you delete the local override and instead use a managed `esp-dl` component
that still references `MALLOC_CAP_SIMD`, apply the patch below after
`idf.py update-dependencies`.

After `idf.py update-dependencies`, run:

```bash
sed -i 's/MALLOC_CAP_DEFAULT | MALLOC_CAP_SIMD/MALLOC_CAP_DEFAULT/g' \
    managed_components/espressif__esp-dl/vision/image/dl_image_process.cpp
```

This must be repeated every time you delete `managed_components/` and re-run
`idf.py update-dependencies`.

### 2.4 Quick Reference

| Board | setup | idf.py target | deploy output | firmware behavior |
|---|---|---|---|---|
| esp32s3_mini | `python setup_board.py --board esp32s3_mini` | `esp32s3` | `deployment_output/tiny_cnn_s3.espdl` | full inference (fast) |
| esp32_wroom32 | `python setup_board.py --board esp32_wroom32 [--flash-size 8mb]` | `esp32` | `deployment_output/tiny_cnn_wroom.espdl` | full inference (slow) |
| esp32c3_mini | `python setup_board.py --board esp32c3_mini` | `esp32c3` | `deployment_output/tiny_cnn_c3.onnx` | sysinfo only |
