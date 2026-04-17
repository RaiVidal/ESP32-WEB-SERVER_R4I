"""deployment/deploy.py
====================
Quantise a trained PyTorch model with esp-ppq and export it for ESP-DL.

This script is board-aware (via ``BOARD_PROFILES``):
- If the selected board profile has a ``ppq_target`` it runs INT8 PTQ and exports
    a ``.espdl`` file.
- If ``ppq_target`` is ``None``, ESP-DL export is skipped and an ONNX file is
    exported instead.

Pipeline
--------
    PyTorch checkpoint  →  [esp-ppq INT8 PTQ]  →  model.espdl
                                                       ↓
                                              Flash with ESP-IDF

Supports:
  - Any layer from `pytorch_layers/esp32_layers.py` (via --mode layer)
  - Full TinyCNN from a checkpoint              (via --mode model)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ── Make the repository root importable when called from any working dir ────
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pytorch_layers.esp32_layers import BOARD_PROFILES, LAYER_REGISTRY, TinyCNN


# ---------------------------------------------------------------------------
# esp-ppq imports (must be installed separately — see README/GUIDE)
# ---------------------------------------------------------------------------
try:
    from esp_ppq.api import espdl_quantize_torch

    HAS_ESPPQ = True
except ImportError:
    HAS_ESPPQ = False
    print("[WARN] esp-ppq not found. Install with: pip install esp-ppq")
    print("       Running in DRY-RUN mode — ONNX export only.\n")


# ---------------------------------------------------------------------------
# Calibration dataset helpers
# ---------------------------------------------------------------------------

def get_mnist_calib(n_samples: int = 256, img_size: int = 28, batch_size: int = 32) -> DataLoader:
    """Return an MNIST DataLoader for PTQ calibration."""

    tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    ds = datasets.MNIST("./data", train=False, download=True, transform=tf)
    ds = torch.utils.data.Subset(ds, range(n_samples))
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def get_custom_calib(
    data_dir: str,
    n_samples: int = 256,
    img_size: int = 32,
    in_channels: int = 3,
    batch_size: int = 32,
) -> DataLoader:
    """Return an ImageFolder (val split) DataLoader for PTQ calibration."""

    gray = [transforms.Grayscale(1)] if in_channels == 1 else []
    mean = (0.5,) * in_channels
    std = (0.5,) * in_channels
    tf = transforms.Compose(
        gray
        + [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    val_dir = os.path.join(data_dir, "val")
    ds = datasets.ImageFolder(val_dir, transform=tf)
    ds = torch.utils.data.Subset(ds, range(min(n_samples, len(ds))))
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def calib_dataloader_to_list(loader: DataLoader) -> list[torch.Tensor]:
    """Convert a DataLoader to a list of input tensors (esp-ppq calib format)."""

    return [imgs for imgs, _ in loader]


# ---------------------------------------------------------------------------
# Dummy calibration for single-layer mode (synthetic data)
# ---------------------------------------------------------------------------

def make_dummy_calib(input_shape: tuple[int, ...], n_batches: int = 16) -> list[torch.Tensor]:
    """Generate random calibration inputs for a given model input shape."""

    return [torch.randn(*input_shape) for _ in range(n_batches)]


# ---------------------------------------------------------------------------
# ONNX export (fallback / dry-run)
# ---------------------------------------------------------------------------

def export_onnx(model: nn.Module, dummy_input: torch.Tensor, path: str, opset: int = 18) -> str:
    """Export `model` to ONNX (opset 18) next to the `.espdl` path."""

    model.eval()
    onnx_path = path.replace(".espdl", ".onnx")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=True,
    )
    print(f"[INFO] ONNX model exported → {onnx_path}")
    return onnx_path


# ---------------------------------------------------------------------------
# Quantisation & export
# ---------------------------------------------------------------------------

def quantise_and_export(
    model: nn.Module,
    calib_data: list[torch.Tensor],
    output_path: str,
    input_shape: tuple[int, ...],
    target_platform: str,
    n_bits: int = 8,
) -> None:
    """Quantise `model` with esp-ppq and export to `.espdl`."""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    model.eval().cpu()

    dummy = torch.zeros(*input_shape)

    if not HAS_ESPPQ:
        print("[WARN] DRY-RUN: esp-ppq not installed — exporting ONNX only.")
        export_onnx(model, dummy, output_path)
        return

    print(f"\n[INFO] Quantising with esp-ppq → target={target_platform}, bits={n_bits}")
    print(f"[INFO] Calibration samples : {sum(int(t.shape[0]) for t in calib_data)}")

    espdl_quantize_torch(
        model=model,
        espdl_export_file=output_path,
        calib_dataloader=calib_data,
        calib_steps=len(calib_data),
        input_shape=list(input_shape),
        target=target_platform,
        num_of_bits=n_bits,
        export_test_values=True,  # embeds test I/O for board-side validation
        verbose=1,
    )

    print(f"\n[INFO] ✅ Quantised model exported → {output_path}")

    # Also dump ONNX for Netron inspection
    onnx_path = export_onnx(model, dummy, output_path)
    print(f"[INFO] Visualise with Netron: https://netron.app  (open {onnx_path})")


# ---------------------------------------------------------------------------
# Board helpers
# ---------------------------------------------------------------------------

def _resolve_board(cli_board: str | None, ckpt: dict) -> str:
    """Resolve the active board from CLI override, checkpoint, or default."""

    if cli_board is not None:
        return cli_board

    saved_board = ckpt.get("board")
    if isinstance(saved_board, str) and saved_board in BOARD_PROFILES:
        return saved_board

    saved_args = ckpt.get("args", {})
    if isinstance(saved_args, dict):
        maybe_board = saved_args.get("board")
        if isinstance(maybe_board, str) and maybe_board in BOARD_PROFILES:
            return maybe_board

    default_board = next(iter(BOARD_PROFILES.keys()))
    print(f"[WARN] No board found in checkpoint/CLI. Defaulting to {default_board}.")
    return default_board


def _artifact_tag(profile: dict) -> str:
    """Return a short tag used in output filenames."""

    ppq_target = profile.get("ppq_target")
    if ppq_target is None:
        return "c3"
    if ppq_target == "c":
        return "wroom"
    return "s3"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    p = argparse.ArgumentParser(description="Quantise and export an ESP32-family model with esp-ppq")
    p.add_argument(
        "--mode",
        choices=["layer", "model"],
        default="model",
        help="'layer' = single benchmark layer | 'model' = full TinyCNN",
    )

    p.add_argument(
        "--board",
        choices=list(BOARD_PROFILES.keys()),
        default=None,
        help="Board profile. If omitted, read from checkpoint (or default to the first profile in BOARD_PROFILES).",
    )

    # Layer mode
    p.add_argument(
        "--layer",
        choices=list(LAYER_REGISTRY.keys()),
        default="conv2d",
        help="[layer mode] Which layer to export",
    )

    # Model mode
    p.add_argument(
        "--checkpoint",
        default="./checkpoints/tiny_cnn_best.pth",
        help="[model mode] Path to .pth checkpoint",
    )
    p.add_argument("--dataset", choices=["mnist", "custom"], default="mnist")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--in-channels", type=int, default=1)
    p.add_argument("--n-classes", type=int, default=10)
    p.add_argument("--img-size", type=int, default=28)
    p.add_argument(
        "--base-ch",
        type=int,
        default=None,
        help="TinyCNN base channel width (overrides checkpoint/defaults).",
    )

    # Quantisation
    p.add_argument(
        "--bits",
        type=int,
        choices=[8, 16],
        default=8,
        help="Quantisation bit-width (8=fastest, 16=more accurate)",
    )
    p.add_argument("--calib-samples", type=int, default=256)

    # Output
    p.add_argument("--output-dir", default="./deployment_output")

    return p.parse_args()


def main() -> None:
    """CLI entry-point."""

    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── LAYER MODE ─────────────────────────────────────────────────────────
    if args.mode == "layer":
        board = args.board or next(iter(BOARD_PROFILES.keys()))
        profile = BOARD_PROFILES[board]
        tag = _artifact_tag(profile)

        entry = LAYER_REGISTRY[args.layer]
        model = entry["cls"](**entry["kwargs"])
        in_shape = entry["input_shape"]
        out_path = os.path.join(args.output_dir, f"{args.layer}_{tag}.espdl")

        print(f"[INFO] Layer mode: {args.layer}")
        print(f"[INFO] Board     : {board}")
        print(f"[INFO] Input     : {in_shape}")

        if profile["ppq_target"] is None:
            print(
                "[INFO] ESP32-C3 is not supported by ESP-DL. Exporting ONNX only for PC-side inspection. "
                "No .espdl file will be generated."
            )
            export_onnx(model, torch.zeros(*in_shape), out_path)
            return

        if profile["ppq_target"] == "c":
            print(
                "[INFO] WROOM-32 uses C operators (no SIMD). Inference will be ~4-6x slower than on S3. "
                "Consider base_ch=8 or smaller models."
            )

        calib = make_dummy_calib(in_shape, n_batches=32)
        quantise_and_export(
            model,
            calib,
            out_path,
            in_shape,
            target_platform=str(profile["ppq_target"]),
            n_bits=args.bits,
        )

    # ── MODEL MODE ─────────────────────────────────────────────────────────
    else:
        if not os.path.isfile(args.checkpoint):
            print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
            print("[ERROR] Run the trainer first, or specify --checkpoint path.")
            raise SystemExit(1)

        ckpt = torch.load(args.checkpoint, map_location="cpu")
        board = _resolve_board(args.board, ckpt)
        profile = BOARD_PROFILES[board]
        tag = _artifact_tag(profile)

        saved_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
        if not isinstance(saved_args, dict):
            saved_args = {}

        in_ch = int(saved_args.get("in_channels", args.in_channels))
        n_cls = int(saved_args.get("n_classes", args.n_classes))
        img_sz = int(saved_args.get("img_size", args.img_size))

        # base_ch resolution: CLI override > checkpoint args > board default
        if args.base_ch is not None:
            base_ch = int(args.base_ch)
        else:
            base_ch = saved_args.get("base_ch")
            if base_ch is None:
                base_ch = int(profile["default_base_ch"])
            else:
                base_ch = int(base_ch)

        if profile["ppq_target"] == "c" and args.base_ch is None and saved_args.get("base_ch") is None:
            # Explicitly match the requirement: WROOM default base_ch=8 if missing in checkpoint.
            base_ch = int(profile["default_base_ch"])

        model = TinyCNN(in_channels=in_ch, n_classes=n_cls, base_channels=base_ch)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        print(f"[INFO] Loaded checkpoint: {args.checkpoint}")
        print(f"[INFO] Board           : {board}")
        print(
            f"[INFO] val_acc={100 * float(ckpt.get('val_acc', 0)):.2f}%  "
            f"epoch={ckpt.get('epoch', '?')}"
        )
        print(f"[INFO] TinyCNN config  : in_ch={in_ch}  n_cls={n_cls}  img={img_sz}  base_ch={base_ch}")

        in_shape = (1, in_ch, img_sz, img_sz)
        out_path = os.path.join(args.output_dir, f"tiny_cnn_{tag}.espdl")

        if profile["ppq_target"] is None:
            print(
                "[INFO] ESP32-C3 is not supported by ESP-DL. Exporting ONNX only for PC-side inspection. "
                "No .espdl file will be generated."
            )
            export_onnx(model, torch.zeros(*in_shape), out_path)
            return

        if profile["ppq_target"] == "c":
            print(
                "[INFO] WROOM-32 uses C operators (no SIMD). Inference will be ~4-6x slower than on S3. "
                "Consider base_ch=8 or smaller models."
            )

        if args.dataset == "mnist":
            loader = get_mnist_calib(args.calib_samples, img_sz)
        else:
            loader = get_custom_calib(args.data_dir, args.calib_samples, img_sz, in_ch)
        calib = calib_dataloader_to_list(loader)

        quantise_and_export(
            model,
            calib,
            out_path,
            in_shape,
            target_platform=str(profile["ppq_target"]),
            n_bits=args.bits,
        )

    print("\n[INFO] Next step")
    print("[INFO] Copy the .espdl file into esp32_firmware/spiffs/")
    print("[INFO] then run: idf.py -p <PORT> flash monitor\n")


if __name__ == "__main__":
    main()
