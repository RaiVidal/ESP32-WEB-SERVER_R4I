"""esp32_layers.py
================
PyTorch definitions for a small set of layers and reference models that map
cleanly to ESP-DL operators for the ESP32 family.

Notes
-----
- These are regular PyTorch ``nn.Module`` classes: train on CPU/GPU as usual.
- They are ONNX-exportable (opset 18), as required by esp-ppq.
- Runtime performance depends on the active board profile (SIMD vs non-SIMD).
    See ``BOARD_PROFILES`` for the single source of truth.

Usage
-----
        python esp32_layers.py --board <board> --layer conv2d
        python esp32_layers.py --board <board> --layer tiny_cnn
        python esp32_layers.py --board <board> --layer all
"""

import argparse
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Board profiles (single source of truth for board strings)
# ---------------------------------------------------------------------------

BOARD_PROFILES = {
    "esp32s3_mini":  {
        "idf_target": "esp32s3",
        "ppq_target": "esp32s3",
        "has_simd": True,
        "has_psram": False,
        "default_base_ch": 16,
    },
    "esp32_wroom32": {
        "idf_target": "esp32",
        "ppq_target": "c",
        "has_simd": False,
        "has_psram": False,
        "default_base_ch": 8,
    },
    "esp32c3_mini":  {
        "idf_target": "esp32c3",
        "ppq_target": None,
        "has_simd": False,
        "has_psram": False,
        "default_base_ch": 4,
    },
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def count_parameters(module: nn.Module) -> int:
    """Return the number of trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def benchmark(module: nn.Module, x: torch.Tensor, n_runs: int = 200) -> dict[str, float | int]:
    """Run `n_runs` forward passes and return timing statistics (ms)."""
    module.eval()
    with torch.no_grad():
        # warm-up
        for _ in range(10):
            _ = module(x)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = module(x)
        elapsed = (time.perf_counter() - t0) * 1000  # ms total
    return {
        "total_ms": round(elapsed, 3),
        "avg_ms": round(elapsed / n_runs, 4),
        "params": count_parameters(module),
    }


def print_benchmark(name: str, result: dict[str, float | int]) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Layer         : {name}")
    print(f"  Parameters    : {result['params']:,}")
    print(f"  Avg latency   : {result['avg_ms']} ms  (PC/CPU reference)")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# 1. Conv2D  — PIE dual-core, hardware-accelerated
# ---------------------------------------------------------------------------

class ESP32Conv2d(nn.Module):
    """
    Standard 2-D convolution.

    ESP-DL mapping : Conv  (ONNX opset 18)
    Acceleration   : board-dependent (PIE SIMD + dual-core on S3-class targets)
    Data layout    : NHWC on-chip (esp-ppq handles permutation)
    Quantisation   : INT8 per-channel (highest accuracy)

    Args
    ----
    in_channels, out_channels, kernel_size : standard PyTorch semantics.
    use_bn   : fuse BatchNorm into Conv weights before export (recommended).
    activation: any nn.Module activation.  Use ReLU/ReLU6 for best on-chip
                perf (no LUT needed).  Any other activation is transparently
                replaced by an 8-bit LUT by esp-ppq.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        bias: bool = False,
        use_bn: bool = True,
        activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias if not use_bn else False,
        )
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = activation if activation is not None else nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


# ---------------------------------------------------------------------------
# 2. DepthwiseConv2D  — PIE dual-core, hardware-accelerated
# ---------------------------------------------------------------------------

class ESP32DepthwiseConv2d(nn.Module):
    """
    Depthwise separable convolution (depthwise + pointwise).

    ESP-DL mapping : Conv (groups=C)  +  Conv (1×1)
    Acceleration   : board-dependent (PIE SIMD + dual-core on S3-class targets)
    Use-case       : MobileNet-style efficient feature extraction

    The two sub-convolutions are kept separate so esp-ppq can independently
    quantise the depthwise and pointwise kernels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_bn: bool = True,
        activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        act = activation if activation is not None else nn.ReLU(inplace=True)

        # Depthwise
        self.dw = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False,
        )
        self.dw_bn = nn.BatchNorm2d(in_channels) if use_bn else nn.Identity()
        self.dw_act = act

        # Pointwise
        self.pw = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.pw_act = type(act)()  # fresh instance

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw_act(self.dw_bn(self.dw(x)))
        x = self.pw_act(self.pw_bn(self.pw(x)))
        return x


# ---------------------------------------------------------------------------
# 3. Linear / GEMM  — PIE-accelerated (GEMM op)
# ---------------------------------------------------------------------------

class ESP32Linear(nn.Module):
    """
    Fully-connected layer.

    ESP-DL mapping : Gemm  (ONNX opset 18)
    Acceleration   : PIE SIMD (vectorised dot-product)
    Tip            : Keep output_features ≤ 512 for low single-inference latency
                     on resource-constrained ESP32 boards.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self.act = activation if activation is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.fc(x))


# ---------------------------------------------------------------------------
# 4. BatchNorm2d  — fused into Conv before deployment (no runtime cost)
# ---------------------------------------------------------------------------

class ESP32BatchNorm2d(nn.Module):
    """
    Standalone BN layer (useful for architecture experiments).

    Deployment note: esp-ppq folds BN into the preceding Conv weight/bias
    during export, so this layer has ZERO runtime cost on supported ESP32 targets.
    Only define standalone BN if it is NOT preceded by a conv (rare).
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)


# ---------------------------------------------------------------------------
# 5. Add (element-wise)  — PIE-accelerated
# ---------------------------------------------------------------------------

class ESP32Add(nn.Module):
    """
    Element-wise addition of two tensors (residual connections).

    ESP-DL mapping : Add  (ONNX opset 18)
    Acceleration   : PIE SIMD
    Note           : esp-ppq uses joint quantisation for Add inputs to
                     minimise rounding error.  Avoid large range differences
                     between the two branches.
    """

    def __init__(self, activation: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.act = activation if activation is not None else nn.Identity()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.act(x + y)


# ---------------------------------------------------------------------------
# 6. GlobalAveragePool  — PIE-accelerated
# ---------------------------------------------------------------------------

class ESP32GlobalAvgPool(nn.Module):
    """
    Global average pooling.

    ESP-DL mapping : GlobalAveragePool  (ONNX opset 18)
    Data layout    : NHWC on-chip (permuted by esp-ppq)
    Use-case       : Classification head, replaces Flatten+Linear in tiny CNNs
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Output shape: (N, C, 1, 1) — consistent with ONNX GlobalAveragePool
        return F.adaptive_avg_pool2d(x, 1)


# ---------------------------------------------------------------------------
# 7. MaxPool2d / AvgPool2d  — PIE-accelerated
# ---------------------------------------------------------------------------

class ESP32Pool2d(nn.Module):
    """
    Spatial pooling (max or average).

    ESP-DL mapping : MaxPool / AveragePool  (ONNX opset 18)
    Acceleration   : PIE SIMD, NHWC layout
    """

    def __init__(
        self,
        pool_type: str = "max",  # "max" | "avg"
        kernel_size: int = 2,
        stride: int = 2,
        padding: int = 0,
    ) -> None:
        super().__init__()
        assert pool_type in ("max", "avg"), "pool_type must be 'max' or 'avg'"
        if pool_type == "max":
            self.pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
        else:
            self.pool = nn.AvgPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


# ---------------------------------------------------------------------------
# 8. MobileNet-style Inverted Residual Block  — composed accelerated ops
# ---------------------------------------------------------------------------

class ESP32InvertedResidual(nn.Module):
    """
    MobileNetV2-style Inverted Residual (expand → depthwise → project).

    Every sub-operation maps to a PIE-accelerated ESP-DL operator.
    The residual Add is PIE-accelerated too.

    Args
    ----
    expand_ratio : channel expansion factor (6 in MobileNetV2).
    use_residual : only True when stride==1 and in==out.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expand_ratio: int = 6,
    ) -> None:
        super().__init__()
        hidden = int(in_channels * expand_ratio)
        self.use_residual = (stride == 1 and in_channels == out_channels)

        layers: list[nn.Module] = []
        # Expand (pointwise)
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_channels, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU6(inplace=True),
            ]
        # Depthwise
        layers += [
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1,
                      groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
        ]
        # Project (pointwise, linear — no activation)
        layers += [
            nn.Conv2d(hidden, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)


# ---------------------------------------------------------------------------
# 9. TinyCNN — complete deployable image-classifier for ESP32 family targets
# ---------------------------------------------------------------------------

class TinyCNN(nn.Module):
    """
    A tiny, ESP-DL-compatible CNN for image classification.

    Architecture (fits on supported ESP32 targets; scale down for smaller boards):
        Input  : (1, C_in, H, W)   e.g. (1, 1, 28, 28) for MNIST
                                        (1, 3, 32, 32) for CIFAR-10
        Block1 : Conv2d(C_in→16, 3×3) + BN + ReLU  →  H/2, W/2
        Block2 : DepthwiseSep(16→32, 3×3) + BN + ReLU  →  H/4, W/4
        Block3 : DepthwiseSep(32→64, 3×3) + BN + ReLU  →  H/8, W/8
        GAP    : GlobalAveragePool  →  (1, 64, 1, 1)
        Head   : Linear(64→n_classes)

    INT8 model size:  ~40-80 KB (well within 8 MB flash)
    Typical latency:  board-dependent (fastest on S3-class targets @ 240 MHz, INT8)

    Args
    ----
    in_channels  : 1 (greyscale) or 3 (RGB)
    n_classes    : number of output classes
    base_channels: channel width multiplier starting point (default 16)
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 10,
        base_channels: int = 16,
    ) -> None:
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4

        self.block1 = ESP32Conv2d(in_channels, c1, kernel_size=3, stride=2, padding=1)
        self.block2 = ESP32DepthwiseConv2d(c1, c2, stride=2)
        self.block3 = ESP32DepthwiseConv2d(c2, c3, stride=2)
        self.gap = ESP32GlobalAvgPool()
        self.head = ESP32Linear(c3, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)   # (N, c1, H/2, W/2)
        x = self.block2(x)   # (N, c2, H/4, W/4)
        x = self.block3(x)   # (N, c3, H/8, W/8)
        x = self.gap(x)      # (N, c3, 1,   1  )
        x = x.flatten(1)     # (N, c3)
        x = self.head(x)     # (N, n_classes)
        return x


# ---------------------------------------------------------------------------
# Registry — used by CLI and deployment script
# ---------------------------------------------------------------------------

LAYER_REGISTRY = {
    "conv2d": {
        "cls": ESP32Conv2d,
        "kwargs": {"in_channels": 3, "out_channels": 16},
        "input_shape": (1, 3, 32, 32),
        "description": "Standard Conv2d — PIE dual-core accelerated",
    },
    "depthwise": {
        "cls": ESP32DepthwiseConv2d,
        "kwargs": {"in_channels": 16, "out_channels": 32},
        "input_shape": (1, 16, 16, 16),
        "description": "Depthwise separable Conv — PIE dual-core accelerated",
    },
    "linear": {
        "cls": ESP32Linear,
        "kwargs": {"in_features": 256, "out_features": 64},
        "input_shape": (1, 256),
        "description": "Fully-connected / GEMM — PIE SIMD accelerated",
    },
    "batchnorm": {
        "cls": ESP32BatchNorm2d,
        "kwargs": {"num_features": 32},
        "input_shape": (1, 32, 16, 16),
        "description": "BatchNorm2d — fused into Conv at export (0 runtime cost)",
    },
    "pool": {
        "cls": ESP32Pool2d,
        "kwargs": {"pool_type": "max"},
        "input_shape": (1, 32, 16, 16),
        "description": "MaxPool2d — PIE SIMD accelerated",
    },
    "mobilenet_block": {
        "cls": ESP32InvertedResidual,
        "kwargs": {"in_channels": 32, "out_channels": 32, "expand_ratio": 6},
        "input_shape": (1, 32, 16, 16),
        "description": "MobileNetV2 Inverted Residual — all ops PIE-accelerated",
    },
    "tiny_cnn": {
        "cls": TinyCNN,
        "kwargs": {"in_channels": 1, "n_classes": 10},
        "input_shape": (1, 1, 28, 28),
        "description": "Full TinyCNN (MNIST/CIFAR-ready) — end-to-end ESP-DL model",
    },
}


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry-point for PC-side layer benchmarking."""

    parser = argparse.ArgumentParser(description="ESP32 family PyTorch layer definitions & benchmarks")
    parser.add_argument(
        "--board",
        choices=list(BOARD_PROFILES.keys()),
        default=next(iter(BOARD_PROFILES.keys())),
        help="Board profile to report against (affects printed warnings/notes).",
    )
    parser.add_argument(
        "--layer",
        choices=list(LAYER_REGISTRY.keys()) + ["all"],
        default="all",
        help="Which layer to instantiate and benchmark.",
    )
    parser.add_argument(
        "--runs", type=int, default=200,
        help="Number of forward-pass repetitions for benchmarking (default: 200).",
    )
    args = parser.parse_args()

    profile = BOARD_PROFILES[args.board]
    print("\nBoard         :", args.board)
    if profile["has_simd"]:
        print("SIMD          : YES (PIE)")
        print("est. speedup  : ~5x vs FP32 reference")
    else:
        print("SIMD          : NO")
    if profile["ppq_target"] is None:
        print("WARNING       : C3 has no ESP-DL support — PC benchmark only")

    targets = list(LAYER_REGISTRY.keys()) if args.layer == "all" else [args.layer]

    print("\nESP32 Layer Benchmark  (FP32, CPU reference)\n")

    for name in targets:
        entry = LAYER_REGISTRY[name]
        model = entry["cls"](**entry["kwargs"])
        x = torch.randn(*entry["input_shape"])
        result = benchmark(model, x, n_runs=args.runs)
        print_benchmark(f"{name}  [{entry['description']}]", result)


if __name__ == "__main__":
    main()