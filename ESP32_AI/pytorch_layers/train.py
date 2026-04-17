"""pytorch_layers/train.py
=======================
Train TinyCNN (or any ESP32-family-compatible model) on MNIST (toy example)
or on a custom dataset via the --data-dir flag.

This script is board-aware:
- `--board` selects board profile defaults (notably `--base-ch`).
- For ESP32-C3, training is allowed for PC-side benchmarking, but the model is
  not deployable to hardware via ESP-DL.

Usage
-----
    # Toy example: MNIST
    python pytorch_layers/train.py --board <board> --dataset mnist --epochs 10

    # Custom dataset (ImageFolder layout)
    python pytorch_layers/train.py --board <board> --dataset custom \
        --data-dir /path/to/data --in-channels 3 --n-classes 5 --img-size 32 --epochs 20
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ── Make the repository root importable when called from any working dir ────
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pytorch_layers.esp32_layers import BOARD_PROFILES, TinyCNN


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def get_mnist_loaders(batch_size: int, data_root: str) -> tuple[DataLoader, DataLoader, int, int]:
    """Return MNIST train/val loaders and inferred (in_channels, n_classes)."""

    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_ds = datasets.MNIST(data_root, train=True, download=True, transform=tf)
    val_ds = datasets.MNIST(data_root, train=False, download=True, transform=tf)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    return train_loader, val_loader, 1, 10  # in_channels=1, n_classes=10


def get_custom_loaders(
    data_dir: str,
    batch_size: int,
    img_size: int,
    in_channels: int,
) -> tuple[DataLoader, DataLoader, int, int]:
    """Return ImageFolder train/val loaders and inferred (in_channels, n_classes).

    Expects ImageFolder layout:
        data_dir/
          train/
            class_a/  img1.jpg …
            class_b/  img2.jpg …
          val/
            class_a/  …
            class_b/  …
    """

    if in_channels == 1:
        mean, std = (0.5,), (0.5,)
        gray = [transforms.Grayscale(num_output_channels=1)]
    else:
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        gray = []

    train_tf = transforms.Compose(
        gray
        + [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    val_tf = transforms.Compose(
        gray
        + [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_tf)

    n_classes = len(train_ds.classes)
    print(f"[INFO] Custom dataset: {n_classes} classes → {train_ds.classes}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return train_loader, val_loader, in_channels, n_classes


# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    """Run one training epoch and return (avg_loss, accuracy)."""

    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        n += imgs.size(0)

        if (batch_idx + 1) % 50 == 0:
            print(
                f"[INFO] Epoch {epoch} [{batch_idx + 1}/{len(loader)}] "
                f"loss={total_loss / n:.4f}  acc={100 * correct / n:.2f}%"
            )

    return total_loss / n, correct / n


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run validation and return (avg_loss, accuracy)."""

    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        n += imgs.size(0)
    return total_loss / n, correct / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry-point."""

    parser = argparse.ArgumentParser(description="Train TinyCNN for ESP32 family boards")
    parser.add_argument(
        "--board",
        choices=list(BOARD_PROFILES.keys()),
        default=next(iter(BOARD_PROFILES.keys())),
        help="Board profile (controls defaults like --base-ch).",
    )
    parser.add_argument("--dataset", choices=["mnist", "custom"], default="mnist")
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Root dir for MNIST download or custom ImageFolder root",
    )
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--n-classes", type=int, default=10)
    parser.add_argument("--img-size", type=int, default=28)
    parser.add_argument(
        "--base-ch",
        type=int,
        default=None,
        help="Base channel width of TinyCNN (default depends on --board).",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-dir", default="./checkpoints")
    args = parser.parse_args()

    profile = BOARD_PROFILES[args.board]
    print(f"[INFO] Board profile: {args.board}")

    if profile["ppq_target"] is None:
        print("[WARN] ESP32-C3 does not support ESP-DL inference. Training anyway for")
        print("       PC-side benchmarking. Model will NOT be deployable to hardware.")

    if args.base_ch is None:
        args.base_ch = int(profile["default_base_ch"])

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ── Dataset ─────────────────────────────────────────────────────────────
    if args.dataset == "mnist":
        train_loader, val_loader, in_ch, n_cls = get_mnist_loaders(args.batch_size, args.data_dir)
    else:
        train_loader, val_loader, in_ch, n_cls = get_custom_loaders(
            args.data_dir,
            args.batch_size,
            args.img_size,
            args.in_channels,
        )
        n_cls = args.n_classes if args.n_classes != 10 else n_cls

    # ── Model ───────────────────────────────────────────────────────────────
    model = TinyCNN(in_channels=in_ch, n_classes=n_cls, base_channels=args.base_ch).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] TinyCNN — {total_params:,} parameters")

    # ── Training setup ──────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    best_ckpt = os.path.join(args.output_dir, "tiny_cnn_best.pth")

    print(f"\n[INFO] Training for {args.epochs} epochs …\n")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"[INFO] Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={tr_loss:.4f}  train_acc={100 * tr_acc:.2f}% | "
            f"val_loss={val_loss:.4f}  val_acc={100 * val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "args": vars(args),
                    "board": args.board,
                },
                best_ckpt,
            )
            print(f"[INFO] ✔ Saved best checkpoint → {best_ckpt}")

    # Also save the final weights
    final_ckpt = os.path.join(args.output_dir, "tiny_cnn_final.pth")
    torch.save(
        {
            "epoch": args.epochs,
            "state_dict": model.state_dict(),
            "val_acc": val_acc,
            "args": vars(args),
            "board": args.board,
        },
        final_ckpt,
    )
    print(f"\n[INFO] Final checkpoint → {final_ckpt}")
    print(f"[INFO] Best val accuracy : {100 * best_val_acc:.2f}%")
    print("\n[INFO] Training complete. Run deploy.py to export.\n")


if __name__ == "__main__":
    main()
