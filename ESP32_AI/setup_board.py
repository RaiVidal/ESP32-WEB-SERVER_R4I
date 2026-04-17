"""setup_board.py
================
Generate ESP-IDF config files for a selected ESP32 board.

This script writes:
- esp32_firmware/sdkconfig.defaults
- esp32_firmware/partitions.csv

It does NOT run `idf.py`. It only prints the next commands to run.

Usage
-----
    python setup_board.py --board esp32s3_mini
    python setup_board.py --board esp32_wroom32
    python setup_board.py --board esp32c3_mini
    python setup_board.py --board esp32_wroom32 --flash-size 8mb
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


BoardName = Literal["esp32s3_mini", "esp32_wroom32", "esp32c3_mini"]
FlashSize = Literal["4mb", "8mb"]


@dataclass(frozen=True)
class BoardConfig:
    """Board-specific configuration for generated ESP-IDF files."""

    board: BoardName
    idf_target: str


BOARD_CONFIGS: dict[BoardName, BoardConfig] = {
    "esp32s3_mini": BoardConfig(board="esp32s3_mini", idf_target="esp32s3"),
    "esp32_wroom32": BoardConfig(board="esp32_wroom32", idf_target="esp32"),
    "esp32c3_mini": BoardConfig(board="esp32c3_mini", idf_target="esp32c3"),
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate esp32_firmware/sdkconfig.defaults and esp32_firmware/partitions.csv "
            "for a selected board. Does not run idf.py."
        )
    )
    parser.add_argument(
        "--board",
        required=True,
        choices=list(BOARD_CONFIGS.keys()),
        help="Target board profile.",
    )
    parser.add_argument(
        "--flash-size",
        choices=["4mb", "8mb"],
        default=None,
        help="Flash size override (only valid with --board esp32_wroom32).",
    )
    return parser.parse_args()


def _render_sdkconfig_defaults(board: BoardName, flash_size: FlashSize | None) -> str:
    """Return sdkconfig.defaults content for the selected board."""

    if board == "esp32_wroom32":
        selected_flash = flash_size or "4mb"
        flash_line = (
            'CONFIG_ESPTOOLPY_FLASHSIZE_8MB=y' if selected_flash == "8mb" else 'CONFIG_ESPTOOLPY_FLASHSIZE_4MB=y'
        )
        return "\n".join(
            [
                'CONFIG_IDF_TARGET="esp32"',
                'CONFIG_ESP32_DEFAULT_CPU_FREQ_MHZ=240',
                'CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ=240',
                flash_line,
                'CONFIG_PARTITION_TABLE_CUSTOM=y',
                'CONFIG_PARTITION_TABLE_CUSTOM_FILENAME="partitions.csv"',
                'CONFIG_SPIFFS_PAGE_SIZE=256',
                'CONFIG_ESP_MAIN_TASK_STACK_SIZE=8192',
                'CONFIG_LOG_DEFAULT_LEVEL_INFO=y',
                'CONFIG_LOG_MAXIMUM_LEVEL_DEBUG=y',
                'CONFIG_COMPILER_OPTIMIZATION_PERF=y',
                "",
            ]
        )

    if board == "esp32s3_mini":
        return "\n".join(
            [
                'CONFIG_IDF_TARGET="esp32s3"',
                'CONFIG_ESP32S3_DEFAULT_CPU_FREQ_MHZ=240',
                'CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ=240',
                'CONFIG_ESP32S3_DATA_CACHE_LINE_64B=y',
                'CONFIG_ESPTOOLPY_FLASHSIZE_4MB=y',
                'CONFIG_PARTITION_TABLE_CUSTOM=y',
                'CONFIG_PARTITION_TABLE_CUSTOM_FILENAME="partitions.csv"',
                'CONFIG_SPIFFS_PAGE_SIZE=256',
                'CONFIG_ESP_MAIN_TASK_STACK_SIZE=16384',
                'CONFIG_LOG_DEFAULT_LEVEL_INFO=y',
                'CONFIG_LOG_MAXIMUM_LEVEL_DEBUG=y',
                'CONFIG_COMPILER_OPTIMIZATION_PERF=y',
                "",
            ]
        )

    # esp32c3_mini
    return "\n".join(
        [
            'CONFIG_IDF_TARGET="esp32c3"',
            'CONFIG_ESP32C3_DEFAULT_CPU_FREQ_MHZ=160',
            'CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ=160',
            'CONFIG_ESPTOOLPY_FLASHSIZE_4MB=y',
            'CONFIG_PARTITION_TABLE_CUSTOM=y',
            'CONFIG_PARTITION_TABLE_CUSTOM_FILENAME="partitions.csv"',
            'CONFIG_SPIFFS_PAGE_SIZE=256',
            'CONFIG_ESP_MAIN_TASK_STACK_SIZE=4096',
            'CONFIG_LOG_DEFAULT_LEVEL_INFO=y',
            'CONFIG_LOG_MAXIMUM_LEVEL_DEBUG=y',
            'CONFIG_COMPILER_OPTIMIZATION_SIZE=y',
            "",
        ]
    )


def _render_partitions_csv(board: BoardName, flash_size: FlashSize | None) -> str:
    """Return partitions.csv content for the selected board."""

    header = "# Name,   Type, SubType, Offset,   Size,   Flags"

    if board == "esp32_wroom32":
        selected_flash = flash_size or "4mb"
        if selected_flash == "8mb":
            factory_size = "2M"
            spiffs_size = "3M"
        else:
            factory_size = "1200K"
            spiffs_size = "1M"

        return "\n".join(
            [
                header,
                f"nvs,      data, nvs,     0x9000,  0x6000,",
                f"phy_init, data, phy,     0xf000,  0x1000,",
                f"factory,  app,  factory, 0x10000, {factory_size},",
                f"spiffs,   data, spiffs,  ,        {spiffs_size},",
                "",
            ]
        )

    # esp32s3_mini and esp32c3_mini share the same 4 MB table here
    return "\n".join(
        [
            header,
            "nvs,      data, nvs,     0x9000,  0x6000,",
            "phy_init, data, phy,     0xf000,  0x1000,",
            "factory,  app,  factory, 0x10000, 1200K,",
            "spiffs,   data, spiffs,  ,        1M,",
            "",
        ]
    )


def _write_text(path: Path, content: str) -> None:
    """Write a text file, creating parent directories as needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    """CLI entry-point."""

    args = parse_args()
    board: BoardName = args.board

    flash_size: FlashSize | None = args.flash_size
    if flash_size is not None and board != "esp32_wroom32":
        raise SystemExit("[ERROR] --flash-size is only valid with --board esp32_wroom32")

    if board == "esp32_wroom32" and flash_size is None:
        flash_size = "4mb"

    cfg = BOARD_CONFIGS[board]

    firmware_dir = Path(__file__).resolve().parent / "esp32_firmware"
    sdkconfig_path = firmware_dir / "sdkconfig.defaults"
    partitions_path = firmware_dir / "partitions.csv"

    sdkconfig = _render_sdkconfig_defaults(board, flash_size)
    partitions = _render_partitions_csv(board, flash_size)

    _write_text(sdkconfig_path, sdkconfig)
    _write_text(partitions_path, partitions)

    print("[INFO] Generated board configuration")
    print(f"[INFO] Board       : {board}")
    print(f"[INFO] IDF target  : {cfg.idf_target}")
    if board == "esp32_wroom32":
        assert flash_size is not None
        print(f"[INFO] Flash size  : {flash_size}")

    print(f"[INFO] Wrote       : {sdkconfig_path}")
    print(f"[INFO] Wrote       : {partitions_path}")

    print("\n[INFO] Next commands")
    print("  cd esp32_firmware")
    print(f"  idf.py set-target {cfg.idf_target}")
    print("  idf.py build")
    print("  idf.py -p <PORT> flash monitor")


if __name__ == "__main__":
    main()
