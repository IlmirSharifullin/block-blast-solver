"""Screen capture and region calibration.

Calibration flow (first run):
  1. Take full screenshot.
  2. Ask user to click top-left and bottom-right corners of the 8×8 grid.
  3. Ask user to click top-left and bottom-right corners of the piece panel.
  4. Save coordinates to config.json.

Subsequent runs read config.json directly.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import NamedTuple

import mss
import mss.tools
import numpy as np
from PIL import Image

CONFIG_PATH = Path(__file__).parent.parent / "config.json"


class Region(NamedTuple):
    x: int
    y: int
    width: int
    height: int


class CaptureConfig(NamedTuple):
    grid_region: Region
    pieces_region: Region


def load_config() -> CaptureConfig | None:
    if not CONFIG_PATH.exists():
        return None
    data = json.loads(CONFIG_PATH.read_text())
    return CaptureConfig(
        grid_region=Region(**data["grid_region"]),
        pieces_region=Region(**data["pieces_region"]),
    )


def save_config(cfg: CaptureConfig) -> None:
    data = {
        "grid_region": cfg.grid_region._asdict(),
        "pieces_region": cfg.pieces_region._asdict(),
    }
    CONFIG_PATH.write_text(json.dumps(data, indent=2))


def screenshot_full() -> np.ndarray:
    """Capture entire primary screen, return as RGB numpy array."""
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # primary monitor
        raw = sct.grab(monitor)
        img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
        return np.array(img)


def screenshot_region(region: Region) -> np.ndarray:
    """Capture a specific region, return as RGB numpy array."""
    with mss.mss() as sct:
        monitor = {"left": region.x, "top": region.y, "width": region.width, "height": region.height}
        raw = sct.grab(monitor)
        img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
        return np.array(img)


def calibrate_interactive() -> CaptureConfig:
    """CLI calibration: show a full screenshot and ask user to enter coordinates."""
    print("\n=== Calibration ===")
    print("Take a screenshot of your Block Blast game now.")
    print("You can use Cmd+Shift+4 on macOS to get pixel coordinates.\n")

    def ask_region(name: str) -> Region:
        print(f"Enter coordinates for: {name}")
        x1 = int(input("  Top-left X: "))
        y1 = int(input("  Top-left Y: "))
        x2 = int(input("  Bottom-right X: "))
        y2 = int(input("  Bottom-right Y: "))
        return Region(x=x1, y=y1, width=x2 - x1, height=y2 - y1)

    grid_region = ask_region("8×8 game grid")
    pieces_region = ask_region("piece panel (the 3 pieces shown below the grid)")

    cfg = CaptureConfig(grid_region=grid_region, pieces_region=pieces_region)
    save_config(cfg)
    print(f"Config saved to {CONFIG_PATH}\n")
    return cfg


def get_config(recalibrate: bool = False) -> CaptureConfig:
    if not recalibrate:
        cfg = load_config()
        if cfg is not None:
            return cfg
    return calibrate_interactive()
