"""Detect grid state and piece shapes from captured screenshot regions.

Grid detection strategy:
  - Divide the grid ROI into 8×8 equal cells.
  - For each cell, sample a central 5×5 pixel patch.
  - Compare average brightness to a threshold: dark = empty, bright/colored = occupied.

Piece detection strategy:
  - Divide the pieces ROI into 3 equal horizontal zones (one per piece).
  - In each zone, find the bounding box of non-background pixels.
  - Divide the bounding box into a small grid (cell size estimated from piece scale).
  - Build a boolean shape matrix.
"""
from __future__ import annotations

import numpy as np
import cv2
from .models import Grid, Piece

GRID_SIZE = 8
PIECE_COUNT = 3

# HSV saturation threshold: pixels with saturation > this are considered "colored" (filled)
SAT_THRESHOLD = 136
# Brightness threshold for the grid background (0-255); below = dark = empty
BRIGHTNESS_THRESHOLD = 90


def parse_grid(grid_img: np.ndarray) -> Grid:
    """Given an RGB image of the 8×8 grid area, return a Grid."""
    h, w = grid_img.shape[:2]
    cell_h = h / GRID_SIZE
    cell_w = w / GRID_SIZE

    cells = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    hsv = cv2.cvtColor(grid_img, cv2.COLOR_RGB2HSV)

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            cy = int((r + 0.5) * cell_h)
            cx = int((c + 0.5) * cell_w)
            # Sample 5×5 patch around cell center
            patch = hsv[
                max(0, cy - 2): cy + 3,
                max(0, cx - 2): cx + 3,
            ]
            avg_sat = float(patch[:, :, 1].mean())
            avg_val = float(patch[:, :, 2].mean())
            print(avg_sat, avg_val)
            # Occupied if colorful OR bright (white blocks in some themes)
            cells[r, c] = avg_sat > SAT_THRESHOLD or avg_val > 200
    return Grid(cells=cells)


def parse_pieces(pieces_img: np.ndarray) -> list[Piece]:
    """Given an RGB image of the piece panel, return up to 3 Piece objects."""
    h, w = pieces_img.shape[:2]
    zone_w = w // PIECE_COUNT
    pieces: list[Piece] = []

    print(f"[pieces] panel size: {w}×{h}px, zone width: {zone_w}px each")

    for i in range(PIECE_COUNT):
        zone = pieces_img[:, i * zone_w: (i + 1) * zone_w]
        print(f"[pieces] --- zone {i + 1} ---")
        piece = _detect_piece_in_zone(zone, slot=i + 1)
        if piece is not None:
            pieces.append(piece)
            print(f"[pieces] zone {i + 1}: detected {piece.height}×{piece.width} piece, {piece.cell_count()} cells")
            print(piece)
        else:
            print(f"[pieces] zone {i + 1}: empty (no piece detected)")

    return pieces


def _detect_piece_in_zone(zone: np.ndarray, slot: int = 0) -> Piece | None:
    """Detect a single piece shape within a zone image."""
    hsv = cv2.cvtColor(zone, cv2.COLOR_RGB2HSV)
    sat_mask = hsv[:, :, 1] > SAT_THRESHOLD
    val_mask = hsv[:, :, 2] > 180
    not_white = hsv[:, :, 1] > 30
    mask = sat_mask | (val_mask & not_white)

    mask_u8 = mask.astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)

    coords = np.argwhere(mask_u8 > 0)
    print(f"[pieces] zone {slot}: colored pixels after morphology: {len(coords)}")
    if len(coords) < 9:
        return None

    r_min, c_min = coords.min(axis=0)
    r_max, c_max = coords.max(axis=0)
    bbox_h = r_max - r_min + 1
    bbox_w = c_max - c_min + 1
    print(f"[pieces] zone {slot}: bounding box {bbox_w}×{bbox_h}px at ({c_min},{r_min})")

    bbox = mask_u8[r_min: r_max + 1, c_min: c_max + 1]

    cell_size = _estimate_cell_size(bbox)
    if cell_size < 2:
        cell_size = max(bbox_h, bbox_w) // 5 or 1
    print(f"[pieces] zone {slot}: estimated cell size: {cell_size:.1f}px")

    grid_rows = max(1, round(bbox_h / cell_size))
    grid_cols = max(1, round(bbox_w / cell_size))
    print(f"[pieces] zone {slot}: shape grid {grid_cols}×{grid_rows}")

    shape = np.zeros((grid_rows, grid_cols), dtype=bool)
    for gr in range(grid_rows):
        for gc in range(grid_cols):
            y0 = int(gr * cell_size)
            y1 = int(min((gr + 1) * cell_size, bbox_h))
            x0 = int(gc * cell_size)
            x1 = int(min((gc + 1) * cell_size, bbox_w))
            cell_patch = bbox[y0:y1, x0:x1]
            fill_ratio = cell_patch.mean() / 255.0
            shape[gr, gc] = fill_ratio > 0.3
            print(f"[pieces] zone {slot}:   cell ({gr},{gc}) fill={fill_ratio:.2f} → {'█' if shape[gr, gc] else '·'}")

    shape = _trim_shape(shape)
    if shape.size == 0:
        return None
    return Piece(shape=shape)

    
def _estimate_cell_size(mask: np.ndarray) -> float:
    """Estimate single cell pixel size by finding the most common gap between filled rows."""
    row_sums = mask.mean(axis=1)
    filled_rows = np.where(row_sums > 50)[0]
    if len(filled_rows) < 2:
        return float(mask.shape[0])
    gaps = np.diff(filled_rows)
    gaps = gaps[gaps > 1]
    if len(gaps) == 0:
        return float(mask.shape[0])
    return float(np.median(gaps))


def _trim_shape(shape: np.ndarray) -> np.ndarray:
    rows = np.any(shape, axis=1)
    cols = np.any(shape, axis=0)
    if not rows.any() or not cols.any():
        return np.zeros((0, 0), dtype=bool)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return shape[r0: r1 + 1, c0: c1 + 1]


def save_debug_image(grid_img: np.ndarray, grid: Grid, output_path: str = "debug_grid.png") -> None:
    """Draw detected cell states on top of the grid image and save."""
    from PIL import Image, ImageDraw
    img = Image.fromarray(grid_img)
    draw = ImageDraw.Draw(img)
    h, w = grid_img.shape[:2]
    cell_h = h / GRID_SIZE
    cell_w = w / GRID_SIZE
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            cy = int((r + 0.5) * cell_h)
            cx = int((c + 0.5) * cell_w)
            color = (0, 255, 0) if grid.cells[r, c] else (255, 0, 0)
            draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=color)
    img.save(output_path)
    print(f"Debug image saved: {output_path}")


def save_debug_pieces_image(pieces_img: np.ndarray, pieces: list[Piece], output_path: str = "debug_pieces.png") -> None:
    """Draw detected piece cell states on top of the piece panel image and save."""
    from PIL import Image, ImageDraw
    img = Image.fromarray(pieces_img)
    draw = ImageDraw.Draw(img)
    h, w = pieces_img.shape[:2]
    zone_w = w // PIECE_COUNT

    for i, piece in enumerate(pieces):
        zone_x = i * zone_w

        # Find bounding box of colored pixels in this zone to match _detect_piece_in_zone logic
        zone = pieces_img[:, zone_x: zone_x + zone_w]
        hsv = cv2.cvtColor(zone, cv2.COLOR_RGB2HSV)
        sat_mask = hsv[:, :, 1] > SAT_THRESHOLD
        val_mask = hsv[:, :, 2] > 180
        not_white = hsv[:, :, 1] > 30
        mask = (sat_mask | (val_mask & not_white)).astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        coords = np.argwhere(mask > 0)
        if len(coords) < 9:
            continue

        r_min, c_min = coords.min(axis=0)
        r_max, c_max = coords.max(axis=0)
        bbox_h = r_max - r_min + 1
        bbox_w = c_max - c_min + 1

        cell_size = _estimate_cell_size(mask[r_min: r_max + 1, c_min: c_max + 1])
        if cell_size < 2:
            cell_size = max(bbox_h, bbox_w) // 5 or 1

        # Draw a dot at the center of each detected shape cell
        for pr in range(piece.height):
            for pc in range(piece.width):
                cy = int(r_min + (pr + 0.5) * cell_size)
                cx = zone_x + int(c_min + (pc + 0.5) * cell_size)
                color = (0, 255, 0) if piece.shape[pr, pc] else (255, 0, 0)
                draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=color)

        # Draw zone boundary
        draw.rectangle([zone_x + 1, 1, zone_x + zone_w - 2, h - 2], outline=(255, 200, 0), width=2)

    img.save(output_path)
    print(f"Debug image saved: {output_path}")
