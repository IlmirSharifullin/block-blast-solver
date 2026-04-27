"""Visualize the solver result in the terminal and as a PNG overlay."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .models import Grid, Piece, Placement, SolveResult
from .solver import place_and_clear, can_place

PIECE_COLORS_TERM = ["\033[92m", "\033[93m", "\033[94m"]  # green, yellow, blue
RESET = "\033[0m"
CLEARED_COLOR = "\033[91m"  # red for cleared lines

PIECE_COLORS_RGB = [(80, 220, 100), (255, 210, 60), (80, 140, 255)]


def print_solution(grid: Grid, pieces: list[Piece], result: SolveResult) -> None:
    """Print the board state and solution moves to the terminal."""
    print("\n=== Detected board ===")
    _print_grid(grid, {}, set(), set())

    print("\n=== Pieces ===")
    for i, piece in enumerate(pieces):
        color = PIECE_COLORS_TERM[i % len(PIECE_COLORS_TERM)]
        print(f"\nPiece {i + 1}:")
        for row in str(piece).splitlines():
            print(f"  {color}{row}{RESET}")

    print("\n=== Solution ===")
    print(f"Score: {result.total_score}\n")

    current_grid = grid.copy()
    all_rows_cleared: set[int] = set()
    all_cols_cleared: set[int] = set()

    for step, placement in enumerate(result.placements):
        piece = pieces[placement.piece_idx]
        color = PIECE_COLORS_TERM[placement.piece_idx % len(PIECE_COLORS_TERM)]
        lines = placement.lines_cleared
        print(
            f"Step {step + 1}: place piece {placement.piece_idx + 1} "
            f"at row={placement.row + 1}, col={placement.col + 1}  "
            f"({piece.cell_count()} cells"
            + (f", clears {lines} line(s)" if lines else "")
            + ")"
        )
        if placement.rows_cleared:
            print(f"  Rows cleared: {[r + 1 for r in placement.rows_cleared]}")
        if placement.cols_cleared:
            print(f"  Cols cleared: {[c + 1 for c in placement.cols_cleared]}")

        occupied: dict[tuple[int, int], int] = {}
        for dr, dc in piece.cells():
            occupied[(placement.row + dr, placement.col + dc)] = placement.piece_idx

        all_rows_cleared.update(placement.rows_cleared)
        all_cols_cleared.update(placement.cols_cleared)

        current_grid, _, _ = place_and_clear(current_grid, piece, placement.row, placement.col)

    print("\n=== Board after all moves ===")
    _print_grid(current_grid, {}, set(), set())


def _print_grid(
    grid: Grid,
    highlights: dict[tuple[int, int], int],
    cleared_rows: set[int],
    cleared_cols: set[int],
) -> None:
    print("  +" + "-" * 8 + "+")
    for r in range(8):
        row_str = f"{r + 1} |"
        for c in range(8):
            if r in cleared_rows or c in cleared_cols:
                row_str += f"{CLEARED_COLOR}×{RESET}"
            elif (r, c) in highlights:
                piece_idx = highlights[(r, c)]
                color = PIECE_COLORS_TERM[piece_idx % len(PIECE_COLORS_TERM)]
                row_str += f"{color}█{RESET}"
            elif grid.cells[r, c]:
                row_str += "█"
            else:
                row_str += "·"
        row_str += "|"
        print(row_str)
    print("  +" + "-" * 8 + "+")
    print("   12345678")


def render_solution_image(
    grid_img: np.ndarray,
    grid: Grid,
    pieces: list[Piece],
    result: SolveResult,
    output_path: str = "solution.png",
) -> None:
    """Draw colored overlays on the grid screenshot showing where to place each piece."""
    img = Image.fromarray(grid_img).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    h, w = grid_img.shape[:2]
    cell_h = h / 8
    cell_w = w / 8

    for placement in result.placements:
        piece = pieces[placement.piece_idx]
        color = PIECE_COLORS_RGB[placement.piece_idx % len(PIECE_COLORS_RGB)]
        rgba = (*color, 150)

        for dr, dc in piece.cells():
            r = placement.row + dr
            c = placement.col + dc
            x0 = int(c * cell_w) + 2
            y0 = int(r * cell_h) + 2
            x1 = int((c + 1) * cell_w) - 2
            y1 = int((r + 1) * cell_h) - 2
            draw.rectangle([x0, y0, x1, y1], fill=rgba)

        # Draw step number at piece center
        center_r = placement.row + piece.height / 2
        center_c = placement.col + piece.width / 2
        tx = int(center_c * cell_w)
        ty = int(center_r * cell_h)
        label = str(result.placements.index(placement) + 1)
        draw.text((tx, ty), label, fill=(255, 255, 255, 255))

    combined = Image.alpha_composite(img, overlay).convert("RGB")
    combined.save(output_path)
    print(f"Solution image saved: {output_path}")
