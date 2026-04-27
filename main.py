"""Block Blast Solver — entry point.

Usage:
  python main.py               # run full pipeline (calibrate if needed)
  python main.py --calibrate   # force re-calibration
  python main.py --demo        # run with a built-in demo board (no screen capture)
"""
from __future__ import annotations
import argparse
import sys
import numpy as np

from src.capture import get_config, screenshot_region
from src.detector import parse_grid, parse_pieces, save_debug_image, save_debug_pieces_image
from src.solver import solve
from src.visualizer import print_solution, render_solution_image
from src.models import Grid, Piece


def run_demo() -> None:
    """Run solver on a hardcoded board for testing without screen capture."""
    # Almost-full board: row 7 needs one more block to clear
    cells = np.zeros((8, 8), dtype=bool)
    cells[7, :7] = True   # row 7 col 0-6 filled, col 7 empty
    cells[0, :7] = True   # row 0 almost full, col 7 empty
    grid = Grid(cells=cells)

    # Three test pieces
    pieces = [
        # 1×1 single block
        Piece(shape=np.array([[True]], dtype=bool)),
        # 1×3 horizontal bar
        Piece(shape=np.array([[True, True, True]], dtype=bool)),
        # L-shape
        Piece(shape=np.array([
            [True, False],
            [True, False],
            [True, True],
        ], dtype=bool)),
    ]

    print("=== DEMO MODE ===")
    print("\nInitial board:")
    print(grid)
    print()

    result = solve(grid, pieces)
    if result is None:
        print("No valid placement found for all pieces!")
        return

    print_solution(grid, pieces, result)


def run_live(recalibrate: bool = False) -> None:
    """Capture screen, detect state, solve, show result."""
    cfg = get_config(recalibrate=recalibrate)

    print("Capturing screen...")
    grid_img = screenshot_region(cfg.grid_region)
    pieces_img = screenshot_region(cfg.pieces_region)

    print("Parsing grid...")
    grid = parse_grid(grid_img)
    print(grid)

    print("\nParsing pieces...")
    pieces = parse_pieces(pieces_img)
    if not pieces:
        print("ERROR: No pieces detected. Check calibration.")
        sys.exit(1)
    print(f"Detected {len(pieces)} piece(s)")
    for i, p in enumerate(pieces):
        print(f"\nPiece {i + 1} ({p.height}×{p.width}, {p.cell_count()} cells):")
        print(p)

    save_debug_image(grid_img, grid, "debug_grid.png")
    save_debug_pieces_image(pieces_img, pieces, "debug_pieces.png")

    print("\nSolving...")
    result = solve(grid, pieces)
    if result is None:
        print("No valid placement found for all pieces.")
        return

    print_solution(grid, pieces, result)
    render_solution_image(grid_img, grid, pieces, result, "solution.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Block Blast Solver")
    parser.add_argument("--calibrate", action="store_true", help="Force re-calibration of screen regions")
    parser.add_argument("--demo", action="store_true", help="Run with built-in demo board")
    args = parser.parse_args()

    if args.demo:
        run_demo()
    else:
        run_live(recalibrate=args.calibrate)


if __name__ == "__main__":
    main()
