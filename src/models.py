from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Grid:
    cells: np.ndarray  # shape (8, 8), dtype bool; True = occupied

    @classmethod
    def empty(cls) -> Grid:
        return cls(cells=np.zeros((8, 8), dtype=bool))

    def copy(self) -> Grid:
        return Grid(cells=self.cells.copy())

    def __str__(self) -> str:
        rows = []
        for r in range(8):
            row = ""
            for c in range(8):
                row += "█" if self.cells[r, c] else "·"
            rows.append(row)
        return "\n".join(rows)


@dataclass
class Piece:
    shape: np.ndarray  # 2D bool array (H x W), minimal bbox

    @property
    def height(self) -> int:
        return self.shape.shape[0]

    @property
    def width(self) -> int:
        return self.shape.shape[1]

    def cells(self) -> list[tuple[int, int]]:
        """List of (row, col) offsets that are True."""
        return [(r, c) for r in range(self.height) for c in range(self.width) if self.shape[r, c]]

    def cell_count(self) -> int:
        return int(self.shape.sum())

    def __str__(self) -> str:
        rows = []
        for r in range(self.height):
            row = ""
            for c in range(self.width):
                row += "█" if self.shape[r, c] else " "
            rows.append(row)
        return "\n".join(rows)


@dataclass
class Placement:
    piece_idx: int
    row: int
    col: int
    rows_cleared: list[int] = field(default_factory=list)
    cols_cleared: list[int] = field(default_factory=list)

    @property
    def lines_cleared(self) -> int:
        return len(self.rows_cleared) + len(self.cols_cleared)


@dataclass
class SolveResult:
    placements: list[Placement]
    total_score: int
    order: list[int]  # piece indices in placement order
