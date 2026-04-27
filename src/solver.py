from __future__ import annotations
from itertools import permutations
from .models import Grid, Piece, Placement, SolveResult

SCORE_PER_CELL = 1
SCORE_PER_LINE = 10


def can_place(grid: Grid, piece: Piece, row: int, col: int) -> bool:
    if row + piece.height > 8 or col + piece.width > 8:
        return False
    for dr, dc in piece.cells():
        if grid.cells[row + dr, col + dc]:
            return False
    return True


def place_and_clear(grid: Grid, piece: Piece, row: int, col: int) -> tuple[Grid, list[int], list[int]]:
    new_grid = grid.copy()
    for dr, dc in piece.cells():
        new_grid.cells[row + dr, col + dc] = True

    rows_cleared: list[int] = []
    cols_cleared: list[int] = []
    for r in range(8):
        if new_grid.cells[r, :].all():
            rows_cleared.append(r)
    for c in range(8):
        if new_grid.cells[:, c].all():
            cols_cleared.append(c)

    for r in rows_cleared:
        new_grid.cells[r, :] = False
    for c in cols_cleared:
        new_grid.cells[:, c] = False

    return new_grid, rows_cleared, cols_cleared


def solve(grid: Grid, pieces: list[Piece]) -> SolveResult | None:
    """Try all permutations and all positions; return highest-scoring placement of all pieces."""
    best: list[SolveResult | None] = [None]
    for perm in permutations(range(len(pieces))):
        _recurse(grid, [pieces[i] for i in perm], list(perm), [], best)
    return best[0]


def _recurse(
    grid: Grid,
    pieces_left: list[Piece],
    idx_left: list[int],
    acc: list[tuple[Placement, int]],  # (placement, cells_placed)
    best: list[SolveResult | None],
) -> None:
    if not pieces_left:
        score = sum(lines * SCORE_PER_LINE + cells * SCORE_PER_CELL for _, (lines, cells) in _unpack(acc))
        if best[0] is None or score > best[0].total_score:
            best[0] = SolveResult(
                placements=[p for p, _ in acc],
                total_score=score,
                order=[p.piece_idx for p, _ in acc],
            )
        return

    piece = pieces_left[0]
    piece_idx = idx_left[0]
    cell_count = piece.cell_count()

    for r in range(8):
        for c in range(8):
            if can_place(grid, piece, r, c):
                new_grid, rows_cl, cols_cl = place_and_clear(grid, piece, r, c)
                placement = Placement(piece_idx=piece_idx, row=r, col=c,
                                      rows_cleared=rows_cl, cols_cleared=cols_cl)
                _recurse(new_grid, pieces_left[1:], idx_left[1:],
                         acc + [(placement, cell_count)], best)


def _unpack(acc: list[tuple[Placement, int]]):
    for placement, cells in acc:
        yield placement, (placement.lines_cleared, cells)
