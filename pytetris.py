from enum import Enum
from dataclasses import dataclass
from typing import Iterator

class PieceKind(Enum):
    I = "I"
    J = "J"
    L = "L"
    O = "O"
    S = "S"
    T = "T"
    Z = "Z"

Board = list[list[PieceKind | None]]

PIECE_SHAPES = {
    PieceKind.I: [[-1,  0], [ 0,  0], [ 1,  0], [ 2,  0]],
    PieceKind.J: [[-1,  1], [-1,  0], [ 0,  0], [ 1,  0]],
    PieceKind.L: [[-1,  0], [ 0,  0], [ 1,  1], [ 1,  0]],
    PieceKind.O: [[ 0,  1], [ 0,  0], [ 1,  1], [ 1,  0]],
    PieceKind.S: [[-1,  0], [ 0,  1], [ 0,  0], [ 1,  1]],
    PieceKind.T: [[-1,  0], [ 0,  1], [ 0,  0], [ 1,  0]],
    PieceKind.Z: [[-1,  1], [ 0,  1], [ 0,  0], [ 1,  0]],
}

JLSTZ_OFFSET_TABLE = [
    [[ 0,  0], [ 0,  0], [ 0,  0], [ 0,  0], [ 0,  0]],
    [[ 0,  0], [ 1,  0], [ 1, -1], [ 0,  2], [ 1,  2]],
    [[ 0,  0], [ 0,  0], [ 0,  0], [ 0,  0], [ 0,  0]],
    [[ 0,  0], [-1,  0], [-1, -1], [ 0,  2], [-1,  2]],
]

I_OFFSET_TABLE = [
    [[ 0,  0], [-1,  0], [ 2,  0], [-1,  0], [ 2,  0]],
    [[-1,  0], [ 0,  0], [ 0,  0], [ 0,  1], [ 0, -2]],
    [[-1,  1], [ 1,  1], [-2,  1], [ 1,  0], [-2,  0]],
    [[ 0,  1], [ 0,  1], [ 0,  1], [ 0, -1], [ 0,  2]],
]

O_OFFSET_TABLE = [
    [[ 0,  0], [ 0,  0], [ 0,  0], [ 0,  0], [ 0,  0]],
    [[ 0, -1], [ 0,  0], [ 0,  0], [ 0,  0], [ 0,  0]],
    [[-1, -1], [ 0,  0], [ 0,  0], [ 0,  0], [ 0,  0]],
    [[-1,  0], [ 0,  0], [ 0,  0], [ 0,  0], [ 0,  0]],
]

PIECE_OFFSET_TABLES = {
    PieceKind.I: I_OFFSET_TABLE,
    PieceKind.J: JLSTZ_OFFSET_TABLE,
    PieceKind.L: JLSTZ_OFFSET_TABLE,
    PieceKind.O: O_OFFSET_TABLE,
    PieceKind.S: JLSTZ_OFFSET_TABLE,
    PieceKind.T: JLSTZ_OFFSET_TABLE,
    PieceKind.Z: JLSTZ_OFFSET_TABLE,
}

@dataclass(frozen=True)
class Piece:
    kind: PieceKind
    rotation: int
    x: int
    y: int

    @staticmethod
    def spawned(kind: PieceKind) -> "Piece":
        return Piece(kind, 0, 3, 19)

    def cells(self) -> Iterator[tuple[int, int]]:
        for x, y in PIECE_SHAPES[self.kind]:
            for _ in range(self.rotation):
                x, y = y, -x
            yield self.x + x, self.y + y

    def collides(self, board) -> bool:
        for x, y in self.cells():
            if x < 0 or x >= 10 or y < 0 or y >= 40:
                return True
            if board[y][x] is not None:
                return True
        
        return False

    def rotated(self, dir: int, board: Board) -> "Piece | None":
        target = (self.rotation + dir) % 4

        offset_table = PIECE_OFFSET_TABLES[self.kind]
        from_table = offset_table[self.rotation]
        to_table = offset_table[target]

        for i in range(len(from_table)):
            fx, fy = from_table[i]
            tx, ty = to_table[i]
            rotated = Piece(
                self.kind,
                target,
                self.x + fx - tx,
                self.y + fy - ty,
            )
            if not rotated.collides(board):
                return rotated

        return None

    def shifted(self, sx: int, sy: int, board: Board) -> "Piece | None":
        shifted = Piece(
            self.kind,
            self.rotation,
            self.x + sx,
            self.y + sy,
        )
        if not shifted.collides(board):
            return shifted

        return None

    def lock_to(self, board: Board):
        for x, y in self.cells():
            assert x in range(10)
            assert y in range(40)
            assert board[y][x] is None
            board[y][x] = self.kind
        
        height = 0
        for i in range(len(board)):
            if None in board[i]:
                board[height] = board[i]
                height += 1

        for i in range(height, len(board)):
            board[i] = [None] * 10
