import torch, random
from math import prod
from collections import deque
from model import Model
from pytetris import PieceKind, Piece, Board

checkpoint = torch.load("checkpoints/090000-model.pth", "cpu")
model = Model(8)
model.load_state_dict(checkpoint["model"])
model.eval()

def get_moves(board: Board, start: Piece) -> set[Piece]:
    visited = { start }
    queue = deque([start])
    while len(queue) > 0:
        piece = queue.popleft()

        for sx, sy in (-1, 0), (1, 0), (0, -1):
            shifted = piece.shifted(sx, sy, board)
            if shifted is not None and shifted not in visited:
                queue.append(shifted)
                visited.add(shifted)

        for dir in -1, 1:
            rotated = piece.rotated(dir, board)
            if rotated is not None and rotated not in visited:
                queue.append(rotated)
                visited.add(rotated)

    return { piece for piece in visited if piece.shifted(0, -1, board) is None }

def get_score_matrix(board: Board, queue: list[PieceKind]) -> torch.Tensor:
    with torch.no_grad():
        field = torch.zeros(200)
        for y in range(20):
            for x in range(10):
                if board[y][x] is not None:
                    field[y * 10 + x] = 1.0

        queue_tensor = torch.zeros(5, dtype=torch.long)
        for i in range(5):
            queue_tensor[i] = list(PieceKind).index(queue[i])

        pred = model(field.unsqueeze(0), queue_tensor.unsqueeze(0))
        pred = torch.sigmoid(pred).reshape((20, 10))

        return pred

board: Board = [[None] * 10 for _ in range(40)]
queue: list[PieceKind] = []

while True:
    while len(queue) < 5:
        bag = list(PieceKind)
        random.shuffle(bag)
        queue.extend(bag)

    score_matrix = get_score_matrix(board, queue)
    # print(score_matrix.round(decimals=1).reshape(20, 10))

    active_a = Piece.spawned(queue[0])
    active_b = Piece.spawned(queue[1])

    best = max(
        get_moves(board, active_a) | get_moves(board, active_b),
        key=lambda piece: sum(
            score_matrix[min(y, 19), x].item()
            for x, y in piece.cells()
        ),
    )
    best.lock_to(board)
    if best.kind == active_a:
        queue.pop(0)
    else:
        queue.pop(1)

    for y in reversed(range(20)):
        for x in range(10):
            if board[y][x] is not None:
                print("[]", end="")
            else:
                print("..", end="")
        print()
    print()
