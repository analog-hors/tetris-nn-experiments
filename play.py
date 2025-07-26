import torch, random, sys
from math import log
from collections import deque
from model import Model
from pytetris import PieceKind, Piece, Board

checkpoint = torch.load(sys.argv[1], "cpu")
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

def beam_search(board: Board, queue: list[PieceKind], depth: int) -> tuple[float, Piece]:
    
    score_matrix = get_score_matrix(board, queue)
    def score_move(piece: Piece) -> float:
        return sum(
            log(score_matrix[min(y, 19), x].item())
            for x, y in piece.cells()
        )
    
    active_a = Piece.spawned(queue[0])
    active_b = Piece.spawned(queue[1])
    moves = get_moves(board, active_a) | get_moves(board, active_b)

    scored_moves = [(score_move(move), move) for move in moves]
    scored_moves.sort(key=lambda p: p[0], reverse=True)

    if depth == 0:
        return scored_moves[0]

    best = None
    for score, move in scored_moves[:2]:
        child_board = [row.copy() for row in board]
        move.lock_to(child_board)

        child_queue = queue.copy()
        if move.kind == active_a.kind:
            child_queue.pop(0)
        else:
            child_queue.pop(1)
        
        child_score, _ = beam_search(child_board, child_queue, depth - 1)
        if best is None or best[0] < child_score + score:
            best = (child_score + score), move

    assert best is not None
    return best

board: Board = [[None] * 10 for _ in range(40)]
queue: list[PieceKind] = []

while True:
    while len(queue) < 9:
        bag = list(PieceKind)
        random.shuffle(bag)
        queue.extend(bag)

    score, best = beam_search(board, queue, 4)

    best.lock_to(board)
    if best.kind == queue[0]:
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
