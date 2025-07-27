use libtetris::*;
use ndarray::prelude::*;
use num_traits::{Zero, One};

pub fn piece_index(piece: Piece) -> usize {
    match piece {
        Piece::I => 0,
        Piece::J => 1,
        Piece::L => 2,
        Piece::O => 3,
        Piece::S => 4,
        Piece::T => 5,
        Piece::Z => 6,
    }
}

pub fn rotation_index(rotation: RotationState) -> usize {
    match rotation {
        RotationState::North => 0,
        RotationState::East => 1,
        RotationState::South => 2,
        RotationState::West => 3,
    }
}

pub fn field_tensor<T: Zero + One + Clone>(board: &Board) -> Array2<T> {
    let mut tensor = Array2::zeros([20, 10]);
    for y in 0..20 {
        for x in 0..10 {
            if board.occupied(x as i32, y as i32) {
                tensor[[y, x]] = T::one();
            }
        }
    }

    tensor
}

pub fn queue_tensor<T: Zero + Clone + TryFrom<usize>>(board: &Board, len: usize) -> Array1<T> {
    let mut tensor = Array1::zeros(len);
    let mut queue_iter = board.hold_piece.into_iter().chain(board.next_queue());
    for out in &mut tensor {
        let piece = queue_iter.next().unwrap();
        *out = piece_index(piece).try_into().ok().unwrap();
    }

    tensor
}

pub fn move_index(mv: FallingPiece) -> Option<[usize; 4]> {
    if mv.y >= 20 {
        return None;
    }

    Some([
        piece_index(mv.kind.0),
        rotation_index(mv.kind.1),
        mv.y as usize,
        mv.x as usize,
    ])
}
