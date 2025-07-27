use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::TensorRef;
use ndarray::Array4;
use libtetris::*;
use rand::prelude::*;

const MIN_QUEUE: usize = 10;

fn piece_to_u8(piece: Piece) -> u8 {
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

fn rotation_to_u8(rotation: RotationState) -> u8 {
    match rotation {
        RotationState::North => 0,
        RotationState::East => 1,
        RotationState::South => 2,
        RotationState::West => 3,
    }
}

fn load_model(path: &str) -> ort::Result<Session> {
    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(path)?;

    Ok(model)
}

fn infer(model: &mut Session, board: &Board) -> Array4<f32> {
    let mut field = [0.0; 200];
    for y in 0..20 {
        for x in 0..10 {
            if board.occupied(x, y) {
                field[(y * 10 + x) as usize] = 1.0f32;
            }
        }
    }

    let mut queue = [0; 5];
    let queue_iter = board.hold_piece.into_iter().chain(board.next_queue());
    for (out, piece) in queue.iter_mut().zip(queue_iter) {
        *out = piece_to_u8(piece) as i64;
    }

    let inputs = ort::inputs![
        "field" => TensorRef::from_array_view(([1, 200], field.as_slice())).unwrap(),
        "queue" => TensorRef::from_array_view(([1, 5], queue.as_slice())).unwrap(),
    ];

    let outputs = model.run(inputs).unwrap();
    let pred = outputs["pred"].try_extract_array::<f32>().unwrap();

    let pred = pred.to_shape([7, 4, 20, 10]).unwrap();
    let pred = pred.to_owned();

    pred
}

fn search(model: &mut Session, board: &Board, beam: usize, depth: u32) -> Option<(f32, Placement)> {
    let policy = infer(model, board);

    let main = board.get_next_piece().unwrap();
    let hold = board.hold_piece.unwrap_or(board.get_next_next_piece().unwrap());

    let mut moves = Vec::new();
    for piece in [main, hold] {
        if let Some(spawned) = SpawnRule::Row19Or20.spawn(piece, board) {
            for mv in find_moves(board, spawned, MovementMode::ZeroG) {
                let logit = policy[[
                    piece_to_u8(piece) as usize,
                    rotation_to_u8(mv.location.kind.1) as usize,
                    mv.location.y as usize,
                    mv.location.x as usize,
                ]];
                let logp = -(-logit).exp().ln_1p();
                moves.push((logp, mv));
            }
        }
    }
    moves.sort_unstable_by(|(a, _), (b, _)| b.total_cmp(a));

    if depth == 0 {
        return moves.into_iter().next();
    }

    moves.into_iter()
        .take(beam)
        .map(|(logp, mv)| {
            let mut child_board = board.clone();

            child_board.lock_piece(mv.location);
            child_board.advance_queue();
            if mv.location.kind.0 != main && child_board.hold(main).is_none() {
                child_board.advance_queue();
            }

            match search(model, &child_board, beam, depth - 1) {
                Some((child_logp, _)) => (logp + child_logp, mv),
                None => (-100.0, mv),
            }
        })
        .max_by(|(a, _), (b, _)| a.total_cmp(b))
}

fn main() {
    let model_path = std::env::args().nth(1).unwrap();
    let mut model = load_model(&model_path).unwrap();

    let mut bag = Vec::new();
    let mut draw_piece = || {
        if bag.is_empty() {
            bag.extend_from_slice(&[
                Piece::I,
                Piece::J,
                Piece::L,
                Piece::O,
                Piece::S,
                Piece::T,
                Piece::Z,
            ]);
            bag.shuffle(&mut thread_rng());
        }
        bag.pop().unwrap()
    };

    let mut board = Board::new();
    for _ in board.next_queue().count()..MIN_QUEUE {
        board.add_next_piece(draw_piece());
    }

    while let Some((logp, mv)) = search(&mut model, &board, 2, 5) {
        board.lock_piece(mv.location);
        let main = board.advance_queue().unwrap();
        if mv.location.kind.0 != main && board.hold(main).is_none() {
            board.advance_queue();
        }

        for _ in board.next_queue().count()..MIN_QUEUE {
            board.add_next_piece(draw_piece());
        }

        for y in (0..20).rev() {
            for x in 0..10 {
                if board.occupied(x, y) {
                    print!("[]");
                } else {
                    print!("..");
                }
            }
            println!();
        }
        
        if let Some(hold) = board.hold_piece {
            print!(" H: {} Q:", hold.to_char());
        } else {
            print!(" H: . Q:");
        }
        
        for piece in board.next_queue().take(5) {
            print!(" {}", piece.to_char());
        }
        println!();

        println!(" P: {:.2}", logp.exp());
        println!();
    }
}
