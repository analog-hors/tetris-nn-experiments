use std::time::Instant;

use rand::prelude::*;
use rand_pcg::Pcg64;
use cold_clear::{Interface, Options, Info};
use cold_clear::evaluation::Standard;
use libtetris::*;
use nn_encoding::*;

const MIN_QUEUE: usize = 6;
const GARBAGE_INTERVAL: u64 = 32;
const MAX_GARBAGE: u32 = 8;
const LOG_INTERVAL: u64 = 10_000;

struct State {
    rng: Pcg64,
    bag: Vec<Piece>,
    queue: Vec<Piece>,
    board: Board,
    bot: Interface,
}

impl State {
    fn new() -> Self {
        let mut this = Self {
            rng: Pcg64::seed_from_u64(random()),
            bag: Vec::new(),
            queue: Vec::new(),
            board: Board::new(),
            bot: Interface::launch(
                Board::new(),
                Options {
                    min_nodes: 2500,
                    max_nodes: 2500,
                    ..Default::default()
                },
                Standard::default(),
                None,
            ),
        };

        while this.queue.len() < MIN_QUEUE {
            this.queue_piece();
        }

        this
    }

    fn queue_piece(&mut self) {
        if self.bag.is_empty() {
            self.bag.extend_from_slice(&[
                Piece::I,
                Piece::J,
                Piece::L,
                Piece::O,
                Piece::S,
                Piece::T,
                Piece::Z,
            ]);
            self.bag.shuffle(&mut self.rng);
        }

        let piece = self.bag.pop().unwrap();
        self.queue.push(piece);
        self.board.add_next_piece(piece);
        self.bot.add_next_piece(piece);
    }

    fn get_move(&mut self) -> Option<(Move, Info)> {
        self.bot.suggest_next_move(0);
        self.bot.block_next_move()
    }

    fn play_move(&mut self, mv: Move) {
        let next = self.board.advance_queue().unwrap();
        self.queue.remove(0);
        self.queue_piece();

        if mv.hold && self.board.hold(next).is_none() {
            self.board.advance_queue();
            self.queue.remove(0);
            self.queue_piece();
        }

        self.board.lock_piece(mv.expected_location);
        self.bot.play_next_move(mv.expected_location);
    }

    fn add_garbage(&mut self, n: u32) {
        for _ in 0..n {
            self.board.add_garbage(self.rng.gen_range(0, 10));
        }
        self.bot.reset(self.board.get_field(), self.board.b2b_bonus, self.board.combo);
    }
}

fn write_sample(out: &mut impl std::io::Write, state: &State, mv: &Move) -> bool {
    let field = field_tensor::<u8>(&state.board);
    let queue = queue_tensor::<u8>(&state.board, MIN_QUEUE - 1);
    let Some(target) = move_index(mv.expected_location.canonical()) else {
        return false;
    };
    let target = target.map(|i| i.try_into().unwrap());

    out.write_all(&field.flatten().to_vec()).unwrap();
    out.write_all(&queue.to_vec()).unwrap();
    out.write_all(&target).unwrap();

    true
}

fn main() {
    let mut stdout = std::io::stdout().lock();
    let mut written = 0;
    let mut last_log = Instant::now();
    loop {
        let mut state = State::new();
        let mut moves = 0;
        while let Some((mv, _)) = state.get_move() {
            if write_sample(&mut stdout, &state, &mv) {
                written += 1;
                if written % LOG_INTERVAL == 0 {
                    eprintln!(
                        "{} samples written ({:.0} samples/second)",
                        written,
                        LOG_INTERVAL as f64 / last_log.elapsed().as_secs_f64(),
                    );
                    last_log = Instant::now();
                }
            }

            state.play_move(mv);
            moves += 1;
            if moves % GARBAGE_INTERVAL == 0 {
                state.add_garbage(thread_rng().gen_range(1, MAX_GARBAGE + 1));
            }
        }
    }
}

#[allow(unused)]
fn print_state(state: &State) {
    for y in (0..20).rev() {
        for x in 0..10 {
            if state.board.occupied(x, y) {
                print!("[]");
            } else {
                print!("..");
            }
        }
        println!();
    }
    
    if let Some(hold) = state.board.hold_piece {
        print!("H: {} Q:", hold.to_char());
    } else {
        print!("H: . Q:");
    }
    
    for piece in state.board.next_queue() {
        print!(" {}", piece.to_char());
    }
    
    println!();
}
