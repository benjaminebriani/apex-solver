mod card;
mod pdcfr;
mod tree;

use card::Card;
use std::time::Instant;
use tree::SubgameSolver;

fn main() {
    println!("=== APEX-SOLVER: Multi-Street Turn -> River PDCFR+ Engine ===");

    // 4-card Turn board (As Kd 7h 4c) -> Automatically enumerates all 48 river runouts!
    let board = [
        Card::from_str("As").unwrap(),
        Card::from_str("Kd").unwrap(),
        Card::from_str("7h").unwrap(),
        Card::from_str("4c").unwrap(),
    ];

    let oop_range = "AA,KK,77,44,AKs,AKo,AQs,AQo,AJs,ATs,A5s-A2s,KQs,KJs,QJs,JTs,T9s,98s,87s";
    let ip_range = "JJ-88,AQs,AQo,AJs,AJo,ATs,A9s-A2s,KQs,KQo,KJs,KTs,QJs,QTs,JTs,T9s,98s,87s,76s,65s";

    let build_start = Instant::now();
    // Pot: 60, Effective Stack: 140, Bet sizes: 50% pot and 100% pot
    let mut solver = SubgameSolver::new(&board, oop_range, ip_range, 60.0, 140.0, &[0.5, 1.0]);
    println!(
        "Built Turn->River tree ({} nodes, {} active combos) in {:?}",
        solver.nodes.len(),
        solver.valid_hands.len(),
        build_start.elapsed()
    );

    let solve_start = Instant::now();
    solver.solve(100); // 100 full multi-street iterations across all 48 river runouts
    println!("Solved Turn->River across 48 river cards in {:?}", solve_start.elapsed());

    let root = &solver.nodes[0];
    println!("\nOOP Turn Root Actions: {:?}", root.actions);

    let target_hands = [
        ("Ac", "Ah", "Top Set (AA)"),
        ("Ad", "Qc", "Top Pair Good Kicker (AQ)"),
        ("Qh", "Jh", "Flush Draw + Gutshot (QJh)"),
        ("8s", "7s", "Middle Pair (87s)"),
    ];

    for (c1, c2, label) in target_hands {
        let h_mask = Card::from_str(c1).unwrap().mask() | Card::from_str(c2).unwrap().mask();
        if let Some(idx) = solver.valid_hands.iter().position(|h| h.mask == h_mask) {
            let strat = root.hand_nodes[idx].get_average_strategy();
            println!(
                "{:<30} -> {:?}",
                label,
                strat.iter().map(|x| format!("{:.1}%", x * 100.0)).collect::<Vec<_>>()
            );
        }
    }
}
