use crate::card::{evaluate_7cards, generate_all_hands, parse_range, Card, Hand};
use crate::pdcfr::PdcfrNode;
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub enum Action {
    Fold,
    Check,
    Call,
    Bet(f32),
    Raise(f32),
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Street {
    Turn,
    River,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TerminalType {
    None,
    Fold(usize),
    Showdown,
    ChanceToRiver,
}

pub struct TreeNode {
    pub street: Street,
    pub player: usize,
    pub pot: f32,
    pub actions: Vec<Action>,
    pub children: Vec<usize>,
    pub hand_nodes: Vec<PdcfrNode>,
    pub is_terminal: bool,
    pub terminal_type: TerminalType,
}

pub struct SubgameSolver {
    pub board: Vec<Card>,
    pub board_mask: u64,
    pub all_hands: Vec<Hand>,
    pub valid_hands: Vec<Hand>,
    pub valid_to_all: Vec<usize>,
    pub oop_range: Vec<f32>,
    pub ip_range: Vec<f32>,
    pub hand_cards: Vec<(usize, usize)>,
    pub nodes: Vec<TreeNode>,
    pub river_scores: Vec<Option<Vec<u32>>>,
    pub river_sorted: Vec<Option<Vec<usize>>>,
    /// Pot-normalized CFVs from the value net, in `generate_all_hands()` order:
    /// (OOP, IP). Each is 1,326 floats.
    pub leaf_cfvs: Option<(Vec<f32>, Vec<f32>)>,
    /// Weight applied to the neural prior at Showdown leaves (0.0 = exact equity).
    pub leaf_lambda: f32,
}

impl SubgameSolver {
    pub fn new(
        board: &[Card],
        oop_range_str: &str,
        ip_range_str: &str,
        starting_pot: f32,
        effective_stack: f32,
        bet_sizes: &[f32],
    ) -> Self {
        let all_hands = generate_all_hands();
        let raw_oop = parse_range(oop_range_str, &all_hands);
        let raw_ip = parse_range(ip_range_str, &all_hands);
        Self::from_ranges(board, &raw_oop, &raw_ip, starting_pot, effective_stack, bet_sizes)
    }

    pub fn from_ranges(
        board: &[Card],
        raw_oop: &[f32],
        raw_ip: &[f32],
        starting_pot: f32,
        effective_stack: f32,
        bet_sizes: &[f32],
    ) -> Self {
        let board_mask = board.iter().fold(0u64, |acc, c| acc | c.mask());
        let all_hands = generate_all_hands();

        let mut valid_hands = Vec::new();
        let mut valid_to_all = Vec::new();
        let mut oop_range = Vec::new();
        let mut ip_range = Vec::new();

        for (idx, h) in all_hands.iter().enumerate() {
            if !h.conflicts_with(board_mask) && (raw_oop[idx] > 0.0 || raw_ip[idx] > 0.0) {
                valid_hands.push(*h);
                valid_to_all.push(idx);
                oop_range.push(raw_oop[idx]);
                ip_range.push(raw_ip[idx]);
            }
        }

        let hand_cards: Vec<(usize, usize)> = valid_hands
            .iter()
            .map(|h| (h.c1.0 as usize, h.c2.0 as usize))
            .collect();

        let start_street = if board.len() == 4 {
            Street::Turn
        } else {
            Street::River
        };

        let mut river_scores = vec![None; 52];
        let mut river_sorted = vec![None; 52];

        if start_street == Street::River {
            let scores: Vec<u32> = valid_hands
                .par_iter()
                .map(|h| evaluate_7cards(h.mask | board_mask))
                .collect();
            let mut sorted: Vec<usize> = (0..valid_hands.len()).collect();
            sorted.sort_by_key(|&i| scores[i]);
            let first_card = board.first().unwrap().0 as usize;
            river_scores[first_card] = Some(scores);
            river_sorted[first_card] = Some(sorted);
        } else {
            let precomputed: Vec<(usize, Vec<u32>, Vec<usize>)> = (0..52usize)
                .into_par_iter()
                .filter(|&c| (board_mask & (1u64 << c)) == 0)
                .map(|c| {
                    let full_mask = board_mask | (1u64 << c);
                    let scores: Vec<u32> = valid_hands
                        .iter()
                        .map(|h| {
                            if h.conflicts_with(1u64 << c) {
                                0
                            } else {
                                evaluate_7cards(h.mask | full_mask)
                            }
                        })
                        .collect();
                    let mut sorted: Vec<usize> = (0..valid_hands.len()).collect();
                    sorted.sort_by_key(|&i| scores[i]);
                    (c, scores, sorted)
                })
                .collect();

            for (c, sc, so) in precomputed {
                river_scores[c] = Some(sc);
                river_sorted[c] = Some(so);
            }
        }

        let mut solver = Self {
            board: board.to_vec(),
            board_mask,
            all_hands,
            valid_hands,
            valid_to_all,
            oop_range,
            ip_range,
            hand_cards,
            nodes: Vec::new(),
            river_scores,
            river_sorted,
            leaf_cfvs: None,
            leaf_lambda: 0.0,
        };

        let init_stacks = [effective_stack, effective_stack];
        solver.build_tree(start_street, 0, starting_pot, init_stacks, 0.0, bet_sizes, false);
        solver
    }

    fn build_tree(
        &mut self,
        street: Street,
        player: usize,
        pot: f32,
        stacks: [f32; 2],
        to_call: f32,
        bet_sizes: &[f32],
        facing_check: bool,
    ) -> usize {
        let node_idx = self.nodes.len();
        let num_hands = self.valid_hands.len();
        self.nodes.push(TreeNode {
            street,
            player,
            pot,
            actions: Vec::new(),
            children: Vec::new(),
            hand_nodes: Vec::new(),
            is_terminal: false,
            terminal_type: TerminalType::None,
        });

        let mut actions = Vec::new();
        let mut children = Vec::new();
        let my_stack = stacks[player];

        if to_call > 0.0 {
            actions.push(Action::Fold);
            children.push(self.add_terminal(street, pot, TerminalType::Fold(player)));

            let actual_call = to_call.min(my_stack);
            let mut call_stacks = stacks;
            call_stacks[player] -= actual_call;

            actions.push(Action::Call);
            if street == Street::Turn {
                let river_root = self.build_tree(
                    Street::River,
                    0,
                    pot + actual_call,
                    call_stacks,
                    0.0,
                    bet_sizes,
                    false,
                );
                children.push(self.add_chance_node(pot + actual_call, river_root));
            } else {
                children.push(self.add_terminal(street, pot + actual_call, TerminalType::Showdown));
            }

            let remaining_after_call = my_stack - actual_call;
            let opp_stack = stacks[1 - player];
            let max_raise = remaining_after_call.min(opp_stack);

            if max_raise > 0.01 {
                for &sz in bet_sizes {
                    let raise_amt = ((pot + actual_call) * sz).min(max_raise);
                    if raise_amt > 0.01 {
                        let mut raise_stacks = stacks;
                        raise_stacks[player] -= actual_call + raise_amt;
                        actions.push(Action::Raise(sz));
                        children.push(self.build_tree(
                            street,
                            1 - player,
                            pot + actual_call + raise_amt,
                            raise_stacks,
                            raise_amt,
                            bet_sizes,
                            false,
                        ));
                    }
                }
            }
        } else {
            let [s_oop, s_ip] = stacks;
            actions.push(Action::Check);
            if facing_check {
                if street == Street::Turn {
                    let river_root = self.build_tree(
                        Street::River,
                        0,
                        pot,
                        stacks,
                        0.0,
                        bet_sizes,
                        false,
                    );
                    children.push(self.add_chance_node(pot, river_root));
                } else {
                    children.push(self.add_terminal(street, pot, TerminalType::Showdown));
                }
            } else {
                children.push(self.build_tree(
                    street,
                    1 - player,
                    pot,
                    stacks,
                    0.0,
                    bet_sizes,
                    true,
                ));
            }

            let max_bet = s_oop.min(s_ip);
            if max_bet > 0.01 {
                for &sz in bet_sizes {
                    let bet_amt = (pot * sz).min(max_bet);
                    if bet_amt > 0.01 {
                        let mut bet_stacks = stacks;
                        bet_stacks[player] -= bet_amt;
                        actions.push(Action::Bet(sz));
                        children.push(self.build_tree(
                            street,
                            1 - player,
                            pot + bet_amt,
                            bet_stacks,
                            bet_amt,
                            bet_sizes,
                            false,
                        ));
                    }
                }
            }
        }

        let num_actions = actions.len();
        self.nodes[node_idx].actions = actions;
        self.nodes[node_idx].children = children;
        self.nodes[node_idx].hand_nodes = (0..num_hands)
            .map(|_| PdcfrNode::new(num_actions))
            .collect();
        node_idx
    }

    fn add_terminal(&mut self, street: Street, pot: f32, terminal_type: TerminalType) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(TreeNode {
            street,
            player: 0,
            pot,
            actions: Vec::new(),
            children: Vec::new(),
            hand_nodes: Vec::new(),            is_terminal: true,
            terminal_type,
        });
        idx
    }

    fn add_chance_node(&mut self, pot: f32, river_root: usize) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(TreeNode {
            street: Street::Turn,
            player: 0,
            pot,
            actions: Vec::new(),
            children: vec![river_root],
            hand_nodes: Vec::new(),
            is_terminal: true,
            terminal_type: TerminalType::ChanceToRiver,
        });
        idx
    }

    /// Solves and returns the CFV estimate from the final CFR iteration.
    pub fn solve(&mut self, iterations: u32) -> (Vec<f32>, Vec<f32>) {
        self.solve_with_averaging(iterations, 1)
    }

    /// Solves and averages each player's CFVs over the last `avg_last` iterations.
    ///
    /// Each player is averaged only over the iterations in the window where
    /// that player was the traverser (i.e. over its own fresh CFR estimates),
    /// so odd window sizes do not bias one side. `avg_last` is clamped to
    /// `[1, iterations]`; when the window contains no update for a player, its
    /// most recent estimate is returned, matching the single-iteration case.
    pub fn solve_with_averaging(
        &mut self,
        iterations: u32,
        avg_last: u32,
    ) -> (Vec<f32>, Vec<f32>) {
        let num_hands = self.valid_hands.len();
        let avg_last = avg_last.max(1).min(iterations.max(1));
        let window_start = iterations.saturating_sub(avg_last) + 1;
        let initial_river = if self.board.len() == 5 {
            self.board.first().unwrap().0 as usize
        } else {
            0
        };
        let mut last_oop_cfvs = vec![0.0f32; num_hands];
        let mut last_ip_cfvs = vec![0.0f32; num_hands];
        let mut sum_oop = vec![0.0f32; num_hands];
        let mut sum_ip = vec![0.0f32; num_hands];
        let mut oop_updates = 0u32;
        let mut ip_updates = 0u32;

        for iter in 1..=iterations {
            let traverser = (iter % 2) as usize;
            let oop_init = self.oop_range.clone();
            let ip_init = self.ip_range.clone();
            let cfvs = self.cfr_traverse(
                0,
                traverser,
                &oop_init,
                &ip_init,
                0.0,
                0.0,
                iter,
                initial_river,
            );
            if traverser == 0 {
                last_oop_cfvs = cfvs;
            } else {
                last_ip_cfvs = cfvs;
            }

            if iter >= window_start {
                if traverser == 0 {
                    for h in 0..num_hands {
                        sum_oop[h] += last_oop_cfvs[h];
                    }
                    oop_updates += 1;
                } else {
                    for h in 0..num_hands {
                        sum_ip[h] += last_ip_cfvs[h];
                    }
                    ip_updates += 1;
                }
            }
        }

        let inv_oop = 1.0 / oop_updates.max(1) as f32;
        let inv_ip = 1.0 / ip_updates.max(1) as f32;
        let mut full_oop_cfvs = vec![0.0f32; 1326];
        let mut full_ip_cfvs = vec![0.0f32; 1326];
        for (v_idx, &all_idx) in self.valid_to_all.iter().enumerate() {
            let oop_val = if oop_updates > 0 {
                sum_oop[v_idx] * inv_oop
            } else {
                last_oop_cfvs[v_idx]
            };
            let ip_val = if ip_updates > 0 {
                sum_ip[v_idx] * inv_ip
            } else {
                last_ip_cfvs[v_idx]
            };
            full_oop_cfvs[all_idx] = oop_val;
            full_ip_cfvs[all_idx] = ip_val;
        }
        (full_oop_cfvs, full_ip_cfvs)
    }

    fn cfr_traverse(
        &mut self,
        node_idx: usize,
        traverser: usize,
        oop_reach: &[f32],
        ip_reach: &[f32],
        oop_invested: f32,
        ip_invested: f32,
        iter: u32,
        river_card_idx: usize,
    ) -> Vec<f32> {
        let is_terminal = self.nodes[node_idx].is_terminal;
        if is_terminal {
            let term_type = self.nodes[node_idx].terminal_type;
            let pot = self.nodes[node_idx].pot;
            match term_type {
                TerminalType::Fold(_) | TerminalType::Showdown => {
                    let opp_reach = if traverser == 0 { ip_reach } else { oop_reach };
                    let my_inv = if traverser == 0 { oop_invested } else { ip_invested };
                    return self.eval_terminal(term_type, pot, traverser, opp_reach, my_inv, river_card_idx);
                }
                TerminalType::ChanceToRiver => {
                    let river_root = *self.nodes[node_idx].children.first().unwrap();
                    let num_hands = self.valid_hands.len();
                    let mut total_cfvs = vec![0.0f32; num_hands];
                    let mut valid_rivers = 0.0f32;

                    for c in 0..52usize {
                        if (self.board_mask & (1u64 << c)) != 0 {
                            continue;
                        }
                        valid_rivers += 1.0;
                        let c_mask = 1u64 << c;
                        let mut next_oop = oop_reach.to_vec();
                        let mut next_ip = ip_reach.to_vec();
                        for h in 0..num_hands {
                            if self.valid_hands[h].conflicts_with(c_mask) {
                                next_oop[h] = 0.0;
                                next_ip[h] = 0.0;
                            }
                        }
                        let branch_cfvs = self.cfr_traverse(
                            river_root,
                            traverser,
                            &next_oop,
                            &next_ip,
                            oop_invested,
                            ip_invested,
                            iter,
                            c,
                        );
                        for h in 0..num_hands {
                            total_cfvs[h] += branch_cfvs[h];
                        }
                    }

                    if valid_rivers > 0.0 {
                        let inv_r = 1.0 / valid_rivers;
                        for h in 0..num_hands {
                            total_cfvs[h] *= inv_r;
                        }
                    }
                    return total_cfvs;
                }
                TerminalType::None => unreachable!(),
            }
        }

        let player = self.nodes[node_idx].player;
        let num_actions = self.nodes[node_idx].actions.len();
        let num_hands = self.valid_hands.len();
        let actions = self.nodes[node_idx].actions.clone();
        let children = self.nodes[node_idx].children.clone();
        let pot = self.nodes[node_idx].pot;

        let mut strategies = vec![0.0f32; num_hands * num_actions];
        for h in 0..num_hands {
            self.nodes[node_idx].hand_nodes[h]
                .get_predictive_strategy(&mut strategies[h * num_actions..(h + 1) * num_actions]);
        }

        let mut action_cfvs = vec![vec![0.0f32; num_hands]; num_actions];
        let mut node_cfvs = vec![0.0f32; num_hands];

        for a in 0..num_actions {
            let mut next_oop = oop_reach.to_vec();
            let mut next_ip = ip_reach.to_vec();

            if player == 0 {
                for h in 0..num_hands {
                    next_oop[h] *= strategies[h * num_actions + a];
                }
            } else {
                for h in 0..num_hands {
                    next_ip[h] *= strategies[h * num_actions + a];
                }
            }

            let added = match actions[a] {
                Action::Fold | Action::Check => 0.0,
                Action::Call => (oop_invested - ip_invested).abs(),
                Action::Bet(sz) => pot * sz,
                Action::Raise(sz) => {
                    let to_call = (oop_invested - ip_invested).abs();
                    to_call + (pot + to_call) * sz
                }
            };

            let (next_oop_inv, next_ip_inv) = if player == 0 {
                (oop_invested + added, ip_invested)
            } else {
                (oop_invested, ip_invested + added)
            };

            action_cfvs[a] = self.cfr_traverse(
                children[a],
                traverser,
                &next_oop,
                &next_ip,
                next_oop_inv,
                next_ip_inv,
                iter,
                river_card_idx,
            );

            if player == traverser {
                for h in 0..num_hands {
                    node_cfvs[h] += strategies[h * num_actions + a] * action_cfvs[a][h];
                }
            } else {
                for h in 0..num_hands {
                    node_cfvs[h] += action_cfvs[a][h];
                }
            }
        }

        if player == traverser {
            let my_reach = if player == 0 { oop_reach } else { ip_reach };
            let mut regrets = [0.0f32; 8];
            for h in 0..num_hands {
                if my_reach[h] == 0.0 {
                    continue;
                }
                for a in 0..num_actions {
                    regrets[a] = action_cfvs[a][h] - node_cfvs[h];
                }
                self.nodes[node_idx].hand_nodes[h].update(
                    &regrets[..num_actions],
                    my_reach[h],
                    iter,
                );
            }
        }

        node_cfvs
    }

    fn eval_terminal(
        &self,
        term_type: TerminalType,
        pot: f32,
        traverser: usize,
        opp_reach: &[f32],
        my_invested: f32,
        river_card_idx: usize,
    ) -> Vec<f32> {
        let num_hands = self.valid_hands.len();
        let mut evs = vec![0.0f32; num_hands];
        let mut total_opp = 0.0f32;
        let mut card_opp = [0.0f32; 52];

        for j in 0..num_hands {
            let w = opp_reach[j];
            if w == 0.0 {
                continue;
            }
            total_opp += w;
            let (c1, c2) = self.hand_cards[j];
            card_opp[c1] += w;
            card_opp[c2] += w;
        }

        match term_type {
            TerminalType::Fold(folder) => {
                let payoff = if folder == traverser {
                    -my_invested
                } else {
                    pot - my_invested
                };
                for i in 0..num_hands {
                    let (c1, c2) = self.hand_cards[i];
                    let valid_opp = total_opp - card_opp[c1] - card_opp[c2] + opp_reach[i];
                    evs[i] = valid_opp * payoff;
                }
            }
            TerminalType::Showdown => {
                let scores = self.river_scores[river_card_idx].as_ref().unwrap();
                let sorted = self.river_sorted[river_card_idx].as_ref().unwrap();
                let river_mask = if self.board.len() == 4 {
                    1u64 << river_card_idx
                } else {
                    0
                };

                let mut running_opp = 0.0f32;
                let mut running_card = [0.0f32; 52];

                let mut idx = 0;
                while idx < num_hands {
                    let h_idx = sorted[idx];
                    if river_mask != 0 && self.valid_hands[h_idx].conflicts_with(river_mask) {
                        idx += 1;
                        continue;
                    }
                    let current_score = scores[h_idx];

                    let mut tie_end = idx + 1;
                    while tie_end < num_hands {
                        let next_h = sorted[tie_end];
                        if river_mask != 0 && self.valid_hands[next_h].conflicts_with(river_mask) {
                            tie_end += 1;
                            continue;
                        }
                        if scores[next_h] != current_score {
                            break;
                        }
                        tie_end += 1;
                    }

                    let mut tie_opp = 0.0f32;
                    let mut tie_card = [0.0f32; 52];
                    for k in idx..tie_end {
                        let tk = sorted[k];
                        if river_mask != 0 && self.valid_hands[tk].conflicts_with(river_mask) {
                            continue;
                        }
                        let w = opp_reach[tk];
                        if w > 0.0 {
                            tie_opp += w;
                            let (c1, c2) = self.hand_cards[tk];
                            tie_card[c1] += w;
                            tie_card[c2] += w;
                        }
                    }

                    for k in idx..tie_end {
                        let tk = sorted[k];
                        if river_mask != 0 && self.valid_hands[tk].conflicts_with(river_mask) {
                            continue;
                        }
                        let (c1, c2) = self.hand_cards[tk];
                        let valid_total = total_opp - card_opp[c1] - card_opp[c2] + opp_reach[tk];
                        let valid_win = running_opp - running_card[c1] - running_card[c2];
                        let valid_tie = tie_opp - tie_card[c1] - tie_card[c2] + opp_reach[tk];
                        let valid_lose = (valid_total - valid_win - valid_tie).max(0.0);

                        let win_payoff = pot - my_invested;
                        let tie_payoff = (pot * 0.5) - my_invested;
                        let lose_payoff = -my_invested;

                        evs[tk] = valid_win * win_payoff
                            + valid_tie * tie_payoff
                            + valid_lose * lose_payoff;
                    }

                    running_opp += tie_opp;
                    for c in 0..52 {
                        running_card[c] += tie_card[c];
                    }
                    idx = tie_end;
                }

                // Neural leaf evaluation: blend the exact showdown EV with the
                // value net's pot-normalized CFV estimate for this traverser.
                if let Some((oop_prior, ip_prior)) = &self.leaf_cfvs {
                    let lambda = self.leaf_lambda.clamp(0.0, 1.0);
                    if lambda > 0.0 {
                        let prior = if traverser == 0 { oop_prior } else { ip_prior };
                        for i in 0..num_hands {
                            let exact_ev = evs[i];
                            let prior_ev = pot * prior[self.valid_to_all[i]];
                            evs[i] = (1.0 - lambda) * exact_ev + lambda * prior_ev;
                        }
                    }
                }
            }
            _ => {}
        }
        evs
    }

    /// CFR regret-based exploitability bound per player, in chip units.
    ///
    /// Uses the standard `sum(reach * max(0, regret)) / iterations` bound at
    /// each decision node. Regret discounting makes this an approximation, but
    /// it is a useful relative measure when comparing leaf configurations.
    #[allow(dead_code)]
    pub fn get_exploitability_bound(&self, iterations: u32) -> (f32, f32) {
        let num_hands = self.valid_hands.len();
        let mut totals = [0.0f32; 2];
        for node in &self.nodes {
            if node.is_terminal || node.hand_nodes.len() != num_hands {
                continue;
            }
            let p = node.player;
            let reach = if p == 0 { &self.oop_range } else { &self.ip_range };
            for h in 0..num_hands {
                if reach[h] == 0.0 {
                    continue;
                }
                let max_regret = node.hand_nodes[h]
                    .regret_sum
                    .iter()
                    .copied()
                    .fold(0.0f32, |a, b| a.max(b));
                totals[p] += reach[h] * max_regret.max(0.0);
            }
        }
        let inv = 1.0 / iterations.max(1) as f32;
        (totals[0] * inv, totals[1] * inv)
    }
}
