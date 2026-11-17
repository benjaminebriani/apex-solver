#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Card(pub u8);

impl Card {
    #[inline(always)]
    pub fn new(rank: u8, suit: u8) -> Self {
        Self(rank * 4 + suit)
    }

    #[inline(always)]
    pub fn mask(self) -> u64 {
        1u64 << self.0
    }

    pub fn from_str(s: &str) -> Option<Self> {
        let bytes = s.as_bytes();
        if bytes.len() != 2 {
            return None;
        }
        let rank = match bytes[0] {
            b'2' => 0, b'3' => 1, b'4' => 2, b'5' => 3,
            b'6' => 4, b'7' => 5, b'8' => 6, b'9' => 7,
            b'T' | b't' => 8, b'J' | b'j' => 9,
            b'Q' | b'q' => 10, b'K' | b'k' => 11, b'A' | b'a' => 12,
            _ => return None,
        };
        let suit = match bytes[1] {
            b'c' | b'C' => 0,
            b'd' | b'D' => 1,
            b'h' | b'H' => 2,
            b's' | b'S' => 3,
            _ => return None,
        };
        Some(Self::new(rank, suit))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hand {
    pub c1: Card,
    pub c2: Card,
    pub mask: u64,
}

impl Hand {
    #[inline(always)]
    pub fn new(c1: Card, c2: Card) -> Self {
        Self {
            c1,
            c2,
            mask: c1.mask() | c2.mask(),
        }
    }

    #[inline(always)]
    pub fn conflicts_with(self, other_mask: u64) -> bool {
        (self.mask & other_mask) != 0
    }
}

pub fn generate_all_hands() -> Vec<Hand> {
    let mut hands = Vec::with_capacity(1326);
    for i in 0..52u8 {
        for j in (i + 1)..52u8 {
            hands.push(Hand::new(Card(i), Card(j)));
        }
    }
    hands
}

#[inline]
pub fn evaluate_7cards(mask: u64) -> u32 {
    let mut ranks = [0u8; 13];
    let mut suits = [0u16; 4];

    let mut m = mask;
    while m != 0 {
        let card = m.trailing_zeros() as usize;
        m &= m - 1;
        let r = card >> 2;
        let s = card & 3;
        ranks[r] += 1;
        suits[s] |= 1 << r;
    }

    for s in 0..4 {
        let suit_ranks = suits[s];
        if suit_ranks.count_ones() >= 5 {
            if let Some(sf_high) = find_straight(suit_ranks) {
                return (8 << 20) | (sf_high as u32);
            }
            return (5 << 20) | top_5_bits(suit_ranks);
        }
    }

    let mut rank_bits = 0u16;
    let mut quads = None;
    let mut trips = [0u32; 2];
    let mut trip_count = 0;
    let mut pairs = [0u32; 3];
    let mut pair_count = 0;

    for r in (0..13).rev() {
        if ranks[r] > 0 {
            rank_bits |= 1 << r;
            match ranks[r] {
                4 => quads = Some(r as u32),
                3 => {
                    trips[trip_count] = r as u32;
                    trip_count += 1;
                }
                2 => {
                    pairs[pair_count] = r as u32;
                    pair_count += 1;
                }
                _ => {}
            }
        }
    }

    if let Some(q) = quads {
        let kicker = (0..13).rev().find(|&r| r as u32 != q && ranks[r] > 0).unwrap_or(0) as u32;
        return (7 << 20) | (q << 4) | kicker;
    }

    if trip_count > 0 && (trip_count > 1 || pair_count > 0) {
        let t = trips[0];
        let p = if trip_count > 1 { trips[1] } else { pairs[0] };
        return (6 << 20) | (t << 4) | p;
    }

    if let Some(st_high) = find_straight(rank_bits) {
        return (4 << 20) | (st_high as u32);
    }

    if trip_count > 0 {
        let t = trips[0];
        let kickers = top_n_kickers(rank_bits & !(1 << t), 2);
        return (3 << 20) | (t << 8) | kickers;
    }

    if pair_count >= 2 {
        let p1 = pairs[0];
        let p2 = pairs[1];
        let kicker = top_n_kickers(rank_bits & !(1 << p1) & !(1 << p2), 1);
        return (2 << 20) | (p1 << 8) | (p2 << 4) | kicker;
    }

    if pair_count == 1 {
        let p = pairs[0];
        let kickers = top_n_kickers(rank_bits & !(1 << p), 3);
        return (1 << 20) | (p << 12) | kickers;
    }

    top_5_bits(rank_bits)
}

#[inline(always)]
fn find_straight(bits: u16) -> Option<u8> {
    for high in (4..13).rev() {
        let mask = 0x1Fu16 << (high - 4);
        if (bits & mask) == mask {
            return Some(high);
        }
    }
    if (bits & 0x100F) == 0x100F {
        return Some(3);
    }
    None
}

#[inline(always)]
fn top_5_bits(mut bits: u16) -> u32 {
    let mut score = 0u32;
    for _ in 0..5 {
        if bits == 0 { break; }
        let highest = 15 - bits.leading_zeros();
        score = (score << 4) | highest;
        bits &= !(1 << highest);
    }
    score
}

#[inline(always)]
fn top_n_kickers(mut bits: u16, n: usize) -> u32 {
    let mut score = 0u32;
    for _ in 0..n {
        if bits == 0 { break; }
        let highest = 15 - bits.leading_zeros();
        score = (score << 4) | highest;
        bits &= !(1 << highest);
    }
    score
}
/// Parses standard poker range strings (e.g., "AA,KK,AKs,AKo,TT-88,A5s-A2s") into 1,326 weights
pub fn parse_range(range_str: &str, all_hands: &[Hand]) -> Vec<f32> {
    let mut weights = vec![0.0f32; all_hands.len()];
    if range_str.trim().eq_ignore_ascii_case("all") || range_str.trim().is_empty() {
        return vec![1.0f32; all_hands.len()];
    }

    for part in range_str.split(',') {
        let token = part.trim();
        if token.is_empty() {
            continue;
        }

        // Check for optional weight syntax like "AKs:0.5"
        let (hand_part, weight) = if let Some((h, w)) = token.split_once(':') {
            (h.trim(), w.trim().parse::<f32>().unwrap_or(1.0))
        } else {
            (token, 1.0)
        };

        if let Some((start_str, end_str)) = hand_part.split_once('-') {
            apply_range_span(start_str.trim(), end_str.trim(), weight, all_hands, &mut weights);
        } else {
            apply_single_combo(hand_part, weight, all_hands, &mut weights);
        }
    }
    weights
}

fn char_to_rank(c: u8) -> Option<u8> {
    match c {
        b'2' => Some(0), b'3' => Some(1), b'4' => Some(2), b'5' => Some(3),
        b'6' => Some(4), b'7' => Some(5), b'8' => Some(6), b'9' => Some(7),
        b'T' | b't' => Some(8), b'J' | b'j' => Some(9),
        b'Q' | b'q' => Some(10), b'K' | b'k' => Some(11), b'A' | b'a' => Some(12),
        _ => None,
    }
}

fn apply_single_combo(token: &str, weight: f32, all_hands: &[Hand], weights: &mut [f32]) {
    let b = token.as_bytes();
    if b.len() < 2 {
        return;
    }
    let r1 = match char_to_rank(b[0]) { Some(r) => r, None => return };
    let r2 = match char_to_rank(b[1]) { Some(r) => r, None => return };
    let suited_filter = if b.len() >= 3 {
        match b[2] {
            b's' | b'S' => Some(true),
            b'o' | b'O' => Some(false),
            _ => None,
        }
    } else {
        None
    };

    for (idx, h) in all_hands.iter().enumerate() {
        let hr1 = h.c1.0 >> 2;
        let hr2 = h.c2.0 >> 2;
        let hs1 = h.c1.0 & 3;
        let hs2 = h.c2.0 & 3;

        let ranks_match = (hr1 == r1 && hr2 == r2) || (hr1 == r2 && hr2 == r1);
        if !ranks_match {
            continue;
        }

        let is_suited = hs1 == hs2;
        match suited_filter {
            Some(true) if !is_suited => continue,
            Some(false) if is_suited => continue,
            _ => {}
        }
        weights[idx] = weight;
    }
}

fn apply_range_span(start: &str, end: &str, weight: f32, all_hands: &[Hand], weights: &mut [f32]) {
    let sb = start.as_bytes();
    let eb = end.as_bytes();
    if sb.len() < 2 || eb.len() < 2 {
        return;
    }
    let sr1 = match char_to_rank(sb[0]) { Some(r) => r, None => return };
    let sr2 = match char_to_rank(sb[1]) { Some(r) => r, None => return };
    let er1 = match char_to_rank(eb[0]) { Some(r) => r, None => return };
    let er2 = match char_to_rank(eb[1]) { Some(r) => r, None => return };

    // Pair span like "TT-77" or "77-TT"
    if sr1 == sr2 && er1 == er2 {
        let low = sr1.min(er1);
        let high = sr1.max(er1);
        for r in low..=high {
            let pair_str = format!("{}{}", rank_to_char(r), rank_to_char(r));
            apply_single_combo(&pair_str, weight, all_hands, weights);
        }
    } else if sr1 == er1 {
        // Kicker span like "A5s-A2s" or "KQo-KTo"
        let suffix = if sb.len() >= 3 { sb[2] as char } else { ' ' };
        let low = sr2.min(er2);
        let high = sr2.max(er2);
        for r in low..=high {
            let combo = if suffix == 's' || suffix == 'o' {
                format!("{}{}{}", rank_to_char(sr1), rank_to_char(r), suffix)
            } else {
                format!("{}{}", rank_to_char(sr1), rank_to_char(r))
            };
            apply_single_combo(&combo, weight, all_hands, weights);
        }
    }
}

fn rank_to_char(r: u8) -> char {
    match r {
        0 => '2', 1 => '3', 2 => '4', 3 => '5', 4 => '6', 5 => '7',
        6 => '8', 7 => '9', 8 => 'T', 9 => 'J', 10 => 'Q', 11 => 'K',
        _ => 'A',
    }
}
