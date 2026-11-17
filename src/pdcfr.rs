pub struct PdcfrNode {
    pub num_actions: usize,
    pub regret_sum: Vec<f32>,
    pub prev_inst_regret: Vec<f32>,
    pub strategy_sum: Vec<f32>,
}

impl PdcfrNode {
    pub fn new(num_actions: usize) -> Self {
        Self {
            num_actions,
            regret_sum: vec![0.0; num_actions],
            prev_inst_regret: vec![0.0; num_actions],
            strategy_sum: vec![0.0; num_actions],
        }
    }

    #[inline]
    pub fn get_predictive_strategy(&self, out_strategy: &mut [f32]) {
        let mut normalizing_sum = 0.0f32;

        for a in 0..self.num_actions {
            let predicted_regret = (self.regret_sum[a] + self.prev_inst_regret[a]).max(0.0);
            out_strategy[a] = predicted_regret;
            normalizing_sum += predicted_regret;
        }

        if normalizing_sum > 1e-9 {
            let inv_sum = 1.0 / normalizing_sum;
            for a in 0..self.num_actions {
                out_strategy[a] *= inv_sum;
            }
        } else {
            let uniform = 1.0 / (self.num_actions as f32);
            for a in 0..self.num_actions {
                out_strategy[a] = uniform;
            }
        }
    }

    #[inline]
    pub fn update(&mut self, inst_regrets: &[f32], reach_prob: f32, iteration: u32) {
        let t = iteration as f32;

        let pos_discount = t.powf(1.5) / (t.powf(1.5) + 1.0);
        let neg_discount = 0.5;
        let strat_discount = (t / (t + 1.0)).powi(2);

        let mut current_strat = [0.0f32; 8]; // Stack-allocated for up to 8 bet sizes (zero heap alloc!)
        self.get_predictive_strategy(&mut current_strat[..self.num_actions]);

        for a in 0..self.num_actions {
            let r = inst_regrets[a];
            let discount = if self.regret_sum[a] > 0.0 {
                pos_discount
            } else {
                neg_discount
            };

            self.regret_sum[a] = (self.regret_sum[a] * discount + r).max(0.0);
            self.prev_inst_regret[a] = r;

            self.strategy_sum[a] =
                self.strategy_sum[a] * strat_discount + reach_prob * current_strat[a];
        }
    }

    pub fn get_average_strategy(&self) -> Vec<f32> {
        let sum: f32 = self.strategy_sum.iter().sum();
        if sum > 1e-9 {
            self.strategy_sum.iter().map(|&s| s / sum).collect()
        } else {
            vec![1.0 / (self.num_actions as f32); self.num_actions]
        }
    }
}
