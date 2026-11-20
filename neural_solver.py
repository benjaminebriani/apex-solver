"""Neural leaf evaluation wrapper for apex-solver.

Loads a trained ``TurnValueNetwork`` checkpoint (or uses reproducible dummy
weights for validation), builds the exact 2,706-dim feature vector used by
``train_cloud_v2.py``, injects the predicted CFVs as leaf priors through the
PyO3 ``set_leaf_cfvs`` binding, and runs the CFR solve.
"""

from pathlib import Path

import numpy as np
import torch

import apex_solver
from train_cloud_v2 import TurnValueNetwork

NUM_HANDS = 1326
NUM_FEATURES = 2706

DRIVE_CHECKPOINT = Path("/content/drive/MyDrive/apex-solver/turn_value_net_v2.pt")
CHECKPOINT_CANDIDATES = (
    DRIVE_CHECKPOINT,
    Path("checkpoints/turn_value_net_v2.pt"),
    Path("turn_value_net_v2.pt"),
    Path("checkpoints/turn_value_net.pt"),
    Path("turn_value_net.pt"),
)


def find_checkpoint():
    """Return the first existing checkpoint path, or None if none is present."""
    for path in CHECKPOINT_CANDIDATES:
        if path.is_file():
            return path
    return None


def build_features(board, oop_range, ip_range, pot, effective_stack):
    """Build the 2,706-dim input in the exact train_cloud_v2.py layout.

    Layout: 52 board one-hots, [log(pot/20), log(spr)], 1,326 OOP range
    weights, 1,326 IP range weights. Range order is ``generate_all_hands()``
    order: for c1 in 0..52, for c2 in c1+1..52.
    """
    board = np.asarray(board, dtype=np.int64).reshape(-1)
    oop_range = np.asarray(oop_range, dtype=np.float32).reshape(-1)
    ip_range = np.asarray(ip_range, dtype=np.float32).reshape(-1)
    if oop_range.size != NUM_HANDS or ip_range.size != NUM_HANDS:
        raise ValueError(
            f"ranges must have {NUM_HANDS} weights (got {oop_range.size} and {ip_range.size})"
        )
    if pot <= 0.0 or effective_stack <= 0.0:
        raise ValueError("pot and effective_stack must be positive")

    board_vec = np.zeros(52, dtype=np.float32)
    board_vec[board] = 1.0
    spr = effective_stack / pot
    meta = np.array([np.log(pot / 20.0), np.log(spr)], dtype=np.float32)
    features = np.concatenate([board_vec, meta, oop_range, ip_range]).astype(np.float32)
    if features.size != NUM_FEATURES:
        raise AssertionError(f"expected {NUM_FEATURES} features, got {features.size}")
    return features


def load_value_model(checkpoint_path=None, num_blocks=6, seed=1234, device="cpu"):
    """Load a checkpoint, inferring its architecture from the state dict.

    Supports both the v2 model (6 residual blocks) and the legacy notebook
    model (4 blocks). With ``checkpoint_path=None`` a seeded, untrained model
    is returned so validation can run before training finishes.
    """
    if checkpoint_path is None:
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            model = TurnValueNetwork(num_blocks=num_blocks)
        return model.to(device).eval()

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # older PyTorch without weights_only
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, dict):
        raise ValueError("checkpoint must contain a state_dict")

    in_dim, hidden_dim = state["input_layer.0.weight"].shape
    out_dim = state["head.weight"].shape[0]
    block_indices = {
        int(key.split(".")[1]) for key in state if key.startswith("blocks.")
    }
    inferred_blocks = len(block_indices)
    model = TurnValueNetwork(
        in_dim=int(in_dim),
        hidden_dim=int(hidden_dim),
        out_dim=int(out_dim),
        num_blocks=inferred_blocks,
    )
    model.load_state_dict(state)
    print(
        f"Loaded {path}: in_dim={in_dim}, hidden={hidden_dim}, "
        f"out_dim={out_dim}, blocks={inferred_blocks}",
        flush=True,
    )
    return model.to(device).eval()


class NeuralLeafSolver:
    """Convenience wrapper around ``PySubgameSolver`` with neural leaf priors."""

    def __init__(self, checkpoint_path=None, num_blocks=6, seed=1234, device="cpu"):
        self.device = torch.device(device)
        self.model = load_value_model(
            checkpoint_path=checkpoint_path,
            num_blocks=num_blocks,
            seed=seed,
            device=self.device,
        )
        self.solver = None
        self.last_prior = None

    def predict_cfvs(self, board, oop_range, ip_range, pot, effective_stack):
        """Return pot-normalized (OOP, IP) CFV arrays of length 1,326 each."""
        features = build_features(board, oop_range, ip_range, pot, effective_stack)
        x = torch.from_numpy(features).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prediction = self.model(x).detach().cpu().numpy()[0]
        oop_prior = np.ascontiguousarray(prediction[:NUM_HANDS], dtype=np.float32)
        ip_prior = np.ascontiguousarray(prediction[NUM_HANDS:], dtype=np.float32)
        return oop_prior, ip_prior

    def create_solver(
        self,
        board,
        oop_range,
        ip_range,
        pot,
        effective_stack,
        bet_sizes=(0.5, 1.0),
        lambda_weight=1.0,
    ):
        """Create a solver, predict leaf priors, and inject them at ``lambda_weight``."""
        solver = apex_solver.PySubgameSolver.from_arrays(
            [int(c) for c in board],
            np.asarray(oop_range, dtype=np.float32),
            np.asarray(ip_range, dtype=np.float32),
            float(pot),
            float(effective_stack),
            list(bet_sizes),
        )
        oop_prior, ip_prior = self.predict_cfvs(
            board, oop_range, ip_range, pot, effective_stack
        )
        solver.set_leaf_cfvs(oop_prior, ip_prior, float(lambda_weight))
        self.solver = solver
        self.last_prior = (oop_prior, ip_prior)
        return solver

    def clear_leaf_cfvs(self, solver=None):
        """Fall back to exact showdown equity evaluation."""
        (solver if solver is not None else self.solver).clear_leaf_cfvs()

    def solve(self, iterations, avg_last=1, solver=None):
        return (solver if solver is not None else self.solver).solve(iterations, avg_last)
