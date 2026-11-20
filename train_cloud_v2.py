import os
import time
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import apex_solver

# Average the noisy final-iteration CFR estimates over this many iterations.
CFV_AVG_LAST = 20

DRIVE_DIR = Path("/content/drive/MyDrive/apex-solver")
LOCAL_DIR = Path("checkpoints")


def get_output_dir():
    """Prefer a mounted Google Drive; fall back to a local checkpoints dir."""
    if DRIVE_DIR.parent.is_dir():
        try:
            DRIVE_DIR.mkdir(parents=True, exist_ok=True)
            probe = DRIVE_DIR / ".write_test"
            probe.touch()
            probe.unlink()
            return DRIVE_DIR
        except OSError as exc:
            print(f"Drive not writable ({exc}); falling back to {LOCAL_DIR}/", flush=True)
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    return LOCAL_DIR


def generate_realistic_range(board_set, tightness):
    """Generates structured poker ranges weighted toward pairs, broadways, suited connectors, and board interaction."""
    rng = np.zeros(1326, dtype=np.float32)
    idx = 0
    for c1 in range(52):
        r1, s1 = c1 // 4, c1 % 4
        for c2 in range(c1 + 1, 52):
            if c1 not in board_set and c2 not in board_set:
                r2, s2 = c2 // 4, c2 % 4
                high, low = max(r1, r2), min(r1, r2)
                suited = 1.0 if s1 == s2 else 0.0
                pair = 1.0 if r1 == r2 else 0.0
                gap = high - low

                strength = (high + low) / 24.0 + 0.25 * pair + 0.12 * suited - 0.03 * max(0, gap - 1)
                prob = np.clip((strength - tightness) * 2.5 + random.uniform(-0.15, 0.25), 0.0, 1.0)
                if prob > 0.15:
                    rng[idx] = prob * random.uniform(0.5, 1.0)
            idx += 1

    total = rng.sum()
    if total > 1e-6:
        rng /= total
    return rng


def solve_single_subgame(job):
    seed, iterations, avg_last = job
    random.seed(seed)
    np.random.seed(seed)

    board = random.sample(range(52), 4)
    board_set = set(board)

    spot_type = random.choice(["3bp", "srp", "bvb"])
    if spot_type == "3bp":
        oop_range = generate_realistic_range(board_set, tightness=0.52)
        ip_range = generate_realistic_range(board_set, tightness=0.48)
        pot = random.uniform(20.0, 60.0)
        spr = random.uniform(0.5, 3.0)
    elif spot_type == "srp":
        oop_range = generate_realistic_range(board_set, tightness=0.40)
        ip_range = generate_realistic_range(board_set, tightness=0.35)
        pot = random.uniform(10.0, 40.0)
        spr = random.uniform(1.0, 6.0)
    else:
        oop_range = generate_realistic_range(board_set, tightness=0.28)
        ip_range = generate_realistic_range(board_set, tightness=0.25)
        pot = random.uniform(8.0, 30.0)
        spr = random.uniform(1.5, 8.0)

    eff_stack = pot * spr
    solver = apex_solver.PySubgameSolver.from_arrays(
        board, oop_range, ip_range, pot, eff_stack, [0.5, 1.0]
    )
    oop_cfvs, ip_cfvs = solver.solve(iterations, avg_last)

    board_vec = np.zeros(52, dtype=np.float32)
    for c in board:
        board_vec[c] = 1.0
    meta_vec = np.array([np.log(pot / 20.0), np.log(spr)], dtype=np.float32)

    features = np.concatenate([board_vec, meta_vec, oop_range, ip_range])
    targets = np.concatenate([oop_cfvs / pot, ip_cfvs / pot])
    return features, targets


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.net(x))


class TurnValueNetwork(nn.Module):
    def __init__(self, in_dim=2706, hidden_dim=1024, out_dim=2652, num_blocks=6):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_blocks)])
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        h = self.input_layer(x)
        for blk in self.blocks:
            h = blk(h)
        return self.head(h)


def run_cloud_pipeline(num_samples=10000, iterations=120, epochs=40, batch_size=128):
    out_dir = get_output_dir()
    dataset_path = out_dir / "turn_dataset_v2.npz"
    weights_path = out_dir / "turn_value_net_v2.pt"
    num_workers = max(1, os.cpu_count() or 4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {num_workers} CPU workers + GPU device: {device}", flush=True)
    print(f"Saving checkpoints to {out_dir}", flush=True)

    t0 = time.time()
    X_list, Y_list = [], []
    tasks = [(1000 + i, iterations, CFV_AVG_LAST) for i in range(num_samples)]

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        for i, (x, y) in enumerate(pool.map(solve_single_subgame, tasks, chunksize=8), 1):
            X_list.append(x)
            Y_list.append(y)
            if i % 250 == 0 or i == num_samples:
                elapsed = time.time() - t0
                rate = i / elapsed
                print(f"[{i}/{num_samples}] solved ({rate:.1f} subgames/sec, {elapsed:.1f}s)", flush=True)
                np.savez_compressed(
                    dataset_path,
                    X=np.array(X_list, dtype=np.float32),
                    Y=np.array(Y_list, dtype=np.float32),
                )

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    Y = torch.tensor(np.array(Y_list), dtype=torch.float32)

    loader = DataLoader(TensorDataset(X, Y), batch_size=batch_size, shuffle=True)
    model = TurnValueNetwork().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    huber = nn.SmoothL1Loss(beta=0.05)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            pred = model(bx)
            loss_cfv = huber(pred, by)

            oop_r = bx[:, 54:54 + 1326]
            ip_r = bx[:, 54 + 1326:]
            total_game_value = (oop_r * pred[:, :1326]).sum(dim=1) + (ip_r * pred[:, 1326:]).sum(dim=1)
            loss = loss_cfv + 0.1 * ((total_game_value - 1.0) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs} | Loss: {total_loss / len(loader):.6f}", flush=True)
            torch.save(model.state_dict(), weights_path)

    torch.save(model.state_dict(), weights_path)
    print(f"TRAINING COMPLETE! Saved {weights_path} and {dataset_path}", flush=True)


if __name__ == "__main__":
    run_cloud_pipeline(num_samples=10000, iterations=120, epochs=40)
