import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import apex_solver


def sample_random_subgame(iterations=120):
    board = random.sample(range(52), 4)
    board_set = set(board)

    oop_range = np.zeros(1326, dtype=np.float32)
    ip_range = np.zeros(1326, dtype=np.float32)

    idx = 0
    for c1 in range(52):
        for c2 in range(c1 + 1, 52):
            if c1 not in board_set and c2 not in board_set:
                if random.random() < 0.35:
                    oop_range[idx] = random.uniform(0.1, 1.0)
                if random.random() < 0.35:
                    ip_range[idx] = random.uniform(0.1, 1.0)
            idx += 1

    if oop_range.sum() > 0:
        oop_range /= oop_range.sum()
    if ip_range.sum() > 0:
        ip_range /= ip_range.sum()

    pot = random.uniform(10.0, 100.0)
    spr = random.uniform(0.5, 6.0)
    eff_stack = pot * spr

    solver = apex_solver.PySubgameSolver.from_arrays(
        board, oop_range, ip_range, pot, eff_stack, [0.5, 1.0]
    )
    oop_cfvs, ip_cfvs = solver.solve(iterations)

    norm_oop_cfvs = oop_cfvs / pot
    norm_ip_cfvs = ip_cfvs / pot

    board_vec = np.zeros(52, dtype=np.float32)
    for c in board:
        board_vec[c] = 1.0
    meta_vec = np.array([np.log(pot / 20.0), np.log(spr)], dtype=np.float32)

    features = np.concatenate([board_vec, meta_vec, oop_range, ip_range])
    targets = np.concatenate([norm_oop_cfvs, norm_ip_cfvs])
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
    def __init__(self, in_dim=2706, hidden_dim=1024, out_dim=2652, num_blocks=4):
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


def train_cloud(num_samples=2000, epochs=25, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    t0 = time.time()
    X_list, Y_list = [], []
    for i in range(num_samples):
        x, y = sample_random_subgame(iterations=120)
        X_list.append(x)
        Y_list.append(y)
        if (i + 1) % 250 == 0:
            print(f"Generated {i + 1}/{num_samples} solved Turn subgames in {time.time() - t0:.1f}s")

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    Y = torch.tensor(np.array(Y_list), dtype=torch.float32)

    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
            pred_oop_v = pred[:, :1326]
            pred_ip_v = pred[:, 1326:]
            total_game_value = (oop_r * pred_oop_v).sum(dim=1) + (ip_r * pred_ip_v).sum(dim=1)
            loss_zerosum = ((total_game_value - 1.0) ** 2).mean()

            loss = loss_cfv + 0.1 * loss_zerosum
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs} | Loss: {total_loss / len(loader):.6f}")

    torch.save(model.state_dict(), "turn_value_net.pt")
    print("Saved trained weights to turn_value_net.pt")


if __name__ == "__main__":
    train_cloud(num_samples=2000, epochs=25)
