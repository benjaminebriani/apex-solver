"""Validate neural leaf evaluation on held-out Turn subgames.

Compares exact CFR leaf evaluation (lambda=0 baseline) against injected value-net
leaf priors at lambda in {0.0, 0.5, 1.0}. Reports:

  * zero regression at lambda=0.0 (bit-identical CFVs to the exact baseline)
  * CFV MAE versus the exact baseline (pot-normalized)
  * zero-sum residual: (r_oop . CFV_oop + r_ip . CFV_ip) / pot - 1
  * exploitability bound: CFR regret bound summed over players, divided by pot
  * solve runtime (and the extra setup/inference cost for the neural variants)

If no checkpoint is found, reproducible dummy weights are used so the plumbing
can be validated before training completes.
"""

import random
import time

import numpy as np

import apex_solver
from neural_solver import NeuralLeafSolver, find_checkpoint
from train_cloud_v2 import generate_realistic_range

LAMBDAS = (0.0, 0.5, 1.0)
HELD_OUT_SEED_BASE = 900_000  # training uses seeds 1000 + i


def sample_subgame(seed):
    random.seed(seed)
    np.random.seed(seed)
    board = random.sample(range(52), 4)
    board_set = set(board)

    spot_type = random.choice(["3bp", "srp", "bvb"])
    if spot_type == "3bp":
        oop = generate_realistic_range(board_set, tightness=0.52)
        ip = generate_realistic_range(board_set, tightness=0.48)
        pot = random.uniform(20.0, 60.0)
        spr = random.uniform(0.5, 3.0)
    elif spot_type == "srp":
        oop = generate_realistic_range(board_set, tightness=0.40)
        ip = generate_realistic_range(board_set, tightness=0.35)
        pot = random.uniform(10.0, 40.0)
        spr = random.uniform(1.0, 6.0)
    else:
        oop = generate_realistic_range(board_set, tightness=0.28)
        ip = generate_realistic_range(board_set, tightness=0.25)
        pot = random.uniform(8.0, 30.0)
        spr = random.uniform(1.5, 8.0)

    return {
        "board": board,
        "oop": oop,
        "ip": ip,
        "pot": pot,
        "eff_stack": pot * spr,
    }


def make_solver(spot):
    return apex_solver.PySubgameSolver.from_arrays(
        spot["board"], spot["oop"], spot["ip"], spot["pot"], spot["eff_stack"], [0.5, 1.0]
    )


def measure(spot, solver, cfvs, iterations, solve_seconds):
    oop_cfvs, ip_cfvs = cfvs
    pot = spot["pot"]
    zero_sum = (
        float(np.dot(spot["oop"], oop_cfvs)) + float(np.dot(spot["ip"], ip_cfvs))
    ) / pot - 1.0
    bound_oop, bound_ip = solver.get_exploitability_bound(iterations)
    return {
        "zero_sum": zero_sum,
        "exploit": (bound_oop + bound_ip) / pot,
        "solve_seconds": solve_seconds,
    }


def check_api_validation(spot):
    """Bad shapes / lambdas must raise ValueError rather than corrupt state."""
    solver = make_solver(spot)
    good = np.zeros(1326, dtype=np.float32)
    for oop, ip, lam in (
        (np.zeros(10, dtype=np.float32), good, 0.5),
        (good, np.zeros(10, dtype=np.float32), 0.5),
        (good, good, -0.1),
        (good, good, 1.1),
        (good, good, float("nan")),
    ):
        try:
            solver.set_leaf_cfvs(oop, ip, lam)
        except ValueError:
            continue
        raise AssertionError(f"set_leaf_cfvs accepted invalid input (lambda={lam})")
    solver.clear_leaf_cfvs()
    print("API validation checks: PASS", flush=True)


def run(num_subgames=8, iterations=40, avg_last=10):
    checkpoint = find_checkpoint()
    if checkpoint is None:
        print("No checkpoint found; using seeded dummy weights for validation.", flush=True)
    wrapper = NeuralLeafSolver(checkpoint_path=checkpoint, seed=7)
    print(
        f"Held-out validation: {num_subgames} subgames (seeds {HELD_OUT_SEED_BASE}+), "
        f"{iterations} CFR iterations, avg_last={avg_last}",
        flush=True,
    )
    check_api_validation(sample_subgame(HELD_OUT_SEED_BASE))

    totals = {
        lam: {"mae": [], "zero_sum": [], "exploit": [], "solve": [], "setup": []}
        for lam in LAMBDAS
    }
    worst_l0_mae = 0.0

    for i in range(num_subgames):
        spot = sample_subgame(HELD_OUT_SEED_BASE + i)

        exact_solver = make_solver(spot)
        t0 = time.perf_counter()
        exact = exact_solver.solve(iterations, avg_last)
        exact_time = time.perf_counter() - t0
        exact_metrics = measure(spot, exact_solver, exact, iterations, exact_time)

        for lam in LAMBDAS:
            t0 = time.perf_counter()
            solver = wrapper.create_solver(
                spot["board"],
                spot["oop"],
                spot["ip"],
                spot["pot"],
                spot["eff_stack"],
                lambda_weight=lam,
            )
            setup_time = time.perf_counter() - t0
            t0 = time.perf_counter()
            cfvs = solver.solve(iterations, avg_last)
            solve_time = time.perf_counter() - t0

            stacked = np.concatenate([cfvs[0], cfvs[1]])
            target = np.concatenate(exact)
            if not np.isfinite(stacked).all():
                raise AssertionError(f"non-finite CFVs at lambda={lam}, subgame {i}")
            mae = float(np.mean(np.abs(stacked - target)) / spot["pot"])
            m = measure(spot, solver, cfvs, iterations, solve_time)
            if lam == 0.0:
                worst_l0_mae = max(worst_l0_mae, mae)
                if mae != 0.0:
                    raise AssertionError("lambda=0 modified the exact CFVs")

            totals[lam]["mae"].append(mae)
            totals[lam]["zero_sum"].append(abs(m["zero_sum"]))
            totals[lam]["exploit"].append(m["exploit"])
            totals[lam]["solve"].append(solve_time)
            totals[lam]["setup"].append(setup_time)

        print(f"  subgame {i + 1}/{num_subgames} done", flush=True)

    print("\nlambda | CFV MAE | |zero-sum| | exploit bound | solve s | setup s")
    print("-------+---------+-----------+---------------+---------+--------")
    for lam in LAMBDAS:
        t = totals[lam]
        mean = lambda key: float(np.mean(t[key]))
        print(
            f"{lam:6.1f} | {mean('mae'):7.5f} | {mean('zero_sum'):9.5f} | "
            f"{mean('exploit'):13.5f} | {mean('solve'):7.3f} | {mean('setup'):7.3f}"
        )

    print(
        f"\nExact baseline: |zero-sum|={abs(exact_metrics['zero_sum']):.5f}, "
        f"exploit bound={exact_metrics['exploit']:.5f}, "
        f"solve={exact_metrics['solve_seconds']:.3f}s (last subgame)"
    )
    print(f"Zero regression at lambda=0.0: PASS (worst MAE {worst_l0_mae:.8f})")
    print("VALIDATION PASSED", flush=True)


if __name__ == "__main__":
    run()
