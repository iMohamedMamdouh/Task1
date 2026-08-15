"""Run the experiment grid. Existing result files are skipped so the grid can be resumed."""

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

GRID = [
    # name, points per class, gamma, seed
    ("dense_ce", 0, 0.0, 0),
    ("points1_ce", 1, 0.0, 0),
    ("points5_ce", 5, 0.0, 0),
    ("points20_ce", 20, 0.0, 0),
    ("points100_ce", 100, 0.0, 0),
    ("points1_focal2", 1, 2.0, 0),
    ("points5_focal2", 5, 2.0, 0),
    ("points20_focal2", 20, 2.0, 0),
    ("points5_ce_seed1", 5, 0.0, 1),
    ("points5_ce_seed2", 5, 0.0, 2),
]


def launch(job, args):
    name, points, gamma, seed = job
    out = Path(args.runs) / f"{name}.json"
    if out.exists():
        print(f"skip {name}", flush=True)
        return
    cmd = [
        sys.executable, "-m", "pointseg.train",
        "--data", args.data,
        "--out", str(out),
        "--points", str(points),
        "--gamma", str(gamma),
        "--seed", str(seed),
        "--epochs", str(args.epochs),
        "--threads", str(args.threads),
    ]
    log = Path(args.runs) / f"{name}.log"
    print(f"start {name}", flush=True)
    with open(log, "w") as handle:
        subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, check=True)
    print(f"done {name}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/rio")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    Path(args.runs).mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(lambda job: launch(job, args), GRID))


if __name__ == "__main__":
    main()
