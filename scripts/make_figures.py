import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pointseg.data import RioBuildings, sample_points, split_tiles
from pointseg.model import UNetResNet18


def load_runs(runs):
    results = {}
    for path in sorted(Path(runs).glob("*.json")):
        results[path.stem] = json.loads(path.read_text())
    return results


def label_efficiency(results, out):
    density = [1, 5, 20, 100]
    ce = [results[f"points{n}_ce"]["test"]["iou_building"] for n in density]
    focal = [results[f"points{n}_focal2"]["test"]["iou_building"] for n in density
             if f"points{n}_focal2" in results]
    dense = results["dense_ce"]["test"]["iou_building"]

    seeds = [results[name]["test"]["iou_building"] for name in
             ["points5_ce", "points5_ce_seed1", "points5_ce_seed2"] if name in results]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(dense, color="grey", ls="--", label=f"full masks ({dense:.3f})")
    if len(seeds) > 1:
        ax.vlines(5, min(seeds), max(seeds), color="tab:blue", alpha=0.35, lw=6,
                  label=f"seed spread at 5 points ({max(seeds) - min(seeds):.3f})")
    ax.plot(density, ce, "o-", label="partial CE")
    if focal:
        ax.plot(density[:len(focal)], focal, "s-", label="partial focal CE")
    ax.set_xscale("log")
    ax.set_xlabel("labelled points per class per tile")
    ax.set_ylabel("building IoU (test)")
    ax.set_title("Label efficiency of point supervision")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "label_efficiency.png", dpi=150)


def training_curves(results, out):
    fig, ax = plt.subplots(figsize=(6, 4))
    for name in ["dense_ce", "points1_ce", "points5_ce", "points20_ce", "points100_ce"]:
        if name not in results:
            continue
        history = results[name]["history"]
        ax.plot([h["epoch"] for h in history], [h["miou"] for h in history], label=name)
    ax.set_xlabel("epoch")
    ax.set_ylabel("val mIoU")
    ax.set_title("Validation mIoU during training")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=150)


def annotation_example(data, out):
    root = Path(data)
    names = (root / "tiles.txt").read_text().split()[:1]
    image = np.array(Image.open(root / "images" / names[0]).convert("RGB"))
    mask = (np.array(Image.open(root / "masks" / names[0])) > 127).astype(np.uint8)
    points = sample_points(mask, 20, "balanced", np.random.default_rng(0))

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    axes[0].imshow(image)
    axes[0].set_title("image")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("full mask (not used for training)")
    axes[2].imshow(image)
    for cls, colour in [(0, "tab:blue"), (1, "tab:red")]:
        ys, xs = np.where(points == cls)
        axes[2].scatter(xs, ys, s=14, c=colour, label="background" if cls == 0 else "building")
    axes[2].set_title("20 points per class")
    axes[2].legend(fontsize=8, loc="lower right")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "annotation_example.png", dpi=150)


def qualitative(data, checkpoint_dir, out, checkpoints):
    root = Path(data)
    _, _, test_names = split_tiles(root)
    subset = test_names[:4]
    dataset = RioBuildings(root, subset)

    available = [(name, Path(checkpoint_dir) / f"{name}.pt") for name in checkpoints]
    available = [(name, path) for name, path in available if path.exists()]
    if not available:
        return

    fig, axes = plt.subplots(len(subset), 2 + len(available), figsize=(3 * (2 + len(available)), 3 * len(subset)))
    predictions = {}
    for name, path in available:
        model = UNetResNet18(2, pretrained=False)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval()
        with torch.no_grad():
            batch = torch.stack([dataset[i][0] for i in range(len(subset))])
            predictions[name] = model(batch).argmax(1).numpy()

    for row in range(len(subset)):
        axes[row, 0].imshow(np.array(Image.open(root / "images" / subset[row]).convert("RGB")))
        axes[row, 1].imshow(dataset.masks[row], cmap="gray")
        if row == 0:
            axes[row, 0].set_title("image")
            axes[row, 1].set_title("ground truth")
        for col, (name, _) in enumerate(available):
            axes[row, 2 + col].imshow(predictions[name][row], cmap="gray")
            if row == 0:
                axes[row, 2 + col].set_title(name)
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "qualitative.png", dpi=130)


def results_table(results, out):
    rows = []
    for name, result in results.items():
        rows.append({
            "run": name,
            "points": result["config"]["points"],
            "gamma": result["config"]["gamma"],
            "seed": result["config"]["seed"],
            "labelled_fraction": result["labelled_fraction"],
            "val_miou": result["val"]["miou"],
            "test_miou": result["test"]["miou"],
            "test_iou_building": result["test"]["iou_building"],
            "test_f1_building": result["test"]["f1_building"],
            "test_oa": result["test"]["overall_accuracy"],
            "minutes": result["minutes"],
        })
    rows.sort(key=lambda r: (r["gamma"], r["points"], r["seed"]))
    header = list(rows[0])
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(f"{row[k]:.6f}" if isinstance(row[k], float) else str(row[k])
                              for k in header))
    (out / "results.csv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    columns = ["run", "points", "gamma", "labelled_fraction", "test_miou",
               "test_iou_building", "test_f1_building", "test_oa"]
    table = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        cells = []
        for key in columns:
            value = row[key]
            if key == "labelled_fraction":
                cells.append(f"{100 * value:.4f}%")
            elif isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        table.append("| " + " | ".join(cells) + " |")
    (out / "results.md").write_text("\n".join(table) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/rio")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--out", default="report/figures")
    parser.add_argument("--checkpoint-dir", default="runs/qualitative")
    parser.add_argument("--checkpoints", nargs="*", default=["points5_ce", "dense_ce"])
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = load_runs(args.runs)

    annotation_example(args.data, out)
    if results:
        results_table(results, out)
        label_efficiency(results, out)
        training_curves(results, out)
        qualitative(args.data, args.checkpoint_dir, out, args.checkpoints)


if __name__ == "__main__":
    main()
