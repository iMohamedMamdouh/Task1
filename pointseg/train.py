import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pointseg.data import IGNORE_INDEX, RioBuildings, split_tiles
from pointseg.losses import PartialFocalCE
from pointseg.model import UNetResNet18

NUM_CLASSES = 2


def confusion_matrix(pred, target):
    k = target * NUM_CLASSES + pred
    return np.bincount(k.ravel(), minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)


def metrics_from_confusion(cm):
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    iou = tp / np.maximum(tp + fp + fn, 1e-9)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)
    return {
        "miou": float(iou.mean()),
        "iou_background": float(iou[0]),
        "iou_building": float(iou[1]),
        "f1_building": float(f1[1]),
        "overall_accuracy": float(tp.sum() / cm.sum()),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for images, _, masks in loader:
        logits = model(images.to(device))
        pred = logits.argmax(1).cpu().numpy()
        cm += confusion_matrix(pred, masks.numpy())
    return metrics_from_confusion(cm)


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)

    root = Path(args.data)
    train_names, val_names, test_names = split_tiles(root, seed=args.split_seed)
    points = None if args.points == 0 else args.points

    train_set = RioBuildings(root, train_names, points, args.point_mode, augment=True, seed=args.seed)
    val_set = RioBuildings(root, val_names)
    test_set = RioBuildings(root, test_names)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    model = UNetResNet18(NUM_CLASSES, pretrained=not args.no_pretrain).to(device)
    criterion = PartialFocalCE(gamma=args.gamma, ignore_index=IGNORE_INDEX).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    labelled = sum(int((t != IGNORE_INDEX).sum()) for t in train_set.targets)
    total = sum(t.size for t in train_set.targets)
    print(f"train tiles {len(train_set)}  labelled pixels {labelled}/{total} "
          f"({100 * labelled / total:.4f}%)", flush=True)

    history = []
    best = {"miou": -1.0}
    best_state = None
    start = time.time()

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for images, targets, _ in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images.to(device)), targets.to(device))
            loss.backward()
            optimizer.step()
            running += loss.item()
        scheduler.step()

        val = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "loss": running / len(train_loader), **val})
        print(f"epoch {epoch:02d}  loss {running / len(train_loader):.4f}  "
              f"val mIoU {val['miou']:.4f}  building IoU {val['iou_building']:.4f}  "
              f"[{time.time() - start:.0f}s]", flush=True)

        if val["miou"] > best["miou"]:
            best = val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test = evaluate(model, test_loader, device)
    print(f"test mIoU {test['miou']:.4f}  building IoU {test['iou_building']:.4f}", flush=True)

    result = {
        "config": vars(args),
        "labelled_pixels": labelled,
        "labelled_fraction": labelled / total,
        "history": history,
        "val": best,
        "test": test,
        "minutes": (time.time() - start) / 60,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    if args.save_checkpoint:
        torch.save(best_state, out.with_suffix(".pt"))
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/rio")
    parser.add_argument("--out", default="runs/run.json")
    parser.add_argument("--points", type=int, default=5,
                        help="labelled pixels per class per tile, 0 means full masks")
    parser.add_argument("--point-mode", default="balanced", choices=["balanced", "uniform"])
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-pretrain", action="store_true")
    parser.add_argument("--save-checkpoint", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
