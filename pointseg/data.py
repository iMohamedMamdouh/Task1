import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IGNORE_INDEX = 255
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def split_tiles(root, val_frac=0.15, test_frac=0.15, seed=0):
    names = (root / "tiles.txt").read_text().split()
    rng = np.random.default_rng(seed)
    rng.shuffle(names)
    n_val = int(len(names) * val_frac)
    n_test = int(len(names) * test_frac)
    return names[n_val + n_test:], names[:n_val], names[n_val:n_val + n_test]


def sample_points(mask, points_per_image, mode, rng):
    """Turn a dense mask into a sparse point annotation.

    balanced: points_per_image pixels per class, mimicking an annotator who
    clicks every class present in the tile.
    uniform: the same total budget drawn from the whole tile, so rare classes
    are hit in proportion to their area.
    """
    labels = np.full(mask.shape, IGNORE_INDEX, dtype=np.uint8)
    classes = np.unique(mask)

    if mode == "uniform":
        budget = points_per_image * len(classes)
        flat = rng.choice(mask.size, min(budget, mask.size), replace=False)
        labels.flat[flat] = mask.flat[flat]
        return labels

    for cls in classes:
        candidates = np.flatnonzero(mask == cls)
        chosen = rng.choice(candidates, min(points_per_image, candidates.size), replace=False)
        labels.flat[chosen] = cls
    return labels


class RioBuildings(Dataset):
    def __init__(self, root, names, points_per_image=None, point_mode="balanced",
                 augment=False, seed=0):
        self.root = root
        self.names = names
        self.augment = augment
        self.images = []
        self.masks = []
        self.targets = []

        for index, name in enumerate(names):
            image = np.array(Image.open(root / "images" / name).convert("RGB"))
            mask = (np.array(Image.open(root / "masks" / name)) > 127).astype(np.uint8)
            self.images.append(image)
            self.masks.append(mask)
            if points_per_image is None:
                self.targets.append(mask.copy())
            else:
                rng = np.random.default_rng([seed, index])
                self.targets.append(sample_points(mask, points_per_image, point_mode, rng))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        image = self.images[index]
        mask = self.masks[index]
        target = self.targets[index]

        if self.augment:
            k = np.random.randint(4)
            image, mask, target = (np.rot90(a, k) for a in (image, mask, target))
            if np.random.rand() < 0.5:
                image, mask, target = (np.fliplr(a) for a in (image, mask, target))

        image = (image.astype(np.float32) / 255.0 - MEAN) / STD
        return (
            torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
            torch.from_numpy(np.ascontiguousarray(target)).long(),
            torch.from_numpy(np.ascontiguousarray(mask)).long(),
        )
