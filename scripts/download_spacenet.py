"""Build a small building-segmentation dataset from SpaceNet 1 (AOI 1, Rio de Janeiro).

Images are 3-band pansharpened WorldView-3 tiles, labels are OSM-style building
footprints stored as GeoJSON and rasterised here into binary masks.
"""

import argparse
import io
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
import requests
from PIL import Image
from rasterio.features import rasterize

BUCKET = "https://spacenet-dataset.s3.amazonaws.com/spacenet/SN1_buildings/train"
MAX_IMAGE_ID = 6940


def fetch(url, attempts=4):
    for attempt in range(attempts):
        try:
            r = requests.get(url, timeout=60)
        except requests.RequestException:
            continue
        if r.status_code == 404:
            return None
        if r.status_code == 200:
            return r.content
    return None


def build_tile(image_id, size, min_building_frac):
    label = fetch(f"{BUCKET}/geojson/Geo_AOI_1_RIO_img{image_id}.geojson")
    if label is None:
        return None
    features = json.loads(label)["features"]
    if not features:
        return None

    raw = fetch(f"{BUCKET}/3band/3band_AOI_1_RIO_img{image_id}.tif")
    if raw is None:
        return None

    with rasterio.open(io.BytesIO(raw)) as src:
        image = src.read().transpose(1, 2, 0)
        mask = rasterize(
            [f["geometry"] for f in features],
            out_shape=(src.height, src.width),
            transform=src.transform,
            dtype="uint8",
        )

    if image.shape[2] != 3 or image.max() == 0:
        return None
    if mask.mean() < min_building_frac:
        return None

    side = min(image.shape[0], image.shape[1])
    top = (image.shape[0] - side) // 2
    left = (image.shape[1] - side) // 2
    image = image[top:top + side, left:left + side]
    mask = mask[top:top + side, left:left + side]

    image = Image.fromarray(image).resize((size, size), Image.BILINEAR)
    mask = Image.fromarray(mask * 255).resize((size, size), Image.NEAREST)
    return image, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/rio")
    parser.add_argument("--num-tiles", type=int, default=900)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--min-building-frac", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)

    ids = list(range(1, MAX_IMAGE_ID + 1))
    random.Random(args.seed).shuffle(ids)

    kept = []
    with ThreadPoolExecutor(args.workers) as pool:
        for chunk_start in range(0, len(ids), args.workers * 8):
            if len(kept) >= args.num_tiles:
                break
            chunk = ids[chunk_start:chunk_start + args.workers * 8]
            results = pool.map(
                lambda i: (i, build_tile(i, args.size, args.min_building_frac)), chunk
            )
            for image_id, tile in results:
                if tile is None or len(kept) >= args.num_tiles:
                    continue
                image, mask = tile
                name = f"rio_{image_id:05d}.png"
                image.save(out / "images" / name)
                mask.save(out / "masks" / name)
                kept.append(name)
            print(f"scanned {chunk_start + len(chunk)} ids, kept {len(kept)}", flush=True)

    kept.sort()
    (out / "tiles.txt").write_text("\n".join(kept) + "\n")

    coverage = []
    for name in kept:
        m = np.array(Image.open(out / "masks" / name)) > 127
        coverage.append(float(m.mean()))
    print(f"tiles: {len(kept)}  mean building coverage: {np.mean(coverage):.4f}")


if __name__ == "__main__":
    main()
