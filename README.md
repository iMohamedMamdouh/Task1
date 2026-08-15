# Point-supervised remote sensing segmentation with partial cross entropy

Segmentation networks normally need a complete mask for every training tile. Field
annotation for land cover work usually produces a handful of clicked points instead.
This project implements the partial (focal) cross entropy loss that makes those point
labels usable, applies it to building segmentation on SpaceNet imagery, and measures how
much annotation is actually needed.

```
pointseg/losses.py   partial focal cross entropy
pointseg/data.py     tile dataset and point-label simulation
pointseg/model.py    ResNet-18 U-Net with an ImageNet encoder
pointseg/train.py    training, evaluation and metrics
scripts/             dataset download, experiment grid, figures
report/REPORT.md     method and experiment report
```

## Loss

```python
from pointseg.losses import PartialFocalCE

criterion = PartialFocalCE(gamma=2.0, ignore_index=255)
loss = criterion(logits, target)   # target: class ids at annotated pixels, 255 elsewhere
```

`sum(focal(pred, gt) * mask_labeled) / sum(mask_labeled)`, so unlabelled pixels affect
neither the numerator nor the normaliser. With `gamma=0` it reduces to plain partial CE,
and on a fully labelled target it matches `F.cross_entropy` exactly.

## Data

SpaceNet 1 (AOI 1, Rio de Janeiro): 3-band pansharpened WorldView-3 tiles at 0.5 m with
building footprints. The download script rasterises the GeoJSON footprints, centre-crops
and resizes to 256x256, and keeps tiles with at least 1% building cover.

```bash
pip install -r requirements.txt
python -m scripts.download_spacenet --num-tiles 900
```

## Training

```bash
python -m pointseg.train --points 5              # 5 labelled pixels per class per tile
python -m pointseg.train --points 0              # full masks, upper bound
python -m pointseg.train --points 5 --gamma 2    # partial focal CE
```

Point labels are drawn once per tile with a fixed seed, so a run sees the same annotation
in every epoch, exactly as a real point-annotated dataset would behave. Validation and
test always use the complete masks.

## Experiments

```bash
python -m scripts.run_experiments --epochs 12
python -m scripts.make_figures
```

Results and the discussion are in [report/REPORT.md](report/REPORT.md).
