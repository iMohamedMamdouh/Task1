# Training a segmentation network from point labels with partial cross entropy

Technical report. Code: [`pointseg/`](../pointseg), experiment driver:
[`scripts/run_experiments.py`](../scripts/run_experiments.py).

## 1. Problem

Semantic segmentation is a per-pixel classification problem, and the usual training
recipe assumes a complete mask for every training image. Annotating remote sensing
imagery that way is expensive: a single 0.5 m tile can contain hundreds of building
outlines. Field campaigns and photo-interpreters produce point annotations instead, a
few clicked pixels with a class attached. A tile then has three states per pixel:
class 0, class 1, or unknown, and the unknown pixels dominate — in the experiments below
99.98% of all pixels are unlabelled.

Standard cross entropy cannot be used directly. Treating unlabelled pixels as background
teaches the network that most of a building is background; dropping the images entirely
throws away almost all the data. Partial cross entropy solves this by restricting both
the loss and its normaliser to the annotated pixels.

## 2. Method

### 2.1 Partial cross entropy

For a tile with predictions `p` and a labelled-pixel indicator `M` (1 where an annotation
exists, 0 elsewhere):

```
pCE = sum_i ( CE(p_i, y_i) * M_i ) / sum_i M_i
```

Unlabelled pixels contribute nothing to the numerator, and — the part that matters —
nothing to the denominator either. If they were included in the denominator, the loss
magnitude would shrink with annotation density and the effective learning rate would
change every time the annotation budget changed.

The gradient reaching the decoder is zero at unlabelled pixels. The network still
produces a dense prediction there; what makes those predictions sensible is the spatial
context in the receptive field and the smoothness prior of a convolutional decoder, not
the loss.

### 2.2 Focal extension

The landvisor formulation replaces CE with a focal term:

```
pfCE = sum_i ( focal(p_i, y_i) * M_i ) / sum_i M_i
focal(p, y) = -(1 - p_y)^gamma * log(p_y)
```

Under point supervision the annotated set is small, so a handful of easy points can
dominate the average once the network fits them. The focal weight `(1 - p_y)^gamma`
down-weights points the model already gets right and keeps the gradient on the hard
ones. `gamma = 0` recovers plain partial CE.

[`pointseg/losses.py`](../pointseg/losses.py) implements exactly this expression. Two
properties are checked in the code review sense: on a fully labelled target with
`gamma = 0` it reproduces `F.cross_entropy`, and on a partially labelled target it
reproduces `F.cross_entropy(..., ignore_index=255)`. A tile with no annotated pixel at
all returns 0 instead of NaN because the denominator is clamped.

### 2.3 Simulating point labels

Real point annotations are not available for the public dataset, so they are simulated
from the full masks. For each tile, `N` pixels are drawn uniformly at random from each
class present ([`pointseg/data.py`](../pointseg/data.py)); the rest of the tile is set to
the ignore index. The full mask is then never used for training.

Two properties of the simulation matter:

* Points are drawn **once per tile** with a seed derived from the run seed and the tile
  index, not re-drawn every epoch. Re-drawing would leak the full mask over training —
  after enough epochs the network would effectively have seen a dense label.
* Sampling is **class balanced** (`N` per class rather than `N` per tile). This mimics an
  annotator who clicks every class they see, and it is why a very small budget still
  works: buildings cover 17% of the average tile, so uniform sampling would spend most
  clicks on background. A uniform mode is implemented as well for comparison.

Geometric augmentation (rotations by multiples of 90 degrees, horizontal flips) is applied
to the image and the point map together, so the annotation stays attached to its pixel.

### 2.4 Network and training

* **Architecture**: U-Net with a ResNet-18 encoder ([`pointseg/model.py`](../pointseg/model.py)),
  skip connections at strides 2, 4, 8, 16, bilinear upsampling in the decoder.
* **Transfer learning**: the encoder starts from ImageNet weights. With 0.015% of pixels
  labelled there is not enough signal to learn low-level filters from scratch, so the
  pretrained backbone is what makes the low-budget runs work at all.
* **Optimisation**: AdamW, learning rate 3e-4, weight decay 1e-4, cosine schedule,
  batch size 8, 12 epochs, single CPU (no GPU was available in this environment, which is
  what fixed the scale of the study).
* **Model selection**: the epoch with the best validation mIoU is kept and evaluated once
  on the test split.

## 3. Experiments

### 3.1 Data

SpaceNet 1, AOI 1 Rio de Janeiro: 3-band pansharpened WorldView-3 tiles at 0.5 m ground
sample distance with building footprint polygons. The footprints are rasterised with the
tile's affine transform, the tile is centre-cropped to square and resized to 256x256, and
tiles with less than 1% building cover are dropped so that every tile contains the
positive class. That yields **900 tiles**, split 630 train / 135 validation / 135 test
with a fixed seed. Mean building cover is 17.0%.

![annotation](figures/annotation_example.png)

*Image, the full mask the annotation is drawn from, and a 20-points-per-class annotation.
Only the points on the right are visible to the loss.*

### 3.2 Protocol

Validation and test always use the **complete** masks — the point labels are a training
handicap, not an evaluation one. Metrics are computed from a confusion matrix accumulated
over the split: per-class IoU, mean IoU, building F1 and overall accuracy. Building IoU is
the headline number because background IoU is high for any non-degenerate model.

