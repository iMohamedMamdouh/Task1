import torch
import torch.nn.functional as F

from pointseg.data import IGNORE_INDEX
from pointseg.losses import PartialFocalCE

torch.manual_seed(0)
LOGITS = torch.randn(2, 3, 8, 8)
TARGET = torch.randint(0, 3, (2, 8, 8))


def sparse_target():
    target = TARGET.clone()
    target[0, 2:, :] = IGNORE_INDEX
    target[1, :5, :] = IGNORE_INDEX
    return target


def test_dense_labels_match_cross_entropy():
    loss = PartialFocalCE(gamma=0.0)(LOGITS, TARGET)
    assert torch.allclose(loss, F.cross_entropy(LOGITS, TARGET))


def test_unlabelled_pixels_are_dropped():
    target = sparse_target()
    loss = PartialFocalCE(gamma=0.0)(LOGITS, target)
    assert torch.allclose(loss, F.cross_entropy(LOGITS, target, ignore_index=IGNORE_INDEX))


def test_focal_term():
    log_pt = F.log_softmax(LOGITS, 1).gather(1, TARGET.unsqueeze(1)).squeeze(1)
    expected = (-((1 - log_pt.exp()) ** 2) * log_pt).mean()
    assert torch.allclose(PartialFocalCE(gamma=2.0)(LOGITS, TARGET), expected)


def test_class_weights_scale_the_labelled_pixels():
    weights = torch.tensor([1.0, 2.0, 3.0])
    loss = PartialFocalCE(gamma=0.0, class_weights=weights)(LOGITS, TARGET)
    per_pixel = F.cross_entropy(LOGITS, TARGET, reduction="none") * weights[TARGET]
    assert torch.allclose(loss, per_pixel.mean())


def test_tile_without_annotation_is_finite():
    empty = torch.full((1, 8, 8), IGNORE_INDEX)
    loss = PartialFocalCE(gamma=0.0)(LOGITS[:1], empty)
    assert torch.isfinite(loss) and loss.item() == 0.0
