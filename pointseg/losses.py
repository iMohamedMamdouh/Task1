import torch
import torch.nn.functional as F
from torch import nn


class PartialFocalCE(nn.Module):
    """Focal cross entropy averaged over labelled pixels only.

        pfCE = sum(focal(pred, gt) * mask_labeled) / sum(mask_labeled)

    Pixels carrying `ignore_index` are unlabelled and contribute nothing to the
    loss or to the normaliser. With gamma = 0 this is the plain partial CE.
    """

    def __init__(self, gamma=0.0, class_weights=None, ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        if class_weights is not None and not torch.is_tensor(class_weights):
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.register_buffer("class_weights", class_weights)

    def forward(self, logits, target):
        labeled = (target != self.ignore_index).float()
        target = target.masked_fill(target == self.ignore_index, 0)

        log_prob = F.log_softmax(logits, dim=1)
        log_pt = log_prob.gather(1, target.unsqueeze(1)).squeeze(1)
        loss = -((1.0 - log_pt.exp()) ** self.gamma) * log_pt

        if self.class_weights is not None:
            loss = loss * self.class_weights[target]

        return (loss * labeled).sum() / labeled.sum().clamp(min=1.0)
