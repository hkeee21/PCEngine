from typing import Optional
import numpy as np
import torch
from torch import nn

__all__ = ['SigmoidFocalLoss', 'WeightedSmoothL1Loss', 'L1LossCenterPoint', 'FocalLossCenterPoint']


class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """
    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)

        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss


class SigmoidFocalLoss(nn.Module):
    def __init__(self,
                 gamma: Optional[float] = 2.0,
                 alpha: Optional[float] = 0.25) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, outputs, targets, weights):
        """Compute loss function.

        Args:
            prediction_tensor: A float tensor of shape [batch_size, num_anchors,
              num_classes] representing the predicted logits for each class
            target_tensor: A float tensor of shape [batch_size, num_anchors,
              num_classes] representing one-hot encoded classification targets
            weights: a float tensor of shape [batch_size, num_anchors]

        Returns:
          loss: a float tensor of shape [batch_size, num_anchors, num_classes]
            representing the value of the loss function.
        """
        per_entry_cross_ent = (_sigmoid_cross_entropy_with_logits(
            labels=targets, logits=outputs))

        probs = torch.sigmoid(outputs)
        pt = ((targets * probs) + ((1 - targets) * (1 - probs)))

        if self.gamma:
            modulating_factor = torch.pow(1.0 - pt, self.gamma)
        else:
            modulating_factor = 1.0

        if self.alpha is not None:
            alpha_weight_factor = (targets * self.alpha + (1 - targets) *
                                   (1 - self.alpha))
        else:
            alpha_weight_factor = 1.0

        focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor *
                                    per_entry_cross_ent)
        return focal_cross_entropy_loss * weights


class L1LossCenterPoint(nn.Module):
    def __init__(self, code_weights=None):
        super().__init__()
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()
        else:
            self.code_weights = None

    def forward(self, pred, target, mask):
        loss = torch.nn.functional.l1_loss(pred*mask, target*mask, reduction='none')
        #print(pred[mask.bool().squeeze(-1)], target[mask.bool().squeeze(-1)])
        #print((target * mask).sum(), mask.sum(), target.sum())
        if self.code_weights is not None:
            loss = loss * self.code_weights.view(1, 1, -1)
        loss = loss.sum() / (mask.sum() + 1e-4)
        return loss


class FocalLossCenterPoint(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out, target, mask):
        # out: B x (H*W) x num_classes
        # target: B x (H*W) x num_classes
        out = torch.clamp(out.sigmoid_(), min=1e-4, max=1-1e-4)

        gt = torch.pow(1 - target, 4)
        neg_loss = torch.log(1 - out) * torch.pow(out, 2) * gt
        neg_loss = neg_loss.sum()

        pos_pred = out[mask == 1]
        num_pos = torch.sum(mask).item()
        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
        pos_loss = pos_loss.sum()
        if num_pos == 0:
            return - neg_loss
        
        return - (pos_loss + neg_loss) / num_pos

        
def _sigmoid_cross_entropy_with_logits(logits, labels):
    loss = torch.clamp(logits, min=0) - logits * labels.type_as(logits)
    loss += torch.log1p(torch.exp(-torch.abs(logits)))
    return loss
