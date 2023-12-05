import torch
import torch.nn as nn


class HoALoss(nn.Module):
    def __init__(self, hoa_bins=32, skip_frames=60):
        super().__init__()

        self.hoa_bins = hoa_bins
        self.skip_frames = skip_frames

    def forward(self, target, pred, ignore_weights=None):
        r"""
        target: (B, L, N)
        pred: (B, L, N)
        ignore_weights: (B, L)"""
        n = target.size(2)

        # reshape
        target = target.reshape(-1, self.hoa_bins)
        pred = pred.reshape(-1, self.hoa_bins)
        
        # make each histogram sum to 1
        pred = torch.softmax(pred, dim=1)

        # compute EMD using Mallow's distance
        loss = earth_mover_distance(target, pred)

        # ignore first `self.skip_frames` frames
        ignore_weights[:, :self.skip_frames] = 1.0
        ignore_weights = ignore_weights.unsqueeze(2).repeat((1, 1, n, 1))
        weights = 1 - ignore_weights.view(-1)
        loss = torch.sum(loss * weights) / torch.sum(weights)
        return loss


def earth_mover_distance(y_true, y_pred):
    return torch.mean(
        torch.square(torch.cumsum(y_true, dim=-1) - torch.cumsum(y_pred, dim=-1)),
        dim=-1,
    )
