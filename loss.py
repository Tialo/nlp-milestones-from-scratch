import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


def create_smoothed_vectors(
    n_rows: int,
    n_cols: int,
    confident_inds: torch.Tensor,
    inds_to_zero: torch.Tensor,
    smoothing: float = 0.1,
):
    assert confident_inds.ndim == 1, "confident_inds should be a 1D tensor"
    assert inds_to_zero.ndim == 1, "inds_to_zero should be a 1D tensor"
    assert confident_inds.shape[0] == n_rows, "each row should have one confident index"
    assert not any(torch.isin(confident_inds, inds_to_zero)), (
        "confident_inds and inds_to_zero values should not overlap"
    )
    # smooth all columns except inds_to_zero and confident_inds
    n_smoothed_inds = n_cols - len(inds_to_zero) - 1
    # probability of other columns should be smoothing / n_smoothed_inds
    smoothed_probability = smoothing / n_smoothed_inds
    smoothed_distribution = torch.full((n_rows, n_cols), smoothed_probability)
    # probability of confident_ind in each row should be 1 - smoothing
    smoothed_distribution[torch.arange(n_rows), confident_inds] = 1 - smoothing
    # set the probability of inds_to_zero to 0
    smoothed_distribution[:, inds_to_zero] = 0.0
    # ensure that the smoothed distribution in each row sums to 1
    assert all(abs(smoothed_distribution.sum(axis=1) - 1.0) < 1e-6), (
        "smoothed_distribution does not sum to 1 in each row"
    )
    return smoothed_distribution


class LabelSmoothingLoss(nn.Module):
    def __init__(
        self,
        ignore_index: int,
        smoothing: float = 0.1,
        use_cross_entropy: bool = True,
        reduction: str = "batchmean",
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.smoothing = smoothing
        self.use_cross_entropy = use_cross_entropy

        if use_cross_entropy and reduction == "batchmean":
            # batchmean is not supported for cross entropy loss
            reduction = "mean"

        if not use_cross_entropy and reduction == "mean":
            warnings.warn(
                "Using 'mean' reduction with kl div loss is not recommended. "
                "It does not align with the mathematical definition of mean for kl divergence. "
                "Please use 'batchmean'. "
                "For more details, refer to the PyTorch documentation for KLDivLoss. "
                "https://docs.pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html"
            )

        self.reduction = reduction

    def forward(self, logits, target):
        """
        logits: (n_samples, n_classes)
        target: (n_samples,)
        """
        # ignore samples where target is ignore_index
        valid_mask = target != self.ignore_index
        logits = logits[valid_mask]
        target = target[valid_mask]

        smoothed_distribution = create_smoothed_vectors(
            n_rows=logits.shape[0],
            n_cols=logits.shape[1],
            confident_inds=target,
            inds_to_zero=torch.tensor([self.ignore_index], device=target.device),
            smoothing=self.smoothing,
        ).to(logits.device)
        # if smoothing == 0.0, smoothed_distribution will be all zeros
        # except for the confident indices,
        # so clamp distribution to avoid log(0) in kldiv loss calculation
        smoothed_distribution = smoothed_distribution.clamp(min=1e-42)

        log_probs = F.log_softmax(logits, dim=-1)
        if self.use_cross_entropy:
            loss = -(smoothed_distribution * log_probs).sum(dim=-1)
        else:
            loss = smoothed_distribution * (smoothed_distribution.log() - log_probs)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "batchmean":
            return loss.sum() / loss.shape[0]
        else:
            return loss
