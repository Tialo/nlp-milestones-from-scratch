import torch
import torch.nn as nn
import torch.nn.functional as F


# note: this function isn't used because one does not need
# to materialize those vectors to calculate ce loss
def create_smoothed_vectors(
    n_rows: int,
    n_cols: int,
    confident_inds: torch.Tensor,
    inds_to_zero: torch.Tensor,
    smoothing: float = 0.1,
    device: torch.device | None = None,
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
    smoothed_distribution = torch.full((n_rows, n_cols), smoothed_probability, device=device)
    # probability of confident_ind in each row should be 1 - smoothing
    smoothed_distribution[torch.arange(n_rows), confident_inds] = 1 - smoothing
    # set the probability of inds_to_zero to 0
    smoothed_distribution[:, inds_to_zero] = 0.0
    # ensure that the smoothed distribution in each row sums to 1
    assert all(abs(smoothed_distribution.sum(axis=1) - 1.0) < 1e-6), (
        "smoothed_distribution does not sum to 1 in each row"
    )
    return smoothed_distribution


class SlowLabelSmoothingLoss(nn.Module):
    def __init__(
        self,
        ignore_index: int,
        smoothing: float = 0.1,
        reduction: str = "mean",
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.smoothing = smoothing
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
            device=logits.device,
        )
        # if smoothing == 0.0, smoothed_distribution will be all zeros
        # except for the confident indices,
        # so clamp distribution to avoid log(0) in kldiv loss calculation
        smoothed_distribution = smoothed_distribution.clamp(min=1e-42)

        log_probs = F.log_softmax(logits, dim=-1)

        # each log proba gets multiplied by corresponding smoothed probability
        loss = -(smoothed_distribution * log_probs).sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class LabelSmoothingLoss(nn.Module):
    def __init__(
        self,
        ignore_index: int,
        smoothing: float = 0.1,
        reduction: str = "mean",
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits, target):
        """
        logits: (n_samples, n_classes)
        target: (n_samples,)
        """
        # ignore samples where target is ignore_index
        valid_mask = target != self.ignore_index
        logits = logits[valid_mask]  # (n_filtered, n_classes)
        target = target[valid_mask]  # (n_filtered,)

        log_probs = F.log_softmax(logits, dim=-1)

        confident_log_probs = log_probs[torch.arange(len(target)), target]  # (n_filtered,)
        # log proba of target gets multiplied by confident probability
        confident_component = confident_log_probs * (1 - self.smoothing)  # (n_filtered,)
        # all log probs (including that of target) get multiplied by smoothed probability
        smoothed_component = log_probs * self.smoothing / (log_probs.shape[1] - 1)  # (n_filtered, n_classes)
        # zero out log proba of target multiplied by smoothed_probability
        smoothed_component[torch.arange(len(target)), target] = 0
        smoothed_component = smoothed_component.sum(-1)   # (n_filtered,)

        loss = -(confident_component + smoothed_component)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def main():
    logits = torch.randn(512, 1024)
    target = torch.randint(0, 1024, (512,))
    reduction = 'mean'
    ls = 0.0
    slow = SlowLabelSmoothingLoss(ignore_index=0, reduction=reduction, smoothing=ls)
    fast = LabelSmoothingLoss(ignore_index=0, reduction=reduction, smoothing=ls)
    pt_impl = nn.CrossEntropyLoss(ignore_index=0, reduction=reduction, label_smoothing=ls)
    print(slow(logits, target))
    print(fast(logits, target))
    print(pt_impl(logits, target))


if __name__ == "__main__":
    main()
