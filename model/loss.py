import torch
import torch.nn as nn

class BPRLoss(nn.Module):
    """BPRLoss, based on Bayesian Personalized Ranking

    Args:
        - gamma(float): Small value to avoid division by zero

    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.

    Examples::

        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """

    def __init__(self, gamma=1e-10, reduction='mean'):
        super(BPRLoss, self).__init__()
        if reduction == 'mean':
            self.reduction = torch.mean
        elif reduction == 'sum':
            self.reduction = torch.sum
        elif reduction == 'none':
            self.reduction = lambda x: x
        else:
            raise ValueError(f"{reduction} is not a valid value for reduction, i.e, mean, sum, or none")
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score))
        return self.reduction(loss)


class EntropyLoss(nn.Module):

    def __init__(self, base_loss, dataset, item_id, alpha=1):
        super(EntropyLoss, self).__init__()
        self.base_loss = base_loss(reduction='none')
        items_count = self._build_item_popularity(dataset, item_id)
        boost = 1 - (torch.log1p(items_count) + 1) / (torch.log1p(torch.max(items_count)) + 1)
        boost = boost ** alpha
        self.register_buffer('items_boost', boost)
        pass

    def _build_item_popularity(self, dataset, item_id):
        """Precompute item popularity based on dataset interactions."""
        item_counts = torch.zeros(dataset.item_num, dtype=torch.float)

        # Ensure item_ids is on the same device as item_counts
        item_ids = dataset.inter_feat[item_id]

        # Count occurrences of each item
        ones = torch.ones_like(item_ids, dtype=torch.float)
        item_counts.scatter_add_(0, item_ids, ones)

        return item_counts

    def forward(self, predict, targets, items):
        loss = self.base_loss(predict, targets)
        boost = self.items_boost[items]
        loss = loss * boost
        return torch.sum(loss) / torch.sum(boost)
