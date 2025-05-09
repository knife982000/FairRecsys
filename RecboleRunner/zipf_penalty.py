import torch

from RecBole.recbole.data.utils import create_dataset


def build_item_popularity(config):
    item_id = config["ITEM_ID_FIELD"]
    dataset = create_dataset(config)

    item_counts = torch.zeros(dataset.num(item_id), device=config["device"], dtype=torch.float)

    # Ensure item_ids is on the same device as item_counts
    item_ids = torch.tensor(dataset.inter_feat[item_id].values, device=config["device"])

    # Count occurrences of each item
    ones = torch.ones_like(item_ids, device=config["device"], dtype=torch.float)
    item_counts.scatter_add_(0, item_ids, ones)

    return item_counts


def get_alpha(config):
    return config["zipf_alpha"] if "zipf_alpha" in config else 0.94


def zipf_penalty_singular(config, score, item):
    """Apply Zipf's penalty to the given scores."""
    score = score.clone().detach()

    item_popularity = build_item_popularity(config)

    zipf_penalty = get_alpha(config) * torch.log1p(item_popularity[item] + 1)
    return score - zipf_penalty


def zipf_penalty_batch(config, scores: torch.Tensor):
    scores = scores.clone().detach()

    item_popularity = build_item_popularity(config)

    zipf_penalty = get_alpha(config) * torch.log1p(item_popularity)
    return scores - zipf_penalty.unsqueeze(0)
