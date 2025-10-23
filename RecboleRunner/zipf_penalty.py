import torch

ITEM_COUNTS = None

def build_item_popularity(config, dataset=None):
    global ITEM_COUNTS
    if ITEM_COUNTS is not None:
        return ITEM_COUNTS
    
    if dataset is None:
        raise ValueError("Dataset must be provided to build item popularity.")
    
    item_id = config["ITEM_ID_FIELD"]

    ITEM_COUNTS = torch.zeros(dataset.num(item_id), device=config["device"], dtype=torch.float)

    # Ensure item_ids is on the same device as item_counts
    item_ids = torch.tensor(dataset.inter_feat[item_id], device=config["device"])

    # Count occurrences of each item
    ones = torch.ones_like(item_ids, device=config["device"], dtype=torch.float)
    ITEM_COUNTS.scatter_add_(0, item_ids, ones)

    return ITEM_COUNTS


def get_alpha(config):
    return config["zipf_alpha"] if "zipf_alpha" in config else 0.94


def zipf_penalty_singular(config, score, item):
    """Apply Zipf's penalty to the given scores."""
    score = score.clone().detach()

    item_popularity = build_item_popularity(config)

    zipf_penalty = get_alpha(config) * torch.log1p(item_popularity[item])
    return score - zipf_penalty


def zipf_penalty_batch(config, scores: torch.Tensor):
    scores = scores.clone().detach()

    item_popularity = build_item_popularity(config)

    zipf_penalty = get_alpha(config) * torch.log1p(item_popularity)
    return scores - zipf_penalty.unsqueeze(0)
