import torch
import torch.nn as nn

from RecBole.recbole.model.abstract_recommender import GeneralRecommender


class GeneralRecommenderZipf(GeneralRecommender):
    r"""
    General Recommender with Zipf's penalty to reduce popularity bias.
    Math for Zipf's Penalty:
    s=1+n(\sum_{i=1}^{n}ln(\frac{x_i}{x_{max}}))^{-1}
    """

    def __init__(self, config, dataset):
        super(GeneralRecommenderZipf, self).__init__(config, dataset)

        # Initialize item popularity and Zipf's alpha
        self.item_popularity = None
        self.zipf_alpha = 0.94  # Default value for Zipf's alpha

        # Define embeddings
        self.embedding_size = config["embedding_size"]
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)

        # Compute item popularity for Zipf's penalty
        self.build_item_popularity(dataset)

    def build_item_popularity(self, dataset):
        """Precompute item popularity based on dataset interactions."""
        item_counts = torch.zeros(self.n_items, device=self.device, dtype=torch.float)

        # Ensure item_ids is on the same device as item_counts
        item_ids = dataset.inter_feat[self.ITEM_ID].to(self.device)

        # Count occurrences of each item
        ones = torch.ones_like(item_ids, device=self.device, dtype=torch.float)
        item_counts.scatter_add_(0, item_ids, ones)

        self.item_popularity = item_counts

    def compute_zipf_penalty(self):
        """Compute Zipf's penalty based on item popularity."""
        if self.item_popularity is None:
            raise ValueError("Item popularity has not been built. Call build_item_popularity first.")

        x_max = self.item_popularity.max()
        log_sum = torch.sum(torch.log((self.item_popularity + 1e-10) / x_max))  # Sum log of popularity ratios
        zipf_penalty = self.zipf_alpha * (1 + self.n_items * log_sum.reciprocal())  # Compute penalty term
        return zipf_penalty

    def apply_zipf_penalty(self, score, item):
        """Apply Zipf's penalty to the given scores."""
        zipf_penalty = self.zipf_alpha * torch.log1p(self.item_popularity[item])
        return score - zipf_penalty

    def forward(self, user, item):
        """Retrieve user and item embeddings."""
        user_e = self.user_embedding(user)
        item_e = self.item_embedding(item)
        return user_e, item_e
