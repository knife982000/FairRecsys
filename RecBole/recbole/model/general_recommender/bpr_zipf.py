r"""
BPR with Zipf's Penalty
################################################
Reference:
    Steffen Rendle et al. "BPR: Bayesian Personalized Ranking from Implicit Feedback." in UAI 2009.
    Zipf’s Law Penalty added to discourage over-recommendation of popular items. "Zipf Matrix Factorization : Matrix Factorization with
    Matthew Effect Reduction" in RecSys 2021.
"""

import torch
import torch.nn as nn

from recbole.model.abstract_recommender import GeneralRecommenderZipf
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType


class BPRZipf(GeneralRecommenderZipf):
    r"""
    BPR with Zipf's penalty to reduce popularity bias.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(BPRZipf, self).__init__(config, dataset)

        # Define loss
        self.loss = BPRLoss()

        # Parameters initialization
        self.apply(xavier_normal_initialization)

    def calculate_loss(self, interaction):
        if interaction is None:
            raise ValueError("Interaction data is None. Ensure the dataset is properly loaded.")

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        if user is None or pos_item is None or neg_item is None:
            raise ValueError("Missing required fields in interaction data.")

        user_e, pos_e = self.forward(user, pos_item)
        neg_e = self.item_embedding(neg_item)

        pos_item_score = torch.mul(user_e, pos_e).sum(dim=1)
        neg_item_score = torch.mul(user_e, neg_e).sum(dim=1)

        # Compute standard BPR loss
        loss = self.loss(pos_item_score, neg_item_score)

        # Add Zipf's penalty to the loss
        zipf_penalty = self.compute_zipf_penalty()
        loss += zipf_penalty

        return loss

    def predict(self, interaction):
        if interaction is None:
            raise ValueError("Interaction data is None. Ensure the dataset is properly loaded.")

        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        if user is None or item is None:
            raise ValueError("Missing required fields in interaction data.")

        user_e, item_e = self.forward(user, item)
        score = torch.mul(user_e, item_e).sum(dim=1)

        # Apply Zipf’s penalty to predictions
        score = self.apply_zipf_penalty(score, item)

        return score

    def full_sort_predict(self, interaction):
        if interaction is None:
            raise ValueError("Interaction data is None. Ensure the dataset is properly loaded.")

        user = interaction[self.USER_ID]

        if user is None:
            raise ValueError("Missing required fields in interaction data.")

        user_e = self.user_embedding(user)
        all_item_e = self.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))

        zipf_penalty = self.zipf_alpha * torch.log1p(self.item_popularity)
        score = score - zipf_penalty.unsqueeze(0)
        return score.view(-1)

    def get_zipf_alpha(self):
        """Retrieve the current value of zipf_alpha."""
        return self.zipf_alpha
