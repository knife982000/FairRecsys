import torch

from RecBole.recbole.model.general_recommender.bpr import BPR
from RecBole.recbole.utils import InputType

from model import BPRLoss, EntropyLoss


class BPREntropy(BPR):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way."""

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(BPREntropy, self).__init__(config, dataset)
        self.ep_loss = EntropyLoss(dataset, self.ITEM_ID, config["entropy_alpha"] if "entropy_alpha" in config else 1)
        self.loss = BPRLoss(reduction="none")

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_e, pos_e = self.forward(user, pos_item)
        neg_e = self.get_item_embedding(neg_item)
        pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(
            user_e, neg_e
        ).sum(dim=1)
        loss = self.loss(pos_item_score, neg_item_score, pos_item)
        return loss
