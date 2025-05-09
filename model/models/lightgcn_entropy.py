import torch

from RecBole.recbole.model.general_recommender.lightgcn import LightGCN
from RecBole.recbole.utils import InputType

from model import BPRLoss, EntropyLoss
from model.config_updater import update_config


class LightGCNEntropy(LightGCN):
    r"""LightGCN is a GCN-based recommender model.

    LightGCN includes only the most essential component in GCN — neighborhood aggregation — for
    collaborative filtering. Specifically, LightGCN learns user and item embeddings by linearly
    propagating them on the user-item interaction graph, and uses the weighted sum of the embeddings
    learned at all layers as the final embedding.

    We implement the model following the original author with a pairwise training mode.
    """

    input_type = InputType.PAIRWISE
    file_properties = "RecBole/recbole/properties/model/LightGCN.yaml"

    def __init__(self, config, dataset):
        config = update_config(self.file_properties, config)
        super(LightGCNEntropy, self).__init__(config, dataset)
        self.ep_loss = EntropyLoss(dataset, self.ITEM_ID, config["entropy_alpha"] if "entropy_alpha" in config else 1)
        self.mf_loss = BPRLoss(reduction="none")

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate regularization Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(
            u_ego_embeddings,
            pos_ego_embeddings,
            neg_ego_embeddings,
            require_pow=self.require_pow,
        )

        ep_loss = self.ep_loss(mf_loss, pos_item)

        loss = ep_loss + self.reg_weight * reg_loss

        return loss
