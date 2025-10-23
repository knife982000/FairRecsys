import torch

from RecBole.recbole.model.general_recommender.ngcf import NGCF
from RecBole.recbole.utils import InputType

from model import BPRLoss, EntropyLoss, EntropyLoss2
from model.config_updater import update_config


class NGCFEntropy(NGCF):
    r"""NGCF is a model that incorporate GNN for recommendation.
    We implement the model following the original author with a pairwise training mode.
    """

    input_type = InputType.PAIRWISE
    file_properties = "RecBole/recbole/properties/model/NGCF.yaml"

    def __init__(self, config, dataset):
        config = update_config(self.file_properties, config)
        super(NGCFEntropy, self).__init__(config, dataset)
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

        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)  # calculate BPR Loss

        reg_loss = self.reg_loss(
            u_embeddings, pos_embeddings, neg_embeddings
        )  # L2 regularization of embeddings

        ep_loss = self.ep_loss(mf_loss, pos_item)

        loss = ep_loss + self.reg_weight * reg_loss

        return loss


class NGCFEntropy2(NGCFEntropy):
    r"""NGCF is a model that incorporate GNN for recommendation.
    We implement the model following the original author with a pairwise training mode.
    """

    input_type = InputType.PAIRWISE
    file_properties = "RecBole/recbole/properties/model/NGCF.yaml"

    def __init__(self, config, dataset):
        config = update_config(self.file_properties, config)
        super(NGCFEntropy2, self).__init__(config, dataset)
        self.ep_loss = EntropyLoss2(dataset, self.ITEM_ID, config["entropy_alpha"] if "entropy_alpha" in config else 1)