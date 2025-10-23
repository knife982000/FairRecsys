import torch

from RecBole.recbole.model.transformer_reccomender.ssept import SSEPT
from RecBole.recbole.utils import InputType

from model import BPRLoss, EntropyLoss, EntropyLoss2


class SSEPTEntropy(SSEPT):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way."""

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(SSEPTEntropy, self).__init__(config, dataset)
        self.ep_loss = EntropyLoss(dataset, self.ITEM_ID, config["entropy_alpha"] if "entropy_alpha" in config else 1)
        self.loss = BPRLoss(reduction="none")

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        user_id = interaction[self.USER_ID]
        seq_output = self.forward(item_seq, item_seq_len, user_id)
        pos_items = interaction[self.POS_ITEM_ID]
        
        neg_items = interaction[self.NEG_ITEM_ID]
        user_emb = self.user_embedding(user_id)  # [B, D]
        pos_items_emb = self.item_embedding(pos_items)
        neg_items_emb = self.item_embedding(neg_items)

        pos_items_emb = torch.cat([pos_items_emb, user_emb], dim=-1)
        neg_items_emb = torch.cat([neg_items_emb, user_emb], dim=-1)

        pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
        neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
        loss = self.loss_fct(pos_score, neg_score)

        mf_loss = self.loss_fct(pos_score, neg_score)
        loss = self.ep_loss(mf_loss, pos_items)
        
        return loss


class SSEPTEntropy2(SSEPTEntropy):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way."""

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(SSEPTEntropy2, self).__init__(config, dataset)
        self.ep_loss = EntropyLoss2(dataset, self.ITEM_ID, config["entropy_alpha"] if "entropy_alpha" in config else 1)