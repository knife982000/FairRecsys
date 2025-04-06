import torch
import torch.nn as nn
from numpy.ma.core import append, argmax
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType


class BPRMMRSim(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(BPRMMRSim, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.lambda_mmr = 0.7 # Trade-off between relevance and diversity

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.loss = BPRLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def get_user_embedding(self, user):
        return self.user_embedding(user)

    def get_item_embedding(self, item):
        return self.item_embedding(item)

    def forward(self, user, item):
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        return user_e, item_e

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_e, pos_e = self.forward(user, pos_item)
        neg_e = self.get_item_embedding(neg_item)
        pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(user_e, neg_e).sum(dim=1)
        loss = self.loss(pos_item_score, neg_item_score)
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_e, item_e = self.forward(user, item)
        return torch.mul(user_e, item_e).sum(dim=1)

    def full_sort_predict(self, interaction):
        # Get user embeddings and item embeddings
        user = interaction[self.USER_ID]
        user_e = self.get_user_embedding(user)
        all_item_e = self.item_embedding.weight

        scores = torch.matmul(user_e, all_item_e.transpose(0, 1))  # Shape: (batch_size, num_items)

        all_mmr_indices = []

        for i in range(user_e.shape[0]):
            user_emb = user_e[i].unsqueeze(0)  # [1, embedding_dim]
            user_scores = scores[i].unsqueeze(0)  # [1, num_items]
            mmr_topk = self.compute_mmr(user_emb, all_item_e, scores, user_scores, top_k=10)
            all_mmr_indices.append(mmr_topk)
            print(all_mmr_indices)

        return mmr_topk

    def compute_mmr(self, user_emb, all_item_e, scores, user_scores, top_k):
        selected_items=[]
        candidate_indices = list(range(all_item_e.shape[0]))
        #print(candidate_indices)

        for i in range(top_k):
            mmr_values=[]

            max_relevance_idx = torch.argmax(user_scores[0][i]).item()
            print(max_relevance_idx)

            #for i in candidate_indices:
               #relevance = user_scores[0][i]

        # Cosine similarity (for diversity) between user and all items
        similarity_matrix = torch.matmul(all_item_e, all_item_e.transpose(0, 1))
         # print(similarity_matrix)
        similarities = similarity_matrix.diagonal(0, 0, 1)
        #print(similarities)

        # Compute MMR score: λ * relevance - (1 - λ) * similarity with already selected items
        mmr_scores = self.lambda_mmr * scores - (1 - self.lambda_mmr) * similarities

        return mmr_scores
