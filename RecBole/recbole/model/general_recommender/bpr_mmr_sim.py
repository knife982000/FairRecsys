import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.lambda_mmr = 0.5
        self.save_path = 'reranked_items_all_users2.pth'

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

        topk_scores, topk_indices = torch.topk(scores, k=500, dim=1)
        print("topk_scores shape:", topk_scores.shape)
        print("topk_indices shape:", topk_indices.shape)

        similarity_matrix = self.get_cosine_similarity()
        print("similarity_matrix:", similarity_matrix)

        all_mmr_indices = []

        num_users_to_process = max(2, user_e.shape[0])
        for i in range(num_users_to_process):
            print(f"\n Processing user {i}")
            mmr_topk = self.compute_mmr(
                similarity_matrix,
                all_item_e,
                topk_scores[i],
                topk_indices[i],
                top_k=20
            )
            all_mmr_indices.append(mmr_topk)
            print("Final MMR-selected items for user", i, ":", mmr_topk)

            # Update scores to reflect reranking
            reranked_scores = torch.full_like(scores[i], fill_value=-float('inf'))
            for rank, item_id in enumerate(mmr_topk[::-1]):
                reranked_scores[item_id] = rank + 1
            scores[i] = reranked_scores

        # Save reranked items for all users to a .pth file
        torch.save(all_mmr_indices, self.save_path)
        print(f"All MMR reranked items saved to {self.save_path}")

        return scores

    def compute_mmr(self, similarity_matrix, all_item_e, scores, score_indices, top_k):
        selected_items = []
        remaining_items = score_indices.tolist()

        # Pick first item (highest score)
        first_item = remaining_items.pop(0)
        selected_items.append(first_item)

        # Create dict for item and score of the item
        item_to_score = {item.item(): scores[id].item() for id, item in enumerate(score_indices)}

        for i in range(1, top_k):

            # Calculate relevance (score)
            relevance = torch.tensor([item_to_score[item] for item in remaining_items], device=all_item_e.device)

            # List to store similarity scores for each remaining item with selected items
            similarity_scores = []

            for id, item in enumerate(remaining_items):
                # Get the similarity between the current remaining item and all selected items
                selected_similarity = similarity_matrix[item, selected_items].sum()
                similarity_scores.append(selected_similarity)

            similarity_scores = torch.tensor(similarity_scores, device=all_item_e.device)

            # Compute MMR scores for each remaining item using relevance and similarity scores
            mmr_scores = self.lambda_mmr * relevance - (1 - self.lambda_mmr) * similarity_scores

            # Select the next item with the highest MMR score
            next_item_index = torch.argmax(mmr_scores).item()
            next_item = remaining_items[next_item_index]

            # Add the selected item to the list and remove from remaining
            selected_items.append(next_item)
            remaining_items.remove(next_item)

        return selected_items

    def get_cosine_similarity(self):
        # Normalize the embeddings
        norm_item_e = F.normalize(self.item_embedding.weight, p=2, dim=1)
        # Compute cosine similarity
        similarity_matrix = torch.matmul(norm_item_e, norm_item_e.T)
        return similarity_matrix


