import torch
import torch.nn.functional as F
from tqdm import trange

from RecBole.recbole.utils.logger import set_color


class MMRReranker:
    def __init__(self, config):
        self.lambda_mmr = config["lambda_mmr"]
        self.top_k = config["topk"][0]
        self.n_items = config["mmr_n_items"]

        self.device = config["device"]
        self.config = config

    def rerank(self, interaction, scores, user_embedding, item_embedding):
        """
        Perform the MMR reranking based on relevance (score) and diversity (similarity).
        """
        scores = scores.clone().detach()

        topk_scores, topk_indices = torch.topk(scores, self.n_items, dim=1)

        similarity_matrix = self.get_cosine_similarity(item_embedding.weight)

        user = interaction[self.config["USER_ID_FIELD"]].to(self.device)
        user_e = user_embedding(user)

        all_mmr_indices = []
        # Process each user
        for u in trange(user_e.shape[0], desc=f"{set_color("MMR Reranking", 'blue')}"):
            mmr_topk = self.compute_mmr(
                similarity_matrix,
                topk_scores[u],
                topk_indices[u]
            )
            all_mmr_indices.append(mmr_topk)

        for user_id, reranked in enumerate(all_mmr_indices):
            max_score = scores[user_id].max().detach().item()
            for rank, item_id in enumerate(reranked):
                scores[user_id, item_id] = max_score + self.top_k - rank + 1

        return scores

    def compute_mmr(self, similarity_matrix, topk_scores, topk_indices):
        selected_items = []
        full_list = topk_indices.tolist()

        # Pick first item (highest score)
        first_item = full_list.pop(0)
        selected_items.append(first_item)

        # Create dict for item and score of the item
        item_to_score = {item.item(): topk_scores[id].item() for id, item in enumerate(topk_indices)}

        for _ in range(1, self.top_k):
            # Calculate relevance (score)
            relevance = torch.tensor([item_to_score[item] for item in full_list], device=self.device)

            # List to store similarity scores for each remaining item with selected items
            selected_tensor = torch.tensor(selected_items, device=self.device)
            similarity_scores = similarity_matrix[full_list][:, selected_tensor].max(dim=1).values

            # Compute MMR scores for each remaining item using relevance and similarity scores
            mmr_scores = self.lambda_mmr * relevance - (1 - self.lambda_mmr) * similarity_scores

            # Select the next item with the highest MMR score
            next_item_index = torch.argmax(mmr_scores).item()
            next_item = full_list[next_item_index]

            # Add the selected item to the list and remove from remaining
            selected_items.append(next_item)
            full_list.remove(next_item)

        return selected_items

    def get_cosine_similarity(self, all_item_e):
        norm_item_e = F.normalize(all_item_e, p=2, dim=1)
        similarity_matrix = torch.matmul(norm_item_e, norm_item_e.transpose(0, 1))
        return similarity_matrix.to(self.device)
