import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType


class BPRMMR(GeneralRecommender):
    r"""BPR is a basic matrix factorization model that is trained in a pairwise way."""

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(BPRMMR, self).__init__(config, dataset)

        # Load parameters info
        self.embedding_size = config["embedding_size"]

        # Define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.loss = BPRLoss()

        # Parameters initialization
        self.apply(xavier_normal_initialization)

        # Read the ml-100k.item file
        self.data = pd.read_csv(r"C:\Users\Sandra\PycharmProjects\P6\RecBole\recbole\model\general_recommender\ml-100k.item", sep='\t', header=None,
                                names=['item_id', 'movie_title', 'release_year', 'class'])

        all_genres=self.data['class'].apply(lambda x: x.split()).explode().unique()

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(all_genres)

        # encode the classes (genres), splitting by space (in case there are multiple genres per movie)
        self.data['classes_encoded'] = self.data['class'].apply(lambda x: self.label_encoder.transform(x.split()))

        # mapping from item_id to genres
        self.item_to_genre = dict(zip(self.data['item_id'], self.data['classes_encoded']))

        # Number of items (movies) and number of unique genres
        self.n_items = len(self.data)
        self.n_genres = len(set(self.data['class'].apply(lambda x: x.split()).explode()))

        print("\nAll Item IDs:")
        item_ids = self.data['item_id'].unique()
        print(item_ids)

        print(f"Number of items (movies): {self.n_items}")
        print(f"Number of unique genres: {self.n_genres}")

        # print item-to-genre mapping
        print("\nItem-Genre Mapping:")
        item_genre_mapping = self.get_all_item_genres()
        for item_id, genres in item_genre_mapping.items():
            print(f"Item {item_id}: Genres {genres}")

        # print item classes (id and genres as string labels)
        print("\nItem Classes:")
        item_classes = self.get_item_classes()
        for item in item_classes:
            print(f"Item {item[0]}: Genres {item[1]}")

        print(f"Item 3 genres: {self.get_item_genre('3')}")
        print(f"Item 2 genres: {self.item_to_genre.get('2')}")

        # Get the genre similarity between item1 and item2
        item_similarity = self.genre_similarity('3', '2')

        print(f"Similarity between items: {item_similarity}")

    def get_item_genre(self, item_id):
        """Return the encoded genres for a given movie (item)."""
        genre = self.item_to_genre.get(item_id, None)
        if genre is None:
            print(f"Warning: Item {item_id} has no genres.")
        return genre

    def get_all_item_genres(self):
        """Return all item-to-genre mappings."""
        return self.item_to_genre

    def get_item_classes(self):
        """Return the list of class labels for each movie."""
        return self.data[['item_id', 'class']].values

    def get_user_embedding(self, user):
        """Get a batch of user embedding tensor according to input user's id."""
        return self.user_embedding(user)

    def get_item_embedding(self, item):
        """Get a batch of item embedding tensor according to input item's id."""
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
        pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(
            user_e, neg_e
        ).sum(dim=1)
        loss = self.loss(pos_item_score, neg_item_score)
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_e, item_e = self.forward(user, item)
        return torch.mul(user_e, item_e).sum(dim=1)

    def genre_similarity(self, item_id1, item_id2):
        """Computes similarity between two items."""
        genre1 = set(self.get_item_genre(item_id1))
        genre2 = set(self.get_item_genre(item_id2))

        if genre1 is None or genre2 is None:
            print("error: One of the items has no genres.")
            return 0

        intersection = genre1.intersection(genre2)
        union = genre1.union(genre2)
        similarity = len(intersection) / len(union)
        return similarity

    def mmr_rerank(self, scores: torch.Tensor, top_k: int = 50, lambda_coef: float = 0.5):
       return 0

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_embedding(user)
        all_item_e = self.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1)).view(-1)

        topk = torch.topk(score, k=100)
        topk_indices = topk.indices

        filtered_score = torch.zeros_like(score)

        filtered_score[topk_indices] = score[topk_indices]

        print("Filtered top-k score vector:", filtered_score)

        return filtered_score


