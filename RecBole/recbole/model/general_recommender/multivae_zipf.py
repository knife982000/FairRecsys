r"""
MultiVAE with Zipf's penalty to reduce popularity bias.
################################################
Reference:
    Dawen Liang et al. "Variational Autoencoders for Collaborative Filtering." in WWW 2018.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import AutoEncoderMixin, GeneralRecommenderZipf
from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType


class MultiVAEZipf(GeneralRecommenderZipf, AutoEncoderMixin):
    r"""MultiVAE with Zipf's penalty to reduce popularity bias."""

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(MultiVAEZipf, self).__init__(config, dataset)

        self.layers = [600] 
        self.lat_dim = 128 
        self.drop_out = 0.5
        self.anneal_cap = 0.2
        self.total_anneal_steps = 200000

        self.build_histroy_items(dataset)

        self.update = 0

        self.encode_layer_dims = [self.n_items] + self.layers + [self.lat_dim]
        self.decode_layer_dims = [int(self.lat_dim / 2)] + self.encode_layer_dims[::-1][
            1:
        ]

        self.encoder = self.mlp_layers(self.encode_layer_dims)
        self.decoder = self.mlp_layers(self.decode_layer_dims)

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def mlp_layers(self, layer_dims):
        mlp_modules = []
        for i, (d_in, d_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            mlp_modules.append(nn.Linear(d_in, d_out))
            if i != len(layer_dims[:-1]) - 1:
                mlp_modules.append(nn.Tanh())
        return nn.Sequential(*mlp_modules)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            epsilon = torch.zeros_like(std).normal_(mean=0, std=0.01)
            return mu + epsilon * std
        else:
            return mu

    def forward(self, rating_matrix):
        h = F.normalize(rating_matrix)

        h = F.dropout(h, self.drop_out, training=self.training)

        h = self.encoder(h)

        mu = h[:, : int(self.lat_dim / 2)]
        logvar = h[:, int(self.lat_dim / 2) :]

        z = self.reparameterize(mu, logvar)
        z = self.decoder(z)
        return z, mu, logvar

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        rating_matrix = self.get_rating_matrix(user)

        self.update += 1
        if self.total_anneal_steps > 0:
            anneal = min(self.anneal_cap, 1.0 * self.update / self.total_anneal_steps)
        else:
            anneal = self.anneal_cap

        z, mu, logvar = self.forward(rating_matrix)

        # KL loss
        kl_loss = (
            -0.5
            * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            * anneal
        )

        # CE loss
        ce_loss = -(F.log_softmax(z, 1) * rating_matrix).sum(1).mean()

        loss = ce_loss + kl_loss

        # Add Zipf's penalty to the loss
        zipf_penalty = self.compute_zipf_penalty()
        loss += zipf_penalty

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        rating_matrix = self.get_rating_matrix(user)

        scores, _, _ = self.forward(rating_matrix)

        # Apply Zipf's penalty to predictions
        scores = self.apply_zipf_penalty(scores, item)

        return scores[[torch.arange(len(item)).to(self.device), item]]

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]

        rating_matrix = self.get_rating_matrix(user)

        scores, _, _ = self.forward(rating_matrix)

        # Apply Zipf's penalty to all item scores
        zipf_penalty = self.zipf_alpha * torch.log1p(self.item_popularity)
        scores = scores - zipf_penalty.unsqueeze(0)

        return scores.view(-1)
