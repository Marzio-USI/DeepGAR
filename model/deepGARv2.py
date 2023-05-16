from typing import Any

import torch

from distributions.distributions import Distribution
from layers.graph_lstm_cell import GraphConvLSTMCell
from layers.graph_gru_cell import GraphConvGRUCell
from layers.GATConv import GATConv
from model.net import BaseModelDeepGar
import torch.nn as nn
from tsl.nn.layers import NodeEmbedding
from tsl.nn.layers.multi.recurrent import MultiLSTMCell, MultiGRUCell
from utils import scaling
from torch_geometric.nn.conv.tag_conv import TAGConv
from tsl.nn.layers.graph_convs.diff_conv import DiffConv
from torch_geometric.nn.conv.gcn_conv import GCNConv
from torch.optim.lr_scheduler import ReduceLROnPlateau
class DeepGAR(BaseModelDeepGar):
    def __init__(self, input_size: int,
                 n_nodes: int,
                 distribution: Distribution,
                 perform_scaling: bool = False,
                 encoder_size=32,
                 embedding_size=32,
                 hidden_size_1=32,
                 hidden_size_2=32
                 ):
        super().__init__(input_size, n_nodes, distribution, perform_scaling)
        self.save_hyperparameters()

        self.encoder = nn.Linear(input_size, encoder_size)
        self.node_embeddings = NodeEmbedding(self.n_nodes, embedding_size)

        self.time = MultiGRUCell(embedding_size, hidden_size_1, n_instances=self.n_nodes)
        self.space_time = GraphConvGRUCell(hidden_size_1, hidden_size_2)
        self.space_time1 = GraphConvGRUCell(hidden_size_1, hidden_size_2)

        self.distribution_mu = nn.Linear(hidden_size_2, input_size)
        self.distribution_presigma = nn.Linear(hidden_size_2, input_size)

        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2

    def forward(self, x, edge_index, edge_weight, hc, hc2):
        x_enc = self.encoder(x)
        x_emb = x_enc + self.node_embeddings()

        h1 = self.time(x_emb, hc[0])
        h2 = self.space_time(h1, hc2[0], edge_index=edge_index, edge_weight=edge_weight)
        h3 = self.space_time1(h2, hc2[1], edge_index=edge_index, edge_weight=edge_weight)

        sigma = self.distribution_presigma(h3)
        mu = self.distribution_mu(h3)
        sigma_out = self.distribution_sigma(sigma)

        return mu, sigma_out, (h1, hc[1]), (h2, h3)

    def training_step(self, batch, batch_idx, *args, **kwargs):
        x = batch["x"]
        size_batch, seq_length, _, _ = x.shape
        edge_index = batch["edge_index"]
        edge_weight = batch["edge_weight"]
        y = batch["y"]

        scale = scaling(x)
        x = x / scale

        loss = torch.zeros(1).to(device=self.device)
        rmse = torch.zeros(1).to(device=self.device)
        hidden = torch.zeros(size_batch, self.n_nodes, self.hidden_size_1).to(device=self.device)
        cell = torch.zeros(size_batch, self.n_nodes, self.hidden_size_1).to(device=self.device)

        hidden2 = torch.zeros(size_batch, self.n_nodes, self.hidden_size_2).to(device=self.device)
        cell2 = torch.zeros(size_batch, self.n_nodes, self.hidden_size_2).to(device=self.device)
        for i in range(seq_length):
            mu, sigma, (hidden, cell), (hidden2, cell2) = self.forward(x[:, i], edge_index, edge_weight, (hidden, cell),
                                                                       (hidden2, cell2))
            loss += self.training_loss(mu * torch.squeeze(scale, 1), sigma / torch.sqrt(torch.squeeze(scale, 1)),
                                       y[:, i])
            rmse += self.test_loss(mu * torch.squeeze(scale, 1), y[:, i])
        loss = loss / seq_length
        self.log("rmse", rmse / seq_length, logger=True, on_step=True, on_epoch=True, prog_bar=True,
                 batch_size=size_batch)
        return loss

    def test_forward(self, x, *args, **kwargs):
        pass

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        x = batch["x"]
        size_batch, seq_length, _, _ = x.shape
        edge_index = batch["edge_index"]
        edge_weight = batch["edge_weight"]
        y = batch["y"]

        scale = scaling(x)
        x = x / scale

        hidden = torch.zeros(size_batch, self.n_nodes, self.hidden_size_1).to(device=self.device)
        cell = torch.zeros(size_batch, self.n_nodes, self.hidden_size_1).to(device=self.device)

        hidden2 = torch.zeros(size_batch, self.n_nodes, self.hidden_size_2).to(device=self.device)
        cell2 = torch.zeros(size_batch, self.n_nodes, self.hidden_size_2).to(device=self.device)

        all_mu = []
        sigmas = []

        for i in range(seq_length):
            mu, sigma, (hidden, cell), (hidden2, cell2) = self.forward(x[:, i], edge_index, edge_weight, (hidden, cell),
                                                                       (hidden2, cell2))

        batch_size, seq_length_y, *_ = y.shape
        for i in range(seq_length_y):
            mu = mu * torch.squeeze(scale, 1)
            sigma = sigma * torch.squeeze(scale, 1) / torch.sqrt(torch.squeeze(scale, 1))
            # m = self.distribution.generate_dist(mu, sigma)
            # #TODO: fix this with monte carlo sampling
            # mu = m.sample()
            sigmas.append(sigma)
            all_mu.append(mu)

            mu, sigma, (hidden, cell), (hidden2, cell2) = self.forward(mu / torch.squeeze(scale, 1), edge_index,
                                                                       edge_weight, (hidden, cell), (hidden2, cell2))

        return all_mu, sigmas, y

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        x = batch["x"]
        size_batch, seq_length, _, _ = x.shape
        edge_index = batch["edge_index"]
        edge_weight = batch["edge_weight"]
        y = batch["y"]
        scale = scaling(x)
        x = x / scale

        loss = torch.zeros(1).to(device=self.device)
        rmse = torch.zeros(1).to(device=self.device)
        hidden = torch.zeros(size_batch, self.n_nodes, self.hidden_size_1).to(device=self.device)
        cell = torch.zeros(size_batch, self.n_nodes, self.hidden_size_1).to(device=self.device)

        hidden2 = torch.zeros(size_batch, self.n_nodes, self.hidden_size_2).to(device=self.device)
        cell2 = torch.zeros(size_batch, self.n_nodes, self.hidden_size_2).to(device=self.device)
        for i in range(seq_length):
            mu, sigma, (hidden, cell), (hidden2, cell2) = self.forward(x[:, i], edge_index, edge_weight, (hidden, cell),
                                                                       (hidden2, cell2))
            loss += self.training_loss(mu * torch.squeeze(scale, 1), sigma / torch.sqrt(torch.squeeze(scale, 1)),
                                       y[:, i])
            rmse += self.test_loss(mu * torch.squeeze(scale, 1), y[:, i])
        loss = loss / seq_length
        self.log("val_rmse", rmse / seq_length, logger=True, on_step=True, on_epoch=True, prog_bar=True,
                 batch_size=size_batch)
        self.log("val_loss", loss, batch_size=size_batch, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, min_lr=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # The metric to monitor for lr reduction
                "interval": "epoch",
                "frequency": 1,
            },
        }