from typing import Any, Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl
from metrics.loss import *
from abc import ABC, abstractmethod


class BaseModelDeepGar(pl.LightningModule, ABC):
    def __init__(self,
                 input_size: int,
                 n_nodes: int,
                 distribution: Distribution,
                 test_loss: str = "rmse",
                 perform_scaling: bool = False
                 ):
        super(BaseModelDeepGar, self).__init__()
        self.input_size = input_size
        self.n_nodes = n_nodes
        self.distribution = distribution
        self.perform_scaling = perform_scaling
        self.train_loss_fn = NLL(self.distribution)
        if test_loss == 'rmse':
            self.test_loss_fn = RMSE()
        elif test_loss == 'mae':
            self.test_loss_fn = MAE()
        else:
            print('using default rmse used in deep ar')
            self.test_loss_fn = RMSE_paper()

        self.distribution_sigma = nn.Softplus()

    def training_loss(self, mu, sigma, target):
        return self.train_loss_fn.forward(mu, sigma, target)

    def test_loss(self, y_pred, y_true):
        return self.test_loss_fn.forward(y_pred, y_true)

    @abstractmethod
    def forward(self, x, *args, **kwargs):
        pass

    @abstractmethod
    def training_step(self, batch, batch_idx, *args, **kwargs):
        pass

    @abstractmethod
    def test_forward(self, x, *args, **kwargs):
        pass

    @abstractmethod
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        pass

    @abstractmethod
    def validation_step(self, batch, batch_idx, *args, **kwargs):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
