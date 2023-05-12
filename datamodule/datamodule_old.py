from tsl.datasets.mts_benchmarks import ElectricityBenchmark, ExchangeBenchmark
from tsl.data.datamodule import (SpatioTemporalDataModule,
                                 TemporalSplitter)
from tsl.data import SpatioTemporalDataset
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data.dataloader import DataLoader


class DataModule:
    def __init__(self, name, train_val_test_split: tuple[float, float, float] | None):
        if name == 'electric':
            self.dataset = ElectricityBenchmark(root='./data')
        elif name == 'exchange':
            self.dataset = ExchangeBenchmark(root='./data')
        else:
            raise NotImplementedError('choose from "electric" or "exchange"')
        if train_val_test_split is not None:
            assert np.sum(train_val_test_split) == 1, f'invalid split'
            self.train_size = train_val_test_split[0]
            self.val_size = train_val_test_split[1]
            self.test_size = train_val_test_split[2]
        else:
            self.train_size = 0.7
            self.val_size = 0.15
            self.test_size = 0.15

        self.train_dataset = None
        self.test_dataset = None

    def generate_train_test_dataset(self, window=24, test_window=24, test_horizon=4, test_stride=4, test_delay=0):
        connectivity = self.dataset.get_connectivity(
            method='full',
            threshold=0.1,
            include_self=False,
            normalize_axis=1,
            knn=4,
            layout="edge_index")
        dataframe = self.dataset.dataframe()
        train_df, test_df = train_test_split(dataframe, test_size=self.test_size, shuffle=False)
        mask_train, mask_test = self.dataset.mask[:train_df.shape[0], ...], self.dataset.mask[train_df.shape[0]:, ...]

        self.train_dataset : SpatioTemporalDataset  = SpatioTemporalDataset(
            target=train_df,
            connectivity=connectivity,
            mask=mask_train,
            horizon=window,
            window=window,
            stride=1,
            delay=-(window-1),
        )
        self.test_dataset : SpatioTemporalDataset= SpatioTemporalDataset(
            target=test_df,
            connectivity=connectivity,
            mask=mask_test,
            window=test_window,
            horizon=test_horizon,
            stride=test_stride,
            delay=test_delay
        )

    def get_training_data_module(self):
        splitter = TemporalSplitter(val_len=self.val_size, test_len=0.0)
        dm = SpatioTemporalDataModule(
            dataset=self.train_dataset,
            scalers=None,
            splitter=splitter,
            batch_size=32,
            workers=8,
        )
        dm.setup()
        return dm

    def get_channels(self):
        return self.train_dataset.n_channels

    def get_number_of_nodes(self):
        return self.train_dataset.n_nodes

    def get_horizon_value(self):
        return self.train_dataset.horizon

    def get_testing_dataloader(self):
        dm_test = SpatioTemporalDataModule(
            dataset=self.test_dataset,
            scalers=None,
            batch_size=1,
            workers=8
        )
        dm_test.setup()
        return dm_test.get_dataloader(shuffle=False)

    def get_all(self,window=24, test_window=24, test_horizon=4, test_stride=4, test_delay=0) -> tuple[SpatioTemporalDataModule, DataLoader]:
        self.generate_train_test_dataset(window, test_window, test_horizon, test_stride, test_delay)
        return self.get_training_data_module(), self.get_testing_dataloader()