from typing import Optional, Union, Tuple, List, Mapping, Literal, Iterable, Callable, Iterator

import torch
from torch import Tensor
from tsl.data.datamodule.spatiotemporal_datamodule import StageOptions
from tsl.data.preprocessing import Scaler, ScalerModule
from tsl.datasets.mts_benchmarks import ElectricityBenchmark, ExchangeBenchmark
from tsl.data.datamodule import (SpatioTemporalDataModule,
                                 TemporalSplitter)
from tsl.data import SpatioTemporalDataset, BatchMap, SynchMode, Data, Splitter
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data.dataloader import DataLoader
from copy import deepcopy
from torch.utils.data import BatchSampler
from tsl.typing import SparseTensArray, DataArray, TensArray, IndexSlice, TemporalIndex
from tsl.datasets.elergone import Elergone
from utils import pre_scaling
from tsl.datasets.prototypes import DatetimeDataset
import tsl
import os
import pandas as pd
from tsl.ops.connectivity import adj_to_edge_index

class DataModule:
    def __init__(self, name, train_val_test_split: tuple[float, float, float] | None, batch_size:int=32):
        if name == 'electric':
            self.dataset =HourlyElergone(root='./data')
        elif name == 'exchange':
            self.dataset = ExchangeBenchmark(root='./data')
        else:
            raise NotImplementedError('choose "electric" or "exchange"')
        if train_val_test_split is not None:
            assert np.sum(train_val_test_split) == 1, f'invalid split'
            self.train_size = train_val_test_split[0]
            self.val_size = train_val_test_split[1]
            self.test_size = train_val_test_split[2]
        else:
            self.train_size = 0.7
            self.val_size = 0.15
            self.test_size = 0.15

        self.batch_size = batch_size
        self.name = name
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

    def generate_train_test_dataset(self, window=24, test_window=24, test_horizon=4, test_stride=4, test_delay=0):
        # connectivity = self.dataset.get_connectivity(
        #     method='full',
        #     threshold=0.1,
        #     include_self=False,
        #     normalize_axis=1,
        #     knn=4,
        #     layout="edge_index")
        connectivity = self.dataset.dataframe().corr().values
        connectivity = adj_to_edge_index(connectivity)
        dataframe = self.dataset.dataframe()
        if self.name == 'electric':
            testing_part = 7 * 24 + (test_window)
            train_df = dataframe.iloc[:-testing_part]
            mask_train = self.dataset.mask[:-testing_part]
            test_df = dataframe.iloc[-testing_part:]
            mask_test = self.dataset.mask[-testing_part:]
            assert test_df.shape[0] == testing_part
            test_size = train_df.shape[0] / dataframe.shape[0]
            val_size = 0.1 / test_size
            train_df, val_df = train_test_split(train_df, test_size=val_size, shuffle=False)
            mask_train, mask_val = mask_train[:train_df.shape[0], ...], mask_train[train_df.shape[0]:, ...]
            print(f'Train {train_df.shape}, val {val_df.shape}, test {test_df.shape}. ORIGINAL {dataframe.shape}')
        else:
            train_df, test_df = train_test_split(dataframe, test_size=self.test_size, shuffle=False)
            mask_train, mask_test = self.dataset.mask[:train_df.shape[0], ...], self.dataset.mask[train_df.shape[0]:, ...]
            self.val_size /= self.train_size
            train_df, val_df = train_test_split(train_df, test_size=self.val_size, shuffle=False)
            mask_train, mask_val = mask_train[:train_df.shape[0], ...], mask_train[train_df.shape[0]:, ...]

        assert dataframe.shape[0] == (train_df.shape[0] + val_df.shape[0] + test_df.shape[0]), f'Splitting failed'

        self.train_dataset: SpatioTemporalDataset = SpatioTemporalDataset(
            target=train_df,
            connectivity=connectivity,
            mask=mask_train,
            horizon=window,
            window=window,
            stride=1,
            delay=-(window - 1),
        )

        self.val_dataset: SpatioTemporalDataset = SpatioTemporalDataset(
            target=val_df,
            connectivity=connectivity,
            mask=mask_val,
            horizon=window,
            window=window,
            stride=1,
            delay=-(window - 1),
        )

        self.test_dataset: SpatioTemporalDataset = SpatioTemporalDataset(
            target=test_df,
            connectivity=connectivity,
            mask=mask_test,
            window=test_window,
            horizon=test_horizon,
            stride=test_stride,
            delay=test_delay
        )

    def get_training_data_loader(self):
        dm = CustomSpatioTemporalDataModule(
            dataset=self.train_dataset,
            scalers=None,
            splitter=None,
            batch_size=self.batch_size,
            workers=8,
        )
        dm.setup()
        scal = self.get_weights()
        sampler = WeightedSampler(scal)
        train = dm.get_dataloader_train(shuffle=False, batch_size=self.batch_size, sampler=sampler)
        return train

    def get_val_data_loader(self):
        dm = SpatioTemporalDataModule(
            dataset=self.val_dataset,
            scalers=None,
            splitter=None,
            batch_size=self.batch_size,
            workers=8,
        )
        dm.setup()
        return dm.get_dataloader(None, shuffle=False, batch_size=self.batch_size)

    def get_testing_data_loader(self):
        dm_test = SpatioTemporalDataModule(
            dataset=self.test_dataset,
            scalers=None,
            batch_size=1,
            workers=8
        )
        dm_test.setup()
        return dm_test.get_dataloader(None, shuffle=False, batch_size=1)

    def get_weights(self):
        indices = self.train_dataset._indices
        train_dataset = deepcopy(self.train_dataset)
        scal = []
        for i in range(0, indices[-1]):
            inp = train_dataset[i].input.x
            s = pre_scaling(inp)
            max_s = torch.mean(s)
            scal.append(max_s)
        return torch.as_tensor(scal)

    def get_channels(self):
        return self.train_dataset.n_channels

    def get_number_of_nodes(self):
        return self.train_dataset.n_nodes

    def get_horizon_value(self):
        return self.train_dataset.horizon

    def get_all(self, window=24, test_window=24, test_horizon=4, test_stride=4, test_delay=0):
        self.generate_train_test_dataset(window, test_window, test_horizon, test_stride, test_delay)
        return self.get_training_data_loader(), self.get_val_data_loader(), self.get_testing_data_loader()


from torch.utils.data import DataLoader, Dataset, Sampler, RandomSampler


class WeightedSampler(Sampler):
    def __init__(self, weights):
        self.weights = weights

        self.num_samples = weights.shape[0]
        self.replacement = True
        self.validate_weights()

    def validate_weights(self):
        if torch.sum(self.weights) > 1:
            self.weights /= torch.sum(self.weights)

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, self.replacement).tolist())

    def __len__(self):
        return self.num_samples


class CustomSpatioTemporalDataModule(SpatioTemporalDataModule):
    def __init__(self, dataset: SpatioTemporalDataset, scalers: Optional[Mapping] = None, mask_scaling: bool = True,
                 splitter: Optional[Splitter] = None, batch_size: int = 32, workers: int = 0, pin_memory: bool = False):
        super().__init__(dataset, scalers, mask_scaling, splitter, batch_size, workers, pin_memory)

    def get_dataloader_train(self, split: Literal['train', 'val', 'test'] = None, shuffle: bool = False,
                             batch_size: Optional[int] = None, sampler: WeightedSampler = None) -> Optional[DataLoader]:
        if split is None:
            dataset = self.torch_dataset
        elif split in ['train', 'val', 'test']:
            dataset = getattr(self, f'{split}set')
        else:
            raise ValueError("Argument `split` must be one of "
                             "'train', 'val', or 'test'.")
        if dataset is None:
            return None
        assert sampler is not None
        # pin_memory = self.pin_memory if split == 'train' else None
        return CustomStaticGraphLoader(dataset,
                                       batch_size=batch_size or self.batch_size,
                                       shuffle=shuffle,
                                       drop_last=False,
                                       num_workers=self.workers,
                                       pin_memory=self.pin_memory,
                                       batch_sampler=BatchSampler(sampler, batch_size=batch_size,
                                                                  drop_last=False)
                                       )

    def train_dataloader_custom(self, shuffle: bool = False,
                                batch_size: Optional[int] = None, sampler: WeightedSampler = None) \
            -> Optional[DataLoader]:
        """"""
        return self.get_dataloader_train('train', shuffle, batch_size, sampler)


from tsl.data import SpatioTemporalDataset


def _dummy_collate(x):
    return x


class CustomStaticGraphLoader(DataLoader):
    r"""A data loader for getting temporal graph signals of type
    :class:`~tsl.data.Batch` on a shared (static) topology.

    This loader exploits the efficient indexing of
    :class:`~tsl.data.SpatioTemporalDataset` to get multiple items at once,
    by using a :class:`torch.utils.data.BatchSampler`.

    Args:
        dataset (SpatioTemporalDataset): The dataset from which to load the
            data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If :obj:`True`, then data will be
            reshuffled at every epoch.
            (default: :obj:`False`)
        drop_last (bool, optional): If :obj:`True`, then drop the last
            incomplete batch if the dataset size is not divisible by the batch
            size (which will be smaller otherwise).
            (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """

    def __init__(self,
                 dataset: SpatioTemporalDataset,
                 batch_size: Optional[int] = 1,
                 shuffle: bool = False,
                 drop_last: bool = False,
                 batch_sampler: BatchSampler = None,
                 **kwargs):
        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']
        self.batch_sampler = batch_sampler
        super().__init__(dataset,
                         collate_fn=_dummy_collate,
                         batch_sampler=batch_sampler,
                         **kwargs)

    @property
    def _auto_collation(self):
        return False

    @property
    def _index_sampler(self):
        return self.batch_sampler


class HourlyElergone(Elergone):
    def __init__(self, root=None, freq=None):
        super().__init__(root, freq)
    
    def load(self):
        df = self.load_raw()
        tsl.logger.info('Loaded raw dataset.')
        df /= 4.  # kW -> kWh
        df = df[~df.index.duplicated(keep='first')]
        start_date = pd.to_datetime('2014-01-01')
        end_date = pd.to_datetime('2014-09-02') + pd.Timedelta(days=7)
        df = df.loc[start_date:end_date]
        # Only include data starting from 2014-01-01
        df = df.resample('H').sum()  # Convert data frequency to hourly by summing
        # drop duplicates
        df = df.fillna(0.)
        mask = (df.values != 0.).astype('uint8')
        return df, mask

