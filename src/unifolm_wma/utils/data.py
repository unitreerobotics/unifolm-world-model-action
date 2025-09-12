import os, sys
import numpy as np
import torch
import pytorch_lightning as pl

from functools import partial
from torch.utils.data import (DataLoader, Dataset, ConcatDataset,
                              WeightedRandomSampler)
from unifolm_wma.data.base import Txt2ImgIterableBaseDataset
from unifolm_wma.utils.utils import instantiate_from_config


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # Reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id *
                                               split_size:(worker_id + 1) *
                                               split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DataModuleFromConfig(pl.LightningDataModule):

    def __init__(self,
                 batch_size,
                 train=None,
                 validation=None,
                 test=None,
                 predict=None,
                 wrap=False,
                 num_workers=None,
                 shuffle_test_loader=False,
                 use_worker_init_fn=False,
                 shuffle_val_dataloader=True,
                 train_img=None,
                 dataset_and_weights=None):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader,
                                          shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader,
                                           shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader

        self.img_loader = None
        self.wrap = wrap
        self.collate_fn = None
        self.dataset_weights = dataset_and_weights
        assert round(sum(self.dataset_weights.values()),
                     2) == 1.0, "The sum of dataset weights != 1.0"

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if 'train' in self.dataset_configs:
            self.train_datasets = dict()
            for dataname in self.dataset_weights:
                data_dir = self.dataset_configs['train']['params']['data_dir']
                transition_dir = '/'.join([data_dir, 'transitions'])
                csv_file = f'{dataname}.csv'
                meta_path = '/'.join([data_dir, csv_file])
                self.dataset_configs['train']['params'][
                    'meta_path'] = meta_path
                self.dataset_configs['train']['params'][
                    'transition_dir'] = transition_dir
                self.dataset_configs['train']['params'][
                    'dataset_name'] = dataname
                self.train_datasets[dataname] = instantiate_from_config(
                    self.dataset_configs['train'])

        # Setup validation dataset
        if 'validation' in self.dataset_configs:
            self.val_datasets = dict()
            for dataname in self.dataset_weights:
                data_dir = self.dataset_configs['validation']['params'][
                    'data_dir']
                transition_dir = '/'.join([data_dir, 'transitions'])
                csv_file = f'{dataname}.csv'
                meta_path = '/'.join([data_dir, csv_file])
                self.dataset_configs['validation']['params'][
                    'meta_path'] = meta_path
                self.dataset_configs['validation']['params'][
                    'transition_dir'] = transition_dir
                self.dataset_configs['validation']['params'][
                    'dataset_name'] = dataname
                self.val_datasets[dataname] = instantiate_from_config(
                    self.dataset_configs['validation'])

        # Setup test dataset
        if 'test' in self.dataset_configs:
            self.test_datasets = dict()
            for dataname in self.dataset_weights:
                data_dir = self.dataset_configs['test']['params']['data_dir']
                transition_dir = '/'.join([data_dir, 'transitions'])
                csv_file = f'{dataname}.csv'
                meta_path = '/'.join([data_dir, csv_file])
                self.dataset_configs['test']['params']['meta_path'] = meta_path
                self.dataset_configs['test']['params'][
                    'transition_dir'] = transition_dir
                self.dataset_configs['test']['params'][
                    'dataset_name'] = dataname
                self.test_datasets[dataname] = instantiate_from_config(
                    self.dataset_configs['test'])

        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        is_iterable_dataset = False  # NOTE Hand Code
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
            combined_dataset = []
            sample_weights = []
            for dataname, dataset in self.train_datasets.items():
                combined_dataset.append(dataset)
                sample_weights.append(
                    torch.full((len(dataset), ),
                               self.dataset_weights[dataname] / len(dataset)))
            combined_dataset = ConcatDataset(combined_dataset)
            sample_weights = torch.cat(sample_weights)
            sampler = WeightedRandomSampler(sample_weights,
                                            num_samples=len(combined_dataset),
                                            replacement=True)
            loader = DataLoader(combined_dataset,
                                sampler=sampler,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                worker_init_fn=init_fn,
                                collate_fn=self.collate_fn,
                                drop_last=True
                                )
        return loader

    def _val_dataloader(self, shuffle=False):
        is_iterable_dataset = False  # NOTE Hand Code
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
            combined_dataset = []
            sample_weights = []
            for dataname, dataset in self.val_datasets.items():
                combined_dataset.append(dataset)
                sample_weights.append(
                    torch.full((len(dataset), ),
                               self.dataset_weights[dataname] / len(dataset)))
            combined_dataset = ConcatDataset(combined_dataset)
            sample_weights = torch.cat(sample_weights)
            sampler = WeightedRandomSampler(sample_weights,
                                            num_samples=len(combined_dataset),
                                            replacement=True)
            loader = DataLoader(combined_dataset,
                                sampler=sampler,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                worker_init_fn=init_fn,
                                collate_fn=self.collate_fn)
        return loader

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = False  # NOTE Hand Code
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
            combined_dataset = []
            sample_weights = []
            for dataname, dataset in self.test_datasets.items():
                combined_dataset.append(dataset)
                sample_weights.append(
                    torch.full((len(dataset), ),
                               self.dataset_weights[dataname] / len(dataset)))
            combined_dataset = ConcatDataset(combined_dataset)
            sample_weights = torch.cat(sample_weights)
            sampler = WeightedRandomSampler(sample_weights,
                                            num_samples=len(combined_dataset),
                                            replacement=True)
            loader = DataLoader(combined_dataset,
                                sampler=sampler,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                worker_init_fn=init_fn,
                                collate_fn=self.collate_fn)
        return loader

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'],
                      Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(
            self.datasets["predict"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=init_fn,
            collate_fn=self.collate_fn,
        )

    def __len__(self):
        count = 0
        for _, values in self.train_datasets.items():
            count += len(values)
        return count
