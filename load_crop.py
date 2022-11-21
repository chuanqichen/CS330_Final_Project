"""Dataloading for Omniglot."""
import os
import glob
import random

import google_drive_downloader as gdd
import imageio
import numpy as np
import torch
from torch.utils.data import dataset, sampler, dataloader
from load_csv import read_csv, fields_to_array
import unittest
import argparse

NUM_TRAIN_CLASSES = 840
NUM_VAL_CLASSES = 80
NUM_TEST_CLASSES = 320
NUM_SAMPLES_PER_CLASS = 90
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_image(file_path):
    """Loads and transforms an Omniglot image.

    Args:
        file_path (str): file path of image

    Returns:
        a Tensor containing image data
            shape (1, 28, 28)
    """
    # x = imageio.imread(file_path)
    x = fields_to_array(file_path)
    #x = torch.tensor(x, dtype=torch.float32).reshape([1, 65, 45])
    x = torch.tensor(x, dtype=torch.float32)    
    return x


class OmniglotDataset(dataset.Dataset):
    """Omniglot dataset for meta-learning.

    Each element of the dataset is a task. A task is specified with a key,
    which is a tuple of class indices (no particular order). The corresponding
    value is the instantiated task, which consists of sampled (image, label)
    pairs.
    """

    _BASE_PATH = './omniglot_resized'
    _GDD_FILE_ID = '1iaSFXIYC3AB8q9K_M-oVMa4pmB7yKMtI'

    def __init__(self, num_support, num_query, 
                config={}, 
                device=torch.device("cpu")):
        """Inits OmniglotDataset.

        Args:
            num_support (int): number of support examples per class
            num_query (int): number of query examples per class
        """
        super().__init__()
        self.device = device
        data_folder = config.get("data_folder", r"./data/v1_a.csv")
        self.img_size = config.get("img_size", (65, 45))        
        self.dim_input = np.prod(self.img_size)
        
        self.df = read_csv(data_folder)
        self.names_labels = self.df['golden_label'].unique()
        self.total_classes = self.df['golden_label'].nunique()
        self.num_data = self.df.shape[0]

        random.seed(1)
        # shuffle dataset 
        #np.random.shuffle(self.df.values)
        np.random.default_rng(0).shuffle(self.df.values)
        
        # check problem arguments
        assert num_support + num_query <= NUM_SAMPLES_PER_CLASS
        self._num_support = num_support
        self._num_query = num_query

    def __getitem__(self, class_idxs):
        """Constructs a task.

        Data for each class is sampled uniformly at random without replacement.

        Args:
            class_idxs (tuple[int]): class indices that comprise the task

        Returns:
            images_support (Tensor): task support images
                shape (num_way * num_support, channels, height, width)
            labels_support (Tensor): task support labels
                shape (num_way * num_support,)
            images_query (Tensor): task query images
                shape (num_way * num_query, channels, height, width)
            labels_query (Tensor): task query labels
                shape (num_way * num_query,)
        """
        images_support, images_query = [], []
        labels_support, labels_query = [], []

        for label, class_idx in enumerate(class_idxs):
            # get a class's examples and sample from them
            pd_sample_lands =  [self.df.iloc[land] for land in class_idxs ]                        
            sampled_file_paths = np.random.default_rng().choice(
                pd_sample_lands,
                size=self._num_support + self._num_query,
                replace=False
            )
            images = [load_image(file_path) for file_path in sampled_file_paths]

            # split sampled examples into support and query
            images_support.extend(images[:self._num_support])
            images_query.extend(images[self._num_support:])
            labels_support.extend([label] * self._num_support)
            labels_query.extend([label] * self._num_query)

        # aggregate into tensors
        images_support = torch.stack(images_support)  # shape (N*S, C, H, W)
        labels_support = torch.tensor(labels_support)  # shape (N*S)
        images_query = torch.stack(images_query)
        labels_query = torch.tensor(labels_query)

        return images_support, labels_support, images_query, labels_query


class OmniglotSampler(sampler.Sampler):
    """Samples task specification keys for an OmniglotDataset."""

    def __init__(self, split_idxs, num_way, num_tasks):
        """Inits OmniglotSampler.

        Args:
            split_idxs (range): indices that comprise the
                training/validation/test split
            num_way (int): number of classes per task
            num_tasks (int): number of tasks to sample
        """
        super().__init__(None)
        self._split_idxs = split_idxs
        self._num_way = num_way
        self._num_tasks = num_tasks

    def __iter__(self):
        return (
            np.random.default_rng().choice(
                self._split_idxs,
                size=self._num_way,
                replace=False
            ) for _ in range(self._num_tasks)
        )

    def __len__(self):
        return self._num_tasks


def identity(x):
    return x


def get_omniglot_dataloader(
        split,
        batch_size,
        num_way,
        num_support,
        num_query,
        num_tasks_per_epoch
):
    """Returns a dataloader.DataLoader for Omniglot.

    Args:
        split (str): one of 'train', 'val', 'test'
        batch_size (int): number of tasks per batch
        num_way (int): number of classes per task
        num_support (int): number of support examples per class
        num_query (int): number of query examples per class
        num_tasks_per_epoch (int): number of tasks before DataLoader is
            exhausted
    """

    if split == 'train':
        split_idxs = range(NUM_TRAIN_CLASSES)
    elif split == 'val':
        split_idxs = range(
            NUM_TRAIN_CLASSES,
            NUM_TRAIN_CLASSES + NUM_VAL_CLASSES
        )
    elif split == 'test':
        split_idxs = range(
            NUM_TRAIN_CLASSES + NUM_VAL_CLASSES,
            NUM_TRAIN_CLASSES + NUM_VAL_CLASSES + NUM_TEST_CLASSES
        )
    else:
        raise ValueError

    return dataloader.DataLoader(
        dataset=OmniglotDataset(num_support, num_query),
        batch_size=batch_size,
        sampler=OmniglotSampler(split_idxs, num_way, num_tasks_per_epoch),
        num_workers=8,
        collate_fn=identity,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )

class TestLoadCSV(unittest.TestCase):
    @staticmethod
    def get_config():
        parser = argparse.ArgumentParser()
        parser.add_argument("--num_classes", type=int, default=5)
        parser.add_argument("--num_shot", type=int, default=1)
        parser.add_argument("--num_workers", type=int, default=1)
        parser.add_argument("--eval_freq", type=int, default=100)
        parser.add_argument("--meta_batch_size", type=int, default=128)
        parser.add_argument("--hidden_dim", type=int, default=128)
        parser.add_argument("--random_seed", type=int, default=123)
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        parser.add_argument("--train_steps", type=int, default=25000)
        parser.add_argument("--image_caching", type=bool, default=True)
        parser.add_argument("--data_folder", type=str, help="csv file name or data folders")
        config = parser.parse_args()
        return config
    
    def test_load_csv_success(self):
        df = read_csv()
        names_labels = df['golden_label'].unique()
        print(df.head())
        train_set= df.sample(frac=0.8,random_state=200)
        validation_test = df.drop(train_set.index)
        validation_set = validation_test.sample(frac=0.5, random_state=200)
        test_set = validation_test.drop(validation_set.index)
        print(names_labels)

if __name__ == '__main__':
    unittest.main()