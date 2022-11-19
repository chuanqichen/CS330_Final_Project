import numpy as np
import os
import random
import torch
from torch.utils.data import IterableDataset
import time
import imageio
import csv
import pandas as pd
import movecolumn as mc
import unittest
import argparse


def get_lands(pd_sample_lands, sample_lands, nb_samples=None, shuffle=True):
    """
    Takes a set of character folders and labels and returns paths to image files
    paired with labels.
    Args:
        paths: A list of character folders
        labels: List or numpy array of same length as paths
        nb_samples: Number of images to retrieve per character
    Returns:
        List of (label, image_path) tuples
    """
    if nb_samples is not None:
        #sampler = lambda x: random.sample(x, nb_samples)
        sampler = lambda x: x.sample(nb_samples, replace=False).values
    else:
        #sampler = lambda x: x
        sampler = lambda x: x.sample(1)
    lands_labels = [
        (i, land)
        for i, pd in zip(sample_lands, pd_sample_lands)
        for land in sampler(pd)
    ]
    if shuffle:
        random.shuffle(lands_labels)
    return lands_labels

class DataGenerator(IterableDataset):
    """
    Data Generator capable of generating batches of Omniglot data.
    A "class" is considered a class of omniglot digits.
    """

    def __init__(
        self,
        num_classes,
        num_samples_per_class,
        batch_type,
        config={},
        device=torch.device("cpu"),
        cache=True,
    ):
        """
        Args:
            num_classes: Number of classes for classification (N-way)
            num_samples_per_class: num samples to generate per class in one batch (K+1)
            batch_size: size of meta batch size (e.g. number of functions)
            batch_type: train/val/test
        """
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes

        data_folder = config.get("data_folder", r"./data/v1_a.csv")
        self.img_size = config.get("img_size", (65, 45))

        self.dim_input = np.prod(self.img_size)
        self.dim_output = self.num_classes

        self.df = read_csv(data_folder)
        self.names_labels = self.df['golden_label'].unique()
        self.total_classes = self.df['golden_label'].nunique()
        self.num_data = self.df.shape[0]

        random.seed(1)
        np.random.shuffle(self.df.values)
        self.train_set= self.df.sample(frac=0.8,random_state=200)
        validation_test = self.df.drop(self.train_set.index)
        self.validation_set = validation_test.sample(frac=0.5, random_state=200)
        self.test_set = validation_test.drop(self.validation_set.index)

        self.device = device
        self.image_caching = cache
        self.stored_images = {}

        if batch_type == "train":
            self.folders = self.train_set
        elif batch_type == "val":
            self.folders = self.validation_set
        else:
            self.folders = self.test_set

    def fields_to_array(self, land_fields, dim_input):
        """
        Takes an image path and returns numpy array
        Args:
            dim_input: Flattened shape of image
        Returns:
            1 channel image
        """
        #image = np.concatenate(land_fields[:-1])
        image = np.concatenate(np.vstack(land_fields[:-1]).T)
        return image

    def _sample(self):
        """
        Samples a batch for training, validation, or testing
        Args:
            does not take any arguments
        Returns:
            A tuple of (1) Image batch and (2) Label batch:
                1. image batch has shape [K+1, N, 784] and
                2. label batch has shape [K+1, N, N]
            where K is the number of "shots", N is number of classes
        Note:
            1. The numpy functions np.random.shuffle and np.eye (for creating)
            one-hot vectors would be useful.

            2. For shuffling, remember to make sure images and labels are shuffled
            in the same order, otherwise the one-to-one mapping between images
            and labels may get messed up. Hint: there is a clever way to use
            np.random.shuffle here.
            
            3. The value for `self.num_samples_per_class` will be set to K+1 
            since for K-shot classification you need to sample K supports and 
            1 query.
        """

        #############################
        #### YOUR CODE GOES HERE ####
        sample_lands = np.random.choice(self.names_labels, self.num_classes)

        pd_sample_lands = [self.folders.loc[self.folders['golden_label']==land] for land in sample_lands ]

        labels = np.eye(self.num_classes)
        k = self.num_samples_per_class-1
        sample_set = get_lands(pd_sample_lands, labels, nb_samples=k, shuffle=False)
        query_set = get_lands(pd_sample_lands, labels, nb_samples=1, shuffle=True)
        sample_set.extend(query_set)

        images = []
        labels = []
        for label, land_fields in sample_set:
            labels.append(label)    
            images.append(self.fields_to_array(land_fields, self.dim_input))

        images = torch.from_numpy(np.array(images)).float()
        labels = torch.from_numpy(np.array(labels)).long()
        images_tensor = images.reshape((self.num_samples_per_class, self.num_classes, self.dim_input))
        labels_tensor = labels.reshape((self.num_samples_per_class, self.num_classes, self.num_classes))
        assert(images_tensor.shape==(self.num_samples_per_class, self.num_classes, self.dim_input))
        assert(labels_tensor.shape==(self.num_samples_per_class, self.num_classes, self.num_classes))

        return (images_tensor, labels_tensor)
        #############################

    def __iter__(self):
        while True:
            yield self._sample()

def replace_char(a):
    b = ['[', ']']
    for char in b:
        a = a.replace(char, "")
    a = [float(k) for k in a.split(",")]
    return a

def read_csv(v1_csv_file=r"./data/v1_a.csv")->pd.DataFrame:
    df = pd.read_csv(v1_csv_file)

    # drop the first column 
    # df.drop(columns="Field Index", inplace=True)    
    df = df.iloc[: , 1:]

    columns_to_drop = ['pixel_x', 'tile_height', 'CLOUD_MASK_PNG/time_secs', 'SENTINEL2/time_secs','tile_y', 
                    'SENTINEL2/image_path', 'tile_width', 'field_id', 'CLOUD_MASK_PNG/image_path', 'tile_x', 
                    'county_fips_code', 'zoom_level', 'CLOUD_MASK_PNG/1', 'pixel_y', 'SERENUS_GEOTIFF/image_path', 
                    'SERENUS_GEOTIFF/time_secs']    
    df.drop(columns=columns_to_drop, inplace=True)   
    columns = df.columns
    
    sentinel2_cols = [col for col in columns if col.startswith("SENTINEL2")]
    serenus_geotiff_cols = [col for col in columns if col.startswith("SERENUS_GEOTIFF")]
    feature_cols = sentinel2_cols + serenus_geotiff_cols
    for column in feature_cols:
        df[column] = [replace_char(df[column][i]) for i in range(df.shape[0])]

    # normalize certain column features 
    for column in feature_cols:
        if column.endswith("ndti") or column.endswith("savi") or column.endswith("ndvi"):
            continue
        else:
            df[column] = [[k/10000 for k in df[column][i]] for i in range(df.shape[0])]

    mc.MoveToLast(df, 'golden_label')
    df['golden_label'] = [df['golden_label'][i][3:-2] for i in range(df.shape[0])]

    return df

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

    def test_data_generator_success(self):
        config = TestLoadCSV.get_config()
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        # Create Data Generator
        train_iterable = DataGenerator(
            config.num_classes,
            config.num_shot + 1,
            batch_type="train",
            device=device,
            cache=config.image_caching,
        )
        train_loader = iter(
            torch.utils.data.DataLoader(
                train_iterable,
                batch_size=config.meta_batch_size,
                num_workers=config.num_workers,
                pin_memory=True,
            )
        )

        i, l = next(train_loader)
        i, l = i.to(device), l.to(device)
        print(i)
        print(l)


if __name__ == '__main__':
    unittest.main()