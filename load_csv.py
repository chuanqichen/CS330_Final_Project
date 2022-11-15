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

def get_images(paths, labels, nb_samples=None, shuffle=True):
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
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images_labels = [
        (i, os.path.join(path, image))
        for i, path in zip(labels, paths)
        for image in sampler(os.listdir(path))
    ]
    if shuffle:
        random.shuffle(images_labels)
    return images_labels


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
        self.img_size = config.get("img_size", (28, 28))

        self.dim_input = np.prod(self.img_size)
        self.dim_output = self.num_classes

        character_folders = [
            os.path.join(data_folder, family, character)
            for family in os.listdir(data_folder)
            if os.path.isdir(os.path.join(data_folder, family))
            for character in os.listdir(os.path.join(data_folder, family))
            if os.path.isdir(os.path.join(data_folder, family, character))
        ]

        self.df = read_csv(data_folder)
        self.names_labels = self.df['golden_label'].unique()
        self.num_classes = self.df['golden_label'].nunique()
        self.num_data = self.df.shape[0]

        random.seed(1)
        random.shuffle(character_folders)
        num_val = 100
        num_train = 1100
        self.metatrain_character_folders = character_folders[:num_train]
        self.metaval_character_folders = character_folders[num_train : num_train + num_val]
        self.metatest_character_folders = character_folders[num_train + num_val :]

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

    def image_file_to_array(self, filename, dim_input):
        """
        Takes an image path and returns numpy array
        Args:
            filename: Image filename
            dim_input: Flattened shape of image
        Returns:
            1 channel image
        """
        if self.image_caching and (filename in self.stored_images):
            return self.stored_images[filename]
        image = imageio.imread(filename)  # misc.imread(filename)
        image = image.reshape([dim_input])
        image = image.astype(np.float32) / 255.0
        image = 1.0 - image
        if self.image_caching:
            self.stored_images[filename] = image
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
        sample_characters = random.sample(self.folders, self.num_classes)
        labels = np.eye(self.num_classes)
        k = self.num_samples_per_class-1
        sample_set = get_images(sample_characters, labels, nb_samples=k, shuffle=False)
        query_set = get_images(sample_characters, labels, nb_samples=1, shuffle=True)
        sample_set.extend(query_set)

        images = []
        labels = []
        for label, image_file in sample_set:
            labels.append(label)    
            images.append(self.image_file_to_array(image_file, self.dim_input))

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


if __name__ == "__main__":
    df = read_csv()
    names_labels = df['golden_label'].unique()
    print(df.head())
    print(names_labels)
