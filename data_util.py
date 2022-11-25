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
import glob

def fields_to_array(land_fields):
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

def replace_char(a):
    b = ['[', ']']
    for char in b:
        a = a.replace(char, "")
    a = [float(k) for k in a.split(",")]
    return a

def read_csv(csv_file=r"./data/v1.csv")->pd.DataFrame:
    df = pd.read_csv(csv_file)

    # drop the first index column 
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

    # sort the column orders by column names so that we can combine different csv files 
    #df = df.reindex(sorted(df.columns), axis=1)
    df = df[sorted(df.columns)]

    mc.MoveToLast(df, 'golden_label')
    df['golden_label'] = [df['golden_label'][i][3:-2] for i in range(df.shape[0])]
    print("total row: ", df.shape[0], "\t", csv_file)    
    return df

def read_csv_files(data_folder_or_file=r"./data/v1.csv"):
    if os.path.isfile(data_folder_or_file):
        df = read_csv(data_folder_or_file)
    else:
        all_files = glob.glob(os.path.join(data_folder_or_file, "*.csv"))
        df = pd.concat((read_csv(f) for f in all_files), ignore_index=True)
    
    print("total dataset: ", df.shape[0])    
    return df

class TestLoadCSV(unittest.TestCase):   
    def test_load_csv_success(self):
        df = read_csv_files(r"./data/v1.csv")
        names_labels = df['golden_label'].unique()
        print(df.columns)
        print(df.head())
        train_set= df.sample(frac=0.8,random_state=200)
        validation_test = df.drop(train_set.index)
        validation_set = validation_test.sample(frac=0.5, random_state=200)
        test_set = validation_test.drop(validation_set.index)
        print(names_labels)

    def test_load_csv2_success(self):
        df = read_csv_files(r"./data/meta_learning_part_1.csv")
        names_labels = df['golden_label'].unique()
        print(df.columns)
        print(df.head())
        names_labels = df['golden_label'].unique()
        print(names_labels)
        print(df.shape)

    def test_load_csv3_success(self):
        df1 = read_csv(r"./data/v1.csv")
        df2 = read_csv(r"./data/meta_learning_part_1.csv")
        df = pd.concat([df1, df2])                
        print(df.columns)
        print(df.head())
        names_labels = df['golden_label'].unique()
        print(names_labels)
        assert(df1.shape[0]+df2.shape[0]==df.shape[0])
        print(df.shape)
        
    def test_load_csvs_folder_success(self):
        df = read_csv_files(r"./data/")
        print(df.columns)
        print(df.head())
        names_labels = df['golden_label'].unique()
        print(names_labels)
        print(df.shape)
        

if __name__ == '__main__':
    unittest.main()