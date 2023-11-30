import os
import shutil
from glob import glob
import numpy as np
import pandas as pd
from PIL import Image
import json, csv

source = "/home/nim/Downloads/OCT_and_X-ray/OCT2017/train"
csv_dir = "/home/nim/venv/DL-code/Classification/OCT_Classification/data_utils"

def count_all_files(directory):
    # Use glob to get a list of all files, including those in subdirectories
    all_files = glob(os.path.join(directory, '**', '*'), recursive=True)
    total_files = len(all_files)
    print(f'found {total_files} in source directory')
    return total_files

def add_new_row_to_df(full_path, df):
    phase = full_path.split('/')[-3]
    label = full_path.split('/')[-2]
    name = full_path.split('/')[-1]
    img = Image.open(full_path)
    w, h = img.size
    new_row = {'name': name, 'phase': phase, 'label': label, 'w': w, 'h': h}
    # df.loc[len(df)] = new_row  # only use with a RangeIndex!
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    return df

def create_and_save_df_of_metadata(source, csv_dir):
    assert os.path.exists(source)
    columns = ['name', 'phase', 'label', 'w', 'h']
    df = pd.DataFrame(columns=columns)

    for root, dirs, files in os.walk(source, topdown=False):
        for name in files:
            full_path = os.path.join(root, name)
            df = add_new_row_to_df(full_path, df)
            if len(df) % 5000 == 0:
                print(f'len(df): {len(df)}')
    print(df.head())
    df.to_csv(os.path.join(csv_dir, 'train_all_files.csv'))
    print('csv file save completed')
    return df


def create_train_val_sets(source, csv_dir, data_fr=0.1, val_fr=0.15):
    assert os.path.exists(source)
    total_files = count_all_files(source)

    columns = ['name', 'phase', 'label', 'w', 'h']
    df_train = pd.DataFrame(columns=columns)
    df_val = pd.DataFrame(columns=columns)

    for root, dirs, files in os.walk(source, topdown=False):
        for name in files:
            full_path = os.path.join(root, name)
            rand = np.random.uniform(0, 1)
            if rand < data_fr:
                rand2 = np.random.uniform(0, 1)
                if rand2 < val_fr:
                    df_val = add_new_row_to_df(full_path, df_val)
                else:
                    df_train = add_new_row_to_df(full_path, df_train)

        if (len(df_train) % 1000) == 0:
            print(f'len(df_train): {len(df_train)}')
            print(f'len(df_val): {len(df_val)}')
    df_val.to_csv(csv_dir + '/val_split.csv')
    df_train.to_csv(csv_dir + '/train_split.csv')
    return df_train, df_val

df = create_and_save_df_of_metadata(source, csv_dir)
df = pd.read_csv(csv_dir + '/train_all_files.csv')
df_train, df_val = create_train_val_sets(source, csv_dir, data_fr=0.1, val_fr=0.15)
print('df_train.head():')
print(df_train.head())
print('df_val.head():')
print(df_val.head())












