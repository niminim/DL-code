import os
import shutil
from glob import glob
import numpy as np
import pandas as pd
from PIL import Image
import json, csv

# Paths
source = "/home/nim/Downloads/OCT_and_X-ray/OCT2017/train"
new_data_dir = "/home/nim/Downloads/OCT_and_X-ray/OCT2017/train_split"
csv_dir = "/home/nim/venv/DL-code/Classification/OCT_Classification/data_utils/csv_splits"

# Constants
# DATA_FRACTION = 0.035
# VAL_FRACTION = 0.1

DATA_FRACTION = 0.01
VAL_FRACTION = 0.1

def get_phase_label_name(full_path):
    phase, label, name = full_path.split('/')[-3:]
    return phase, label, name

def count_all_files(directory):
    # Use glob to get a list of all files, including those in subdirectories
    all_files = glob(os.path.join(directory, '**', '*'), recursive=True)
    total_files = len(all_files)
    print(f'found {total_files} in source directory')
    return total_files

def add_new_row_to_df(full_path, df):
    phase, label, name = get_phase_label_name(full_path)
    img = Image.open(full_path)
    w, h = img.size
    new_row = {'name': name, 'phase': phase, 'label': label, 'w': w, 'h': h}
    # df.loc[len(df)] = new_row  # only use with a RangeIndex!
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    return df

def create_and_save_metadata_df(source, csv_dir):
    # creates a dataframe if all the data
    assert os.path.exists(source)
    columns = ['name', 'phase', 'label', 'w', 'h']
    df = pd.DataFrame(columns=columns)

    for root, dirs, files in os.walk(source, topdown=False):
        for name in files:
            full_path = os.path.join(root, name)
            df = add_new_row_to_df(full_path, df)
            if len(df) % 5000 == 0:
                print(f'len(df): {len(df)}')
    df.to_csv(os.path.join(csv_dir, 'train_all_files.csv'), index=False)
    print('csv file of all data - save completed')
    return df


def create_subsets_train_val(source, new_data_dir, csv_dir, data_fr=0.1, val_fr=0.15):
    # data_fr - fraction of data we want to use
    # val_fr - fraction of validation data (out of  the new data. the rest is for train --> train_fr = 1-val_fr)
    # the function gets a source path of the data, and creates 2 dataframes - train amd val, which are subsets of the original data

    assert os.path.exists(source)
    total_files = count_all_files(source)
    data_subset_name = str(data_fr).replace('.','_')

    columns = ['name', 'phase', 'label', 'w', 'h']
    df_train = pd.DataFrame(columns=columns)
    df_val = pd.DataFrame(columns=columns)

    for root, dirs, files in os.walk(source, topdown=False):
        for name in files:
            full_path = os.path.join(root, name)
            rand = np.random.uniform(0, 1)
            if rand < data_fr:
                phase, label, name = get_phase_label_name(full_path)
                rand2 = np.random.uniform(0, 1)
                if rand2 <= val_fr:
                    new_phase = 'val'
                    df_val = add_new_row_to_df(full_path, df_val)
                else:
                    new_phase = 'train'
                    df_train = add_new_row_to_df(full_path, df_train)
                new_dir = os.path.join(new_data_dir + '_' + data_subset_name, new_phase, label)
                os.makedirs(new_dir, exist_ok=True)
                new_dest = os.path.join(new_dir, name)
                shutil.copy(full_path, new_dest)

        if (len(df_train) % 1000) == 0:
            print(f'len(df_train): {len(df_train)}')
            print(f'len(df_val): {len(df_val)}')
    print('finished creating a new train and val sets')
    df_val.to_csv(csv_dir + '/val_split_' + data_subset_name + '.csv', index=False)
    df_train.to_csv(csv_dir + '/train_split_' + data_subset_name + '.csv', index=False)
    print('csv split files save completed')
    return df_train, df_val


# df = create_and_save_metadata_df(source, csv_dir)
# df = pd.read_csv(csv_dir + '/train_all_files.csv')
df_train, df_val = create_subsets_train_val(source, new_data_dir, csv_dir, data_fr=DATA_FRACTION, val_fr=VAL_FRACTION)




