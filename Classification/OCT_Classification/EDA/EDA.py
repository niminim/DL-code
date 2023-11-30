import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')  # or 'Qt5Agg' depending on your system

sys.path.append('/home/nim/venv/DL-code/Classification/OCT_Classification/EDA')
from EDA_helper import *

data_dir = "/home/nim/Downloads/OCT_and_X-ray/OCT2017"
list_dirs = os.listdir(data_dir)

# class distribution
d = get_train_test_class_dist(data_dir) # dict with number of images of each class (train and test)
print_dict_of_count_files_in_subdirectories(d) # prints the distribution
plot_train_test_class_dist(d) # plots the class-distribution in the train and test sets

# image-size distribution
res_dist, res_str_dist = calculate_image_shape_distribution(data_dir) # calculates the image-size dist for each class
plot_image_res_all_classes(res_str_dist, phase='train') # plots the image-size dist for the train-set
plot_image_res_all_classes(res_str_dist, phase='test') # plots the image-size dist for the test-set

# Print the distribution
for folder, subfolders in res_dist.items():
    print(f"\n{folder} Folder:")
    for subfolder, shapes in subfolders.items():
        print(f"  {subfolder} Subfolder:")
        for shape, count in shapes.items():
            print(f"    Shape: {shape}, Count: {count}")

