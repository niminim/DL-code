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
print(list_dirs)

d = get_train_test_class_dist(data_dir)
print_dict_of_count_files_in_subdirectories(d)
plot_train_test_class_dist(d)




########################
# Example usage
data_dir = '/home/nim//Downloads/OCT_and_X-ray/OCT2017'
res_dist, res_str_dist = calculate_image_shape_distribution(data_dir)
plot_image_res_all_classes(res_str_dist, phase='train')
plot_image_res_all_classes(res_str_dist, phase='test')

# Print the distribution
for folder, subfolders in res_dist.items():
    print(f"\n{folder} Folder:")
    for subfolder, shapes in subfolders.items():
        print(f"  {subfolder} Subfolder:")
        for shape, count in shapes.items():
            print(f"    Shape: {shape}, Count: {count}")

