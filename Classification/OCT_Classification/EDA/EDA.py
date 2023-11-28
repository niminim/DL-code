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
print_dict_of_count_files_in_sudirectories(d)
plot_train_test_class_dist(d)





######
from PIL import Image, ImageDraw

# Open an image
im = Image.open(img_p1)

# Create a drawing object
draw = ImageDraw.Draw(im)

# Draw a rectangle
draw.rectangle([50, 50, 100, 100], outline="red", width=2)

# Draw text
draw.text((10, 10), "Hello, PIL!", fill="blue")

# Save or display the modified image
im.show()

########################
import os
from PIL import Image

def calculate_image_shape_distribution(data_dir):
    # Get the list of folders in the data directory (e.g., 'train' and 'test')
    folders = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]

    # Initialize a dictionary to store the distribution
    shape_distribution = {}
    shape_count = {}

    for folder in folders:
        print(f'folder: {folder}')
        folder_path = os.path.join(data_dir, folder)

        # Get the list of subfolders in the current folder (e.g., 'A', 'B', 'C', 'D')
        subfolders = [subfolder for subfolder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, subfolder))]

        # Initialize a dictionary for the current folder
        shape_distribution[folder] = {}
        shape_count[folder] = {}

        for subfolder in subfolders:
            print(f'subfolder: {subfolder}')
            shape_distribution[folder][subfolder] = {}
            shape_count[folder][subfolder] = {}

            subfolder_path = os.path.join(folder_path, subfolder)

            # Get the list of image files in the current subfolder
            image_files = [image for image in os.listdir(subfolder_path) if image.lower().endswith(('.png', '.jpg', '.jpeg'))]

            # Calculate the distribution of image shapes
            shapes = [Image.open(os.path.join(subfolder_path, image)).size for image in image_files]
            shape_distribution[folder][subfolder] = {shape: shapes.count(shape) for shape in set(shapes)}

            # Accumulate counts in the overall distribution for the current dataset (train or test)
            for shape in set(shapes):
                name_shape_name = f'W_{shape[0]}_H_{shape[1]}'
                print('create new - ', name_shape_name)
                shape_count[folder][subfolder][name_shape_name] = {}
                if shape in shape_distribution[folder][subfolder].keys():
                    shape_count[folder][subfolder][name_shape_name] = shape_distribution[folder][subfolder][shape]
                else:
                    shape_count[folder][subfolder][name_shape_name] = 0

    return shape_distribution, shape_count


# Example usage
data_dir = '/home/nim//Downloads/OCT_and_X-ray/OCT2017'
distribution, shape_count = calculate_image_shape_distribution(data_dir)

# Print the distribution
for folder, subfolders in distribution.items():
    print(f"\n{folder} Folder:")
    for subfolder, shapes in subfolders.items():
        print(f"  {subfolder} Subfolder:")
        for shape, count in shapes.items():
            print(f"    Shape: {shape}, Count: {count}")

