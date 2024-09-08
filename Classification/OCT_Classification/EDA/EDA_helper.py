import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def get_train_test_class_dist(data_dir):
    d = {'train': {},
         'test': {}
         }
    for root, dirs, files in os.walk(data_dir, topdown=False):
        for name in files:
            if not name.endswith('.jpeg'):
                continue
            img_path = os.path.join(root, name)
            phase = img_path.split('/')[-3]
            label = img_path.split('/')[-2]
            if not label in d[phase].keys():
                d[phase][label] = 1
            else:
                d[phase][label] += 1
    return d

def print_dict_of_count_files_in_subdirs(dict):

    print('********')
    print('Print the number of images in each folder:')

    for k,v in dict.items():
        print(f'*** Phase: {k} ****')
        for sub_k, sub_v in v.items():
            print(f'class: {sub_k}, count: {sub_v}')
    print('********')

############## Plot the class-distribution
def plot_class_dist(ax, categories, frequencies, title, ylabel):
    total = sum(frequencies)
    ax.bar(categories, frequencies)
    ax.set_title(title)
    ax.set_xlabel('Categories')
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(frequencies) + 50)  # Adjust the y-axis limits
    add_counts(ax, categories, frequencies)
    add_percentages(ax, categories, frequencies, total)

def add_counts(ax, categories, frequencies):
    for category, frequency in zip(categories, frequencies):
        ax.text(category, frequency, str(frequency), ha='center', va='top')

def add_percentages(ax, categories, frequencies, total):
    for category, frequency in zip(categories, frequencies):
        percentage = (frequency / total) * 100
        ax.text(category, frequency, f'{percentage:.2f}%', ha='center', va='bottom')

def plot_train_test_class_dist(d):
    # Increase the figure size by adjusting the figsize parameter
    fig, axes = plt.subplots(1, len(d), figsize=(12, 6))  # Specify width and height in inches

    for i, key in enumerate(d.keys()):
        ax = axes[i]
        categories = list(d[key].keys())
        frequencies = list(d[key].values())
        plot_class_dist(ax, categories, frequencies, f'Class Distribution - {key}', 'Frequencies')

    fig.suptitle('Class Distribution - Train and Test')  # Title for the entire figure
    fig.savefig(f'//home/nim/venv/DL-code/Classification/OCT_Classification/EDA/plots/class_dist_train_test.jpg')
    print('Image saved successfully!')
######################


# Calculate the image-size distribution
def calculate_image_shape_dist(data_dir):
    # Get the list of folders in the data directory (e.g., 'train' and 'test')
    phases = [phase for phase in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, phase))]

    # Initialize a dictionary to store the distribution
    res_dist = {} # resolution is presented as tuple
    res_str_dist = {} # resolution is presented as a string

    for phase in phases:
        print(f'phase: {phase}')
        phase_path = os.path.join(data_dir, phase)

        # Get the list of subfolders in the current class (e.g., 'A', 'B', 'C', 'D')
        classes = [class_name for class_name in os.listdir(phase_path) if os.path.isdir(os.path.join(phase_path, class_name))]

        # Initialize a dictionary for the current phase
        res_dist[phase] = {}
        res_str_dist[phase] = {}

        for class_name in classes:
            print(f'class_name: {class_name}')
            res_dist[phase][class_name] = {}
            res_str_dist[phase][class_name] = {}

            class_path = os.path.join(phase_path, class_name)

            # Get the list of image files in the current class
            image_files = [image for image in os.listdir(class_path) if image.lower().endswith(('.png', '.jpg', '.jpeg'))]

            # Calculate the distribution of image shapes
            resolutions = [Image.open(os.path.join(class_path, image)).size for image in image_files]
            res_dist[phase][class_name] = {res: resolutions.count(res) for res in set(resolutions)}

            # Accumulate counts in the overall distribution for the current dataset (train or test)
            for res in set(resolutions):
                res_name = f'W_{res[0]}_H_{res[1]}'
                print('create new - ', res_name)
                res_str_dist[phase][class_name][res_name] = {}
                if res in res_dist[phase][class_name].keys():
                    res_str_dist[phase][class_name][res_name] = res_dist[phase][class_name][res]
                else:
                    res_str_dist[phase][class_name][res_name] = 0

    return res_dist, res_str_dist


# and plot it (for a specific phase)
def plot_image_res_all_classes(res_str_dist, phase):
    assert phase in ['train', 'test']
    classes = list(res_str_dist[phase].keys())

    # Create subplots for each class
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, class_name in enumerate(classes):
        resolutions = list(res_str_dist[phase][class_name].keys())
        ax = axes[i]

        # Create a bar plot for each subfolder
        for res in resolutions:
            shapes = list(res_str_dist[phase][class_name].keys())
            counts = list(res_str_dist[phase][class_name].values())

            ax.bar(shapes, counts, alpha=0.7, label=f"Resolution: {res}")
            ax.legend(resolutions)

        ax.set_title(f"{phase.upper()} Image Size Distribution - Class: {class_name}")
        ax.set_xlabel("Image Size")
        ax.set_ylabel("Counts")

    plt.tight_layout()
    plt.show()
    fig.savefig(f'/home/nim/venv/DL-code/Classification/OCT_Classification/EDA/plots/{phase}_image_size_dist.jpg')

##
def get_all_resolutions(res_str_dist):
    classes = list(res_str_dist['train'].keys())

    all_resolutions = set()
    for class_name in classes:
        cur_resolutions = list(res_str_dist['train'][class_name].keys())
        all_resolutions.update(cur_resolutions)
    return all_resolutions


def update_resolutions(res_str_dist, all_resolutions):
    classes = list(res_str_dist['train'].keys())

    for class_name in classes:
        for res in all_resolutions:
            if res not in res_str_dist['train'][class_name].keys():
                res_str_dist['train'][class_name][res] = 0

    return res_str_dist
# all_resolutions = get_all_resolutions(res_str_dist)
# res_str_dist = update_resolutions(res_str_dist, all_resolutions)
####################

