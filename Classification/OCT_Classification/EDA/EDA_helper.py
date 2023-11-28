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

def print_dict_of_count_files_in_sudirectories(dict):

    print('********')
    print('Print the number of images in each folder:')

    for k,v in dict.items():
        print(f'*** Phase: {k} ****')
        for sub_k, sub_v in v.items():
            print(f'class: {sub_k}, count: {sub_v}')
    print('********')

############## Plot the Class
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
    fig.savefig('//home/nim/venv/DL-code/Classification/OCT_Classification/data_dist_train_test.jpg')
    print('Image saved successfully!')


# labels, data = list(d['train'].keys()), list(d['train'].values())
# fig = plt.figure(figsize = (9,6))
# ax = fig.add_subplot(111)
# ax.plot(labels, data)
# plt.savefig('//home/nim/venv/DL-code/Classification/OCT_Classification/plot.jpg')

