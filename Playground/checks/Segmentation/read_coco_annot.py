import torch
import json
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

json_path = "/home/nim/Downloads/COCO_2017/annotations/instances_val2017.json"
imd_dir = "/home/nim/Downloads/COCO_2017/val2017"

with open(json_path, 'r') as f:
    data = json.load(f)

print(data.keys())


for k,v, in data.items():
    print('k',k)
    print('v',v)

print('info:')
print(data['info'])
print('licenses:')
print(data['licenses'])
print(f"annotations: {data['annotations'][0].keys()}")
print(f"number of annotations: {len(data['annotations'][0]['segmentation'])}")

