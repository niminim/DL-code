import torch
import torchvision
import torchaudio

import timm
import huggingface_hub
import transformers
import kornia
import skorch
import lightning as lt
import sklearn

import numpy as np
import scipy as sc

import pandas as pd

import cv2
#import tensorflow as tf

print('Current Package Versions')
print(f'torch:  {torch.__version__}, CUDA: {torch.cuda.is_available()}')
print('torchvision: ', torchvision.__version__)
print('torchaudio: ', torchaudio.__version__)
print('timm: ', timm.__version__)
print('huggingface_hub: ', huggingface_hub.__version__)
print('transformers: ', transformers.__version__)
print('kornia: ', kornia.__version__)
print('skorch : ', skorch.__version__)
print('lightning: ', lt.__version__)
print('sklearn: ', sklearn.__version__)

print('numpy: ', np.__version__)
print('scipy: ', sc.__version__)
print('pandas: ', pd.__version__)

print('opencv: ', cv2.__version__)
#print('tensorflow: ', tf.__version__)


