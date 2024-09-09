import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import cv2

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')  # or 'Qt5Agg' depending on your system
# matplotlib.use('TkAgg')  # or 'TkAgg' for interactive mode without Qt


sys.path.append('/home/nim/venv/DL-code/Classification/OCT_Classification/EDA')

def get_pil_img_metadata(img):
    W, H = img.size
    mode = img.mode
    format = img.format
    # Print image metadata
    print('Image meta-data:')
    print(f'size - H: {H}, W: {W}')
    print(f'mode: {mode}')
    print(f'format: {format}')

    # # could get image metadata from:
    # image_info = image.info
    # print("Image Size:", image_info.get("size"))
    return W, H, mode, format


################################## Images
img_path = '/home/nim/Downloads/Data/OCT2017/test/CNV/CNV-81630-3.jpeg'
### PIL Image
img = Image.open(img_path) # type(img) = PIL Image
W, H, mode, format = get_pil_img_metadata(img)
print(f'size - H: {H}, W: {W}')
img.show() # Show with PIL
plt.imshow(img) # show with matplotlib (viewer) (note below)

# PIL.Image and matplotlib.pyplot.imshow work with images in different formats
# matplotlib needs the image in a format that it can handle — typically a NumPy array.
# To fix this, you can convert the PIL.Image to a NumPy array before using plt.imshow()

img_np = np.array(img)
# Display the image using matplotlib
plt.imshow(img_np)
plt.show()



### opencv Image
img_cv2 = cv2.imread(img_path) # type(img) = np.ndarray, img.dtype = uint8
H, W, Ch = img_cv2.shape
print(f'opencv image: size - H: {H}, W: {W}, Ch: {Ch}')
plt.imshow(img_cv2)
plt.show()


# Numpy data
img_data = np.asarray(img)
H, W = img_data.shape
print(f'np_data image: size - H: {H}, W: {W}, Ch: {Ch}')
print(f'dtype: {img_data.dtype}')
plt.imshow(img_data)
plt.show()


# Count pixels
pixels = list(img.getdata())
width, height = img.size
pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]

##### Color images
img_p1 = '/home/nim/Downloads/dog1.jpg'
img_p2 = '/home/nim/Downloads/dog2.jpeg'
img_p3 = '/home/nim/Downloads/dog3.jpg'

im = Image.open(img_p3)
im.show()
plt.imshow(im)

_,_,_,_ = get_pil_img_metadata(im)

