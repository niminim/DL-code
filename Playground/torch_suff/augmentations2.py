import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')  # or 'Qt5Agg' depending on your system


# Load an example image using PIL (or other methods, depending on your data)
image = Image.open('/home/nim/Downloads/image001.png')
print(f"Image mode: {image.mode}")
print(f"Image size: {image.size} (WxH)")


# Define a series of transformations using Compose
transform_pipeline = transforms.Compose([
    transforms.RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
    transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip the image horizontally
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
])

transform_pipeline = transforms.Compose([
    transforms.RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
    transforms.ColorJitter(brightness=.5, hue=.3),
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
])

# Apply the transformation pipeline
augmented_image_tensor = transform_pipeline(image)

# Convert the tensor back to a PIL image for visualization (optional)
transform_to_pil = transforms.ToPILImage()
augmented_image = transform_to_pil(augmented_image_tensor)

# Display the original and transformed images
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(augmented_image)
plt.title("Transformed Image")
