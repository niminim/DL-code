from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

def compute_mean_std(dataset):
    # calculate the mean and std of a dataset (also valid if H!=W)
    loader = DataLoader(dataset, batch_size=100, num_workers=0, shuffle=False)
    mean = 0.0
    std = 0.0
    for images, _ in loader:
        batch_samples = images.size(0) # batch size (last batch can have smaller size)
        # Flatten the height and width dimensions into one
        images = images.view(batch_samples, images.size(1), -1)  # Shape (N, C, H*W)
        mean += images.mean(2).sum(0)  # Mean over all pixels in H*W for each channel
        std += images.std(2).sum(0)    # Standard deviation over all pixels in H*W for each channel

    mean /= len(loader.dataset)  # Average over entire dataset
    std /= len(loader.dataset)   # Average over entire dataset

    print(f"Mean: {mean}")
    print(f"Std: {std}")

    return mean, std


# Example usage with CIFAR-10
transform = transforms.Compose([transforms.ToTensor()])
cifar10_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
mean, std = compute_mean_std(cifar10_dataset)






