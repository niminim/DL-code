import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split


def get_transforms():
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    return transform_train, transform_test


def load_data(data_dir, batch_size, val_split):
    transform_train, transform_test = get_transforms()

    dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, transform=transform_train, download=True)
    class2index = dataset.class_to_idx

    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, transform=transform_test, download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, class2index