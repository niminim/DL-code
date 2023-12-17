import torch
from torchvision import transforms


def get_transform(input_size):
    transform = transforms.Compose([
        transforms.Resize((input_size,input_size)), # (h,w)
        transforms.RandomRotation((-20,20)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform

def get_test_transform(input_size):
    transform = transforms.Compose([
        transforms.Resize((input_size,input_size)), # (h,w)
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform