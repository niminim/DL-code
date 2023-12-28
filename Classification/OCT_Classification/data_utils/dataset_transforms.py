import torch
from torchvision import transforms

def get_train_transform(config):
    transform = transforms.Compose([
        transforms.Resize(config['train']['input_size']), # (h,w)
        transforms.RandomRotation((-20,20)),
        transforms.ToTensor(),
        transforms.Normalize(config['data']['preprocess']['normalize']['mean'], config['data']['preprocess']['normalize']['std'])
    ])
    return transform

def get_one_ch_transform(input_size):
    transform = transforms.Compose([
        transforms.Resize((input_size,input_size)), # (h,w)
        # transforms.RandomRotation((-20,20)),
        transforms.ToTensor(),
    ])
    return transform

def get_test_transform(config):
    transform = transforms.Compose([
        transforms.Resize(config['train']['input_size']), # (h,w)
        transforms.ToTensor(),
        transforms.Normalize(config['data']['preprocess']['normalize']['mean'], config['data']['preprocess']['normalize']['std'])
    ])
    return transform