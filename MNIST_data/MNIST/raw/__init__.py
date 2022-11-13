# Load MNIST dataset
from torchvision import datasets, transforms
import torch

def create_dataloader(opt):
    train_data = datasets.MNIST(root = './../MNIST_data/',
                                train=True,
                                download=True,
                                transform=transforms.ToTensor())

    test_data = datasets.MNIST(root = './../MNIST_data/',
                                train=False,
                                download=True,
                                transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data, batch_size=opt.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data, batch_size=opt.batch_size, shuffle=True
    )
    
    return train_loader, test_loader