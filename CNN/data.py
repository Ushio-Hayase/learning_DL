import torch, torchvision


train_dataset = torchvision.datasets.MNIST(root="./CNN/data", train=True, download=True, transform=torchvision.transforms.ToTensor())
test_dataset = torchvision.datasets.MNIST(root="./CNN/data", download=True, transform=torchvision.transforms.ToTensor())
