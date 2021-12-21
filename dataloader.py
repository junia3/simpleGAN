from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
import os

def download_and_create_dataloader():
    path2data = os.path.join(os.getcwd(), "data")
    os.makedirs(path2data, exist_ok=True)
    train_dataset = datasets.MNIST(path2data, train=True,
                                   transform=transforms.Compose([transforms.ToTensor(),
                                                                 transforms.Normalize([0.5], [0.5])]), download=True)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    return train_dataloader

