from dataloader import *
from model import Generator, Discriminator
from train import *
import torch

def __main__():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloader = download_and_create_dataloader()
    print("The dataset created")
    ###########################
    params = {}
    params["noise"] = 100
    params["image_size"] = (1, 28, 28)
    params["learning rate"] = 5e-5
    params["beta1"] = 0.5
    params["beta2"] = 0.999
    params["epoch"] = 200
    ###########################

    model_gen = Generator(params).to(device)
    model_dis = Discriminator(params).to(device)

    train(params, train_dataloader, model_dis, model_gen, device=device)


if __name__ == "__main__":
    __main__()
