import os
import argparse
import itertools

from os.path import join

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import random_split

from torchvision import datasets

from tqdm.auto import tqdm

from dataloader import build_data_pipes, get_noisy_image, transformations
from metrics import PSNR
from model import Unet


def evaluate(val_loader, model, loss_fn, device):
    model.eval()
    psnr = []
    mse = []
    with torch.no_grad():
        # for data, target in train_loader:
        for data, _ in val_loader:
            data = get_noisy_image(data, 0.1)
            data = data.to(device)

            preds = model(data)
            target = torch.clone(data)
            target = target.to(device)

            psnr.append(PSNR(target.cpu().detach(), preds.cpu().detach()))
            mse.append(loss_fn(target.cpu().detach(), preds.cpu().detach()))

    return np.array(psnr).mean(), np.array(mse).mean()


def train(train_loader, val_loader, model, optimizer, loss_fn, epochs, device, results_path):
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        # for data, target in train_loader:
        for data, _ in tqdm(train_loader):
            data = get_noisy_image(data, 0.1)
            data = data.to(device)

            preds = model(data)
            target = torch.clone(data)
            target = target.to(device)

            loss = loss_fn(preds, target)
            train_psnr = PSNR(target.cpu().detach(), preds.cpu().detach())
            train_psnr = train_psnr.mean()

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        psnr, mse = evaluate(val_loader, model, loss_fn, device)
        print(f"train_psnr: {train_psnr} - train_mse: {loss}", end="")
        print(f" - val_psnr: {psnr} - val_mse: {mse}")

        if epoch % 5 == 0:
            torch.save(model.state_dict(), join(results_path, f"model_{epoch}.pth"))

    torch.save(model.state_dict(), join(results_path, f"model_{epoch}.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_images", "-p", type=str, default="../../../datasets/GAN/chest_xray/")
    parser.add_argument("--epochs", "-e", type=int, default=30)
    parser.add_argument("--learning_rate", "-l", type=float, default=5e-4)
    parser.add_argument("--batch_size", "-b", type=int, default=8)
    parser.add_argument("--noise_parameter", "-n", type=float, default=0.2)
    parser.add_argument("--results", "-r", type=str, default="results/")

    args = parser.parse_args()

    dataset_path = args.path_to_images
    num_epochs = args.epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    noise_parameter = args.noise_parameter
    results_path = args.results

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    transform = transformations()

    train_path = join(dataset_path, "train")
    val_path = join(dataset_path, "val")
    test_path = join(dataset_path, "test")

    # train_loader = build_data_pipes(train_path, transform, noise_parameter, batch_size)
    # val_loader = build_data_pipes(val_path, transform, noise_parameter, batch_size)
    # test_loader = build_data_pipes(test_path, transform, noise_parameter, batch_size)

    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    print("Data loaded")

    model = Unet(in_channels=3, out_channels=3).to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    train(val_loader, val_loader, model, optimizer, loss_fn, num_epochs, device, results_path)

    # psnr, mse = evaluate(test_loader, model, loss_fn, device)
    # print(f"test_psnr : {psnr} - test_mse : {mse}")