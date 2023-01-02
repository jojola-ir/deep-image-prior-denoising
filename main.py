import os
import argparse
import itertools

from os.path import join

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm.auto import tqdm

from dataloader import build_data_pipes, transformations
from metrics import PSNR
from model import Unet


def evaluate(loader, model, loss_fn, device):
    model.eval()
    psnr = []
    mse = []
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)

            preds = model(data)
            target = target.to(device)

            psnr.append(PSNR(target.cpu().detach(), preds.cpu().detach()))
            mse.append(loss_fn(target.cpu().detach(), preds.cpu().detach()))

    return np.array(psnr).mean(), np.array(mse).mean()


def train(train_loader, val_loader, model, optimizer, loss_fn, epochs, device):

    for epoch in range(1, epochs+1):
        print(f"Epoch {epoch} of {epochs} - ", end="")
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)

            loss = loss_fn(model(data), target)
            train_psnr = PSNR(target.cpu().detach(), model(data).cpu().detach())

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        psnr, mse = evaluate(val_loader, model, loss_fn, device)

        print(f"train_psnr: {train_psnr} - train_mse: {loss}", end="")
        print(f" - val_psnr: {psnr} - val_mse: {mse}")


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

    train_loader = build_data_pipes(train_path, transform, noise_parameter, batch_size)
    val_loader = build_data_pipes(val_path, transform, noise_parameter, batch_size)
    test_loader = build_data_pipes(test_path, transform, noise_parameter, batch_size)

    print("Data loaded")

    model = Unet(in_channels=3, out_channels=3).to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    train(val_loader, val_loader, model, optimizer, loss_fn, num_epochs, device)

    psnr, mse = evaluate(test_loader, model, loss_fn, device)
    print(f"test_psnr : {psnr} - test_mse : {mse}")

    # save model
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    torch.save(model.state_dict(), str(results_path + "model.pth"))