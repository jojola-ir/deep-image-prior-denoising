import argparse

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

            psnr.extend(PSNR(target.cpu().detach(), preds.cpu().detach()))
            mse.extend(loss_fn(target.cpu().detach(), preds.cpu().detach()))

    return np.array(psnr).mean(), np.array(mse).mean()


def train(train_loader, val_loader, model, optimizer, loss_fn, epochs, device):

    for epoch in tqdm(range(1, epochs+1)):
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)

            loss = loss_fn(model(data), targets)
            train_psnr = PSNR(target.cpu().detach(), model(data).cpu().detach())

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        psnr, mse = evaluate(val_loader, model, loss_fn, device)

        print(f"epoch: {epoch} - train_psnr: {train_psnr} - train_mse: {loss}", end="")
        print(f" - val_psnr: {psnr} - val_mse: {mse}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_images", "-p", type=str, default="../../../datasets/GAN/chest_xray/")
    parser.add_argument("--epochs", "-e", type=int, default=30)
    parser.add_argument("--learning_rate", "-l", type=float, default=5e-4)
    parser.add_argument("--batch_size", "-b", type=int, default=16)
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

    transform = transformations()

    data_pipe = build_pipeline(dataset_path, transform, noise_parameter, batch_size)

    train_loader, val_loader = data_pipe

    # model = Unet(in_channels=3, out_channels=3).to(device=device)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # loss_fn = nn.MSELoss()
    #
    # train(data_pipe, model, optimizer, loss_fn, scaler)
    #
    # # save model
    # torch.save(model.state_dict(), "model.pth")