import itertools

import numpy as np
import torch
import os
from PIL import Image
from torch.utils.data import Dataset, default_collate
import torchdata.datapipes as dp
import torchvision
from torchvision import transforms
from torchvision.io import read_image


IMG_SIZE = 224


def get_noisy_image(image, noise_parameter):
    """Adds noise to an image.

    Args:
        image: image, np.array with values from 0 to 1
        noise_parameter: std of the noise
        type: type of noise, 'gaussian', 'poisson' or 'bernoulli'
    """
    noise_type = np.random.choice(['gaussian', 'poisson', 'bernoulli'])
    if noise_type == 'gaussian':
        noise = torch.normal(0, noise_parameter, image.shape)
        noisy_image = (image + noise).clip(0, 1)
    elif noise_type == 'poisson':
        a = noise_parameter * torch.ones(image.shape)
        noise = torch.poisson(a)
        noise /= noise.max()
        noisy_image = (image + noise).clip(0, 1)
    elif noise_type == 'bernoulli':
        noise = torch.bernoulli(noise_parameter * torch.ones(image.shape))
        noisy_image = (image * noise).clip(0, 1)

    return noisy_image


def transformations():
    """Applies transformations to an image.

    Args:
        image: image, np.array
    """
    transform = transforms.Compose([
        transforms.ConvertImageDtype(torch.float),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Normalize(mean=(0.5, ),
                             std=(0.5, )),
    ])
    return transform


def build_data_pipes(path_to_images, transform, noise_parameter, batch_size):
    """Builds a data pipe.

    Args:
        path_to_images: path to the folder with images
        transform: transforms to apply to images
        noise_parameter: std of the noise
        batch_size: batch size
    """
    data_pipe = dp.iter.FileLister(path_to_images, recursive=True)\
        .filter(lambda x: x.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG')))\
        .map(lambda x: (x, x))\
        .enumerate()\
        .to_map_datapipe()\
        .map(lambda x: (read_image(x[0]), read_image(x[1])))\
        .map(lambda x: (get_noisy_image(x[0], noise_parameter), x[1]))\
        .map(lambda x: (transform(x[0]), transform(x[1])))\
        .shuffle()\
        .batch(batch_size)\
        .map(lambda x: default_collate(x))
    return data_pipe


if __name__ == '__main__':
    path_to_images = "./DS/Projects/datasets/GAN/chest_xray/val/"
    noise_parameter = 0.2
    batch_size = 32

    transform = transformations()

    data_pipe = build_data_pipes(path_to_images, transform, noise_parameter, batch_size)
    for noisy_image, target in itertools.islice(data_pipe, 5):
        print(noisy_image.shape, target.shape)
        break