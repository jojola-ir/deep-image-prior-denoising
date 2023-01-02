import numpy as np
import torch
import os
from PIL import Image
from torch.utils.data import Dataset


def get_noisy_image(image, noise_parameter, type='gaussian'):
    """Adds noise to an image.

    Args:
        image: image, np.array with values from 0 to 1
        noise_parameter: std of the noise
        type: type of noise, 'gaussian', 'poisson' or 'bernoulli'
    """
    if type == 'gaussian':
        noise = torch.normal(0, noise_parameter, image.shape)
        noisy_image = (image + noise).clip(0, 1)
    elif type == 'poisson':
        a = noise_parameter * torch.ones(image.shape)
        noise = torch.poisson(a)
        noise /= noise.max()
        noisy_image = (image + noise).clip(0, 1)
    elif type == 'bernoulli':
        noise = torch.bernoulli(noise_parameter * torch.ones(image.shape))
        noisy_image = (image * noise).clip(0, 1)
    else:
        raise ValueError('Unknown type of noise')

    return noisy_image


class DataLoader(Dataset):
    """Custom loader for datas."""

    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.transform = transform
        self.images = os.listdir(image_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image_dir = os.path.join(self.image_path, self.images[item])
        image = Image.open(image_dir).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        target = image
        image = get_noisy_image(image) 