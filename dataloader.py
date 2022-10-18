import numpy as np
import torch
import os
from PIL import Image
from torch.utils.data import Dataset


def get_noisy_image(image, sigma):
    """Adds Gaussian noise to an image.

    Args:
        image: image, np.array with values from 0 to 1
        sigma: std of the noise
    """

    noisy_image = np.clip(image + np.random.normal(scale=sigma, size=image.shape), 0, 1).astype(np.float32)
    noisy_image_pil = np_to_pil(noisy_image)

    return noisy_image_pil


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