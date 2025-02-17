import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.xticks([])
    plt.yticks([])
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(64, interpolation=InterpolationMode.NEAREST),  # Scale images to 64×64
        # torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),  # (Optional) randomly crop images within a scale range
        # torchvision.transforms.Grayscale(num_output_channels=1),  # (Optional) convert images to grayscale
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Apply normalization to the images
    ])
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)