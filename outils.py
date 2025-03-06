from typing import Tuple, Dict, Any, Optional
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from PIL import Image


class Generalized_SoftClipping(nn.Module):
    def __init__(self, s1: float = -50.0, s2: float = 50.0, s: float = 1.0, threshold: float = 20.0) -> None:
        """
        Initializes the Generalized Soft Clipping activation function.

        Args:
        s1 (float, optional): The lower bound for clipping. Defaults to -50.0.
        s2 (float, optional): The upper bound for clipping. Defaults to 50.0.
        s (float, optional): The scaling factor for the softplus function. Defaults to 1.0.
        threshold (float, optional): The threshold for the softplus function. Defaults to 20.0.
        """
        super(Generalized_SoftClipping, self).__init__()
        self.s1 = s1  
        self.s2 = s2
        self.s = s
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the generalized soft clipping activation function.

        Args:
        x (torch.Tensor): Input tensor to which the soft clipping function will be applied.

        Returns:
        torch.Tensor:
            The output tensor after applying the generalized soft clipping function.
        """
        term1 = F.softplus(x - self.s1, beta=self.s, threshold=self.threshold)
        term2 = F.softplus(x - self.s2, beta=self.s, threshold=self.threshold)
        return term1 - term2 + self.s1
    

def get_encoder_decoder(dataset: str, latent_size: int, soft_cipping_activation: bool, s: Optional[float] = 1.0) -> Tuple[nn.Module, nn.Module]:
    def get_encoder_decoder_by_type(encoder_decoder_soft_clipping, encoder_decoder_ReLU):
        if soft_cipping_activation:
            return encoder_decoder_soft_clipping(latent_size, s)
        return encoder_decoder_ReLU(latent_size)

    if dataset == 'CIFAR100':
        return get_encoder_decoder_by_type(get_cifar100_encoder_decoder_soft_clipping, get_cifar100_encoder_decoder_ReLU)
    elif dataset == 'CelebA':
        return get_encoder_decoder_by_type(get_celeba_encoder_decoder_soft_clipping, get_celeba_encoder_decoder_ReLU)
    else:
        raise ValueError(f"Dataset '{dataset}' is not supported.")


def load_dataset(params: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    transform_list = [
        transforms.ToTensor()
    ]

    if params['dataset'] == 'CIFAR100':
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transforms.Compose(transform_list))
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transforms.Compose(transform_list))
    elif params['dataset'] == 'CelebA':

        # Set paths and parameters
        data_root = './data/celeba/img_align_celeba'
        annotations_root = './data/celeba'

        transform_list.insert(0, transforms.Resize((64, 64)))  # Insert resize transformation at the beginning
        transform = transforms.Compose(transform_list)

        # Load custom datasets for training and testing
        train_dataset = CustomCelebADataset(root_dir=data_root, annotation_file=os.path.join(annotations_root, 'list_eval_partition.txt'), split='train', transform=transform)
        test_dataset = CustomCelebADataset(root_dir=data_root, annotation_file=os.path.join(annotations_root, 'list_eval_partition.txt'), split='test', transform=transform)
        
    else:
        raise ValueError(f"Dataset '{params['dataset']}' is not supported.")

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=4)

    return train_loader, test_loader


########################################################## CelebA ##########################################################
    
# Ensure the CelebA images and annotations are organized as follows:
    # - Images must be located in './data/celeba/img_align_celeba'
    # - The following annotation files must be in './data/celeba':
    #       - list_attr_celeba.txt
    #       - list_bbox_celeba.txt
    #       - list_eval_partition.txt (contains train/test split)
    #       - list_landmarks_align_celeba.txt
    # The CelebA dataset can be downloaded from: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html


class CustomCelebADataset(Dataset):
    def __init__(self, root_dir, annotation_file, split=None, transform=None):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(annotation_file, delim_whitespace=True, header=None, skiprows=1)
        self.split = split
        self.transform = transform

        # Filter dataset based on split
        if self.split == 'train':
            self.annotations = self.annotations[self.annotations[1] == 0]  # Select rows where the split is 0 (train)
        elif self.split == 'test':
            self.annotations = self.annotations[self.annotations[1] == 2]  # Select rows where the split is 2 (test)
        # Add additional conditions for validation or any other splits if needed

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


def get_celeba_encoder_decoder_soft_clipping(latent_size: int, s: float) -> Tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
    encoder = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # Input: 3x64x64, Output: 64x32x32
        Generalized_SoftClipping(s1=0.0, s2=50.0, s=s, threshold=20),
        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output: 128x16x16
        Generalized_SoftClipping(s1=0.0, s2=50.0, s=s, threshold=20),
        nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # Output: 256x8x8
        Generalized_SoftClipping(s1=0.0, s2=50.0, s=s, threshold=20),
        nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # Output: 512x4x4
        Generalized_SoftClipping(s1=0.0, s2=50.0, s=s, threshold=20),
        nn.Flatten(),
        nn.Linear(512 * 4 * 4, 1024),
        Generalized_SoftClipping(s1=0.0, s2=50.0, s=s, threshold=20)
    )
        
    fc_mu = nn.Sequential(
            nn.Linear(1024, latent_size),
            Generalized_SoftClipping(s1=-50.0, s2=50.0, s=s, threshold=20)
        )
    
    fc_logvar = nn.Sequential(      
            nn.Linear(1024, latent_size),
            Generalized_SoftClipping(s1=-20.0, s2=20.0, s=s, threshold=20) 
        )

    decoder = nn.Sequential(
        nn.Linear(latent_size, 1024),
        Generalized_SoftClipping(s1=0.0, s2=50.0, s=s, threshold=20),
        nn.Linear(1024, 512 * 4 * 4),
        Generalized_SoftClipping(s1=0.0, s2=50.0, s=s, threshold=20),
        nn.Unflatten(1, (512, 4, 4)),
        nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # Output: 256x8x8
        Generalized_SoftClipping(s1=0.0, s2=50.0, s=s, threshold=20),
        nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Output: 128x16x16
        Generalized_SoftClipping(s1=0.0, s2=50.0, s=s, threshold=20),
        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: 64x32x32
        Generalized_SoftClipping(s1=0.0, s2=50.0, s=s, threshold=20),
        nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # Output: 3x64x64
        nn.Sigmoid()
    )

    return encoder, fc_mu, fc_logvar, decoder


def get_celeba_encoder_decoder_ReLU(latent_size: int) -> Tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
    encoder = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # Input: 3x64x64, Output: 64x32x32
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output: 128x16x16
        nn.ReLU(),
        nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # Output: 256x8x8
        nn.ReLU(),
        nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # Output: 512x4x4
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(512 * 4 * 4, 1024),
        nn.ReLU()
    )
    fc_mu = nn.Linear(1024, latent_size)
    fc_logvar = nn.Linear(1024, latent_size)

    decoder = nn.Sequential(
        nn.Linear(latent_size, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512 * 4 * 4),
        nn.ReLU(),
        nn.Unflatten(1, (512, 4, 4)),
        nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # Output: 256x8x8
        nn.ReLU(),
        nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Output: 128x16x16
        nn.ReLU(),
        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: 64x32x32
        nn.ReLU(),
        nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # Output: 3x64x64
        nn.Sigmoid()
    )

    return encoder, fc_mu, fc_logvar, decoder



########################################################## CIFAR100 ##########################################################


def get_cifar100_encoder_decoder_soft_clipping(latent_size: int, s: float) -> Tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
    encoder = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # Input: 3x32x32, Output: 32x16x16
        Generalized_SoftClipping(s1=0.0, s2=50.0, s=s, threshold=20),
        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # Output: 64x8x8
        Generalized_SoftClipping(s1=0.0, s2=50.0, s=s, threshold=20),
        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output: 128x4x4
        Generalized_SoftClipping(s1=0.0, s2=50.0, s=s, threshold=20),
        nn.Flatten(),
        nn.Linear(128 * 4 * 4, 256),
        Generalized_SoftClipping(s1=0.0, s2=50.0, s=s, threshold=20),
    )

    fc_mu = nn.Sequential(
            nn.Linear(256, latent_size),
            Generalized_SoftClipping(s1=-50.0, s2=50.0, s=s, threshold=20)
        )
    
    fc_logvar = nn.Sequential(      
            nn.Linear(256, latent_size),
            Generalized_SoftClipping(s1=-20.0, s2=20.0, s=s, threshold=20) 
        )

    decoder = nn.Sequential(
        nn.Linear(latent_size, 256),
        Generalized_SoftClipping(s1=0.0, s2=50.0, s=s, threshold=20),
        nn.Linear(256, 128 * 4 * 4),
        Generalized_SoftClipping(s1=0.0, s2=50.0, s=s, threshold=20),
        nn.Unflatten(1, (128, 4, 4)),
        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: 64x8x8
        Generalized_SoftClipping(s1=0.0, s2=50.0, s=s, threshold=20),
        nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Output: 32x16x16
        Generalized_SoftClipping(s1=0.0, s2=50.0, s=s, threshold=20),
        nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # Output: 3x32x32
        nn.Sigmoid()
    )

    return encoder, fc_mu, fc_logvar, decoder


def get_cifar100_encoder_decoder_ReLU(latent_size: int) -> Tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
    encoder = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # Input: 3x32x32, Output: 32x16x16
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # Output: 64x8x8
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output: 128x4x4
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(128 * 4 * 4, 256),
        nn.ReLU()
    )
    fc_mu = nn.Linear(256, latent_size)
    fc_logvar = nn.Linear(256, latent_size)

    decoder = nn.Sequential(
        nn.Linear(latent_size, 256),
        nn.ReLU(),
        nn.Linear(256, 128 * 4 * 4),
        nn.ReLU(),
        nn.Unflatten(1, (128, 4, 4)),
        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: 64x8x8
        nn.ReLU(),
        nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Output: 32x16x16
        nn.ReLU(),
        nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # Output: 3x32x32
        nn.Sigmoid()
    )

    return encoder, fc_mu, fc_logvar, decoder


