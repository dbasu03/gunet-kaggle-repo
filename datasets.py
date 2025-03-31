# datasets.py (updated)
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class PairLoader(Dataset):
    def __init__(self, data_dir, mode='train', patch_size=256, edge_decay=0, data_augment=True, cache_memory=False):
        self.data_dir = data_dir
        self.mode = mode
        self.patch_size = patch_size
        self.edge_decay = edge_decay
        self.data_augment = data_augment
        self.cache_memory = cache_memory

        self.hazy_dir = os.path.join(data_dir, 'IN')
        self.clear_dir = os.path.join(data_dir, 'GT')
        self.image_list = sorted(os.listdir(self.hazy_dir))

        # Transform to normalize and resize images
        self.transform = transforms.Compose([
            transforms.Resize((patch_size, patch_size)),  # Resize to fixed patch_size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        hazy_path = os.path.join(self.hazy_dir, self.image_list[idx])
        clear_path = os.path.join(self.clear_dir, self.image_list[idx])

        hazy_img = Image.open(hazy_path).convert('RGB')
        clear_img = Image.open(clear_path).convert('RGB')

        # Apply the transform (which includes resizing)
        hazy_img = self.transform(hazy_img)
        clear_img = self.transform(clear_img)

        if self.mode == 'train' and self.data_augment:
            if torch.rand(1) > 0.5:
                hazy_img = torch.flip(hazy_img, dims=[2])
                clear_img = torch.flip(clear_img, dims=[2])
            if torch.rand(1) > 0.5:
                hazy_img = torch.flip(hazy_img, dims=[1])
                clear_img = torch.flip(clear_img, dims=[1])

        return {'source': hazy_img, 'target': clear_img}
