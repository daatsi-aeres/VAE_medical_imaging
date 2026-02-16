import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob

class ChestXrayDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, image_size=128):
        """
        Args:
            root_dir: Path to chest_xray folder
            split: 'train', 'val', or 'test'
            transform: Optional transform to apply
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data = [] # Will store tuples: (image_path, label)

        # 1. Find the correct split folder (handle case sensitivity)
        split_dir = None
        for s in [split, split.upper(), split.lower(), split.capitalize()]:
            temp_path = os.path.join(root_dir, s)
            if os.path.exists(temp_path):
                split_dir = temp_path
                break
        
        if split_dir is None:
            raise ValueError(f"Could not find split '{split}' in {root_dir}")

        # 2. Load Images and Assign Labels
        # 0 = NORMAL, 1 = PNEUMONIA
        categories = {'NORMAL': 0, 'PNEUMONIA': 1}
        
        for category, label in categories.items():
            cat_dir = os.path.join(split_dir, category)
            if os.path.exists(cat_dir):
                # Recursively find all images
                image_files = []
                for ext in ['*.jpeg', '*.jpg', '*.png', '*.JPG']:
                    image_files.extend(glob.glob(os.path.join(cat_dir, ext)))
                
                for img_path in image_files:
                    self.data.append((img_path, label))

        print(f"Found {len(self.data)} images in {split} set")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        
        # Open as Grayscale (1 channel)
        try:
            image = Image.open(img_path).convert('L')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank black image if file is corrupt
            image = Image.new('L', (128, 128))

        if self.transform:
            image = self.transform(image)
        else:
            # Default transform if none provided
            default_trans = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                # Note: No normalization needed if using Sigmoid in VAE
            ])
            image = default_trans(image)
        
        return image, label

def get_dataloaders(data_root, batch_size=32, image_size=128, num_workers=2):
    """
    Create train, validation, and test dataloaders
    """
    # Transform: Resize -> Tensor
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    
    # We use 'test' set for validation because the official 'val' set is too small (16 images)
    train_dataset = ChestXrayDataset(data_root, split='train', transform=transform)
    val_dataset = ChestXrayDataset(data_root, split='test', transform=transform) 
    test_dataset = ChestXrayDataset(data_root, split='test', transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    
    return train_loader, val_loader, test_loader