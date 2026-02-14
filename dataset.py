import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

class ChestXrayDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, image_size=64):
        """
        Args:
            root_dir: Path to chest_xray folder
            split: 'train', 'val', or 'test'
            transform: Optional transform to apply
            image_size: Size to resize images to
        """
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        
        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                # Normalize to [0, 1] (ToTensor already does this)
            ])
        else:
            self.transform = transform
        
        # Collect all image paths
        self.image_paths = []

        # Try both uppercase and lowercase split names
        split_options = [split.upper(), split.lower(), split.capitalize()]
        split_dir = None

        for split_name in split_options:
            temp_dir = os.path.join(root_dir, split_name)
            if os.path.exists(temp_dir):
                split_dir = temp_dir
                break

        if split_dir is None:
            raise ValueError(f"Could not find split '{split}' in {root_dir}. Available: {os.listdir(root_dir)}")

        # Get both NORMAL and PNEUMONIA folders
        for category in ['NORMAL', 'PNEUMONIA']:
            category_dir = os.path.join(split_dir, category)
            if os.path.exists(category_dir):
                for img_name in os.listdir(category_dir):
                    if img_name.endswith(('.jpeg', '.jpg', '.png')):
                        self.image_paths.append(os.path.join(category_dir, img_name))

        print(f"Found {len(self.image_paths)} images in {split} set")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        if self.transform:
            image = self.transform(image)
        
        return image

def get_dataloaders(data_root, batch_size=32, image_size=64, num_workers=4):
    """
    Create train, validation, and test dataloaders
    """
    train_dataset = ChestXrayDataset(
        root_dir=data_root,
        split='train',
        image_size=image_size
    )
    
    val_dataset = ChestXrayDataset(
        root_dir=data_root,
        split='val',
        image_size=image_size
    )
    
    test_dataset = ChestXrayDataset(
        root_dir=data_root,
        split='test',
        image_size=image_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def visualize_samples(dataloader, num_samples=8):
    """
    Visualize some samples from the dataloader
    """
    images = next(iter(dataloader))
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    
    for i in range(min(num_samples, len(images))):
        img = images[i].squeeze().numpy()
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Sample {i+1}')
    
    plt.tight_layout()
    plt.savefig('outputs/dataset_samples.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved sample images to outputs/dataset_samples.png")