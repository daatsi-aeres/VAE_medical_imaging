import torch
from dataset import get_dataloaders, visualize_samples
import os

# Create outputs directory
os.makedirs('outputs', exist_ok=True)

# Test dataset loading
print("Testing dataset loading...")
print("="*50)

# Adjust this path to where you extracted the data
DATA_ROOT = '/home/daatsi-aeres/UCSD_Q2_classes/ece285/vae_homework/data/chest_xray'

# Load dataloaders
train_loader, val_loader, test_loader = get_dataloaders(
    data_root=DATA_ROOT,
    batch_size=32,
    image_size=128,  # Use 128x128 for testing
    num_workers=0  # Set to 0 for debugging
)

print(f"\nTrain batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

# Get a batch and check shapes
sample_batch = next(iter(train_loader))
print(f"\nBatch shape: {sample_batch.shape}")
print(f"Min pixel value: {sample_batch.min().item():.4f}")
print(f"Max pixel value: {sample_batch.max().item():.4f}")
print(f"Mean pixel value: {sample_batch.mean().item():.4f}")

# Visualize samples
print("\nVisualizing sample images...")
visualize_samples(train_loader, num_samples=8)

print("\n✅ Dataset pipeline verified successfully!")