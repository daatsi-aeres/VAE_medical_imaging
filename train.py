import os
os.environ['MPLBACKEND'] = 'Agg'  # Use non-interactive backend

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime

# Speed optimizations
torch.backends.cudnn.benchmark = True  # Auto-tune convolution algorithms
torch.set_float32_matmul_precision('high')  # Faster matmul on Ampere GPUs

from vae_model import VAE, vae_loss
from dataset import get_dataloaders

# Hyperparameters - OPTIMIZED for RTX 4060 Laptop
HYPERPARAMS = {
    'image_size': 128,
    'latent_dim': 256,        # Increased from 128 (better representations)
    'batch_size': 80,         # Increased from 32 (faster training, ~4-5GB VRAM)
    'learning_rate': 2e-3,    # Slightly lower for stability
    'num_epochs': 100,         # Reduced from 30 (still enough for convergence)
    'beta': 1.0,              # Standard beta-VAE
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def train_epoch(model, train_loader, optimizer, device, beta):
    model.train()
    train_loss = 0
    train_bce = 0
    train_kld = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, data in enumerate(pbar):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        recon_batch, mu, logvar = model(data)
        loss, bce, kld = vae_loss(recon_batch, data, mu, logvar, beta)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track losses
        train_loss += loss.item()
        train_bce += bce.item()
        train_kld += kld.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item() / len(data),
            'bce': bce.item() / len(data),
            'kld': kld.item() / len(data)
        })
    
    # Average losses
    avg_loss = train_loss / len(train_loader.dataset)
    avg_bce = train_bce / len(train_loader.dataset)
    avg_kld = train_kld / len(train_loader.dataset)
    
    return avg_loss, avg_bce, avg_kld

def validate(model, val_loader, device, beta):
    model.eval()
    val_loss = 0
    val_bce = 0
    val_kld = 0
    
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss, bce, kld = vae_loss(recon_batch, data, mu, logvar, beta)
            
            val_loss += loss.item()
            val_bce += bce.item()
            val_kld += kld.item()
    
    avg_loss = val_loss / len(val_loader.dataset)
    avg_bce = val_bce / len(val_loader.dataset)
    avg_kld = val_kld / len(val_loader.dataset)
    
    return avg_loss, avg_bce, avg_kld

def save_reconstructions(model, data_loader, epoch, device, num_images=8):
    model.eval()
    with torch.no_grad():
        data = next(iter(data_loader))[:num_images].to(device)
        recon, _, _ = model(data)
        
        # Plot original vs reconstruction
        fig, axes = plt.subplots(2, num_images, figsize=(15, 4))
        
        for i in range(num_images):
            # Original
            axes[0, i].imshow(data[i].cpu().squeeze(), cmap='gray')
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original', fontsize=10)
            
            # Reconstruction
            axes[1, i].imshow(recon[i].cpu().squeeze(), cmap='gray')
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Reconstructed', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'outputs/reconstruction_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
        plt.close()

def generate_samples(model, epoch, device, num_samples=16):
    model.eval()
    with torch.no_grad():
        # Sample from standard normal
        z = torch.randn(num_samples, model.latent_dim).to(device)
        samples = model.decode(z)
        
        # Plot generated samples
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        axes = axes.flatten()
        
        for i in range(num_samples):
            axes[i].imshow(samples[i].cpu().squeeze(), cmap='gray')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'outputs/generated_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
        plt.close()

def plot_losses(history, save_path='outputs/loss_curves.png'):
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Total Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Loss (BCE + KLD)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # BCE Loss
    axes[1].plot(epochs, history['train_bce'], 'b-', label='Train')
    axes[1].plot(epochs, history['val_bce'], 'r-', label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('BCE Loss')
    axes[1].set_title('Reconstruction Loss (BCE)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # KLD Loss
    axes[2].plot(epochs, history['train_kld'], 'b-', label='Train')
    axes[2].plot(epochs, history['val_kld'], 'r-', label='Val')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('KLD Loss')
    axes[2].set_title('KL Divergence')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved loss curves to {save_path}")

def main():
    print("="*60)
    print("Training VAE on Chest X-Ray Dataset")
    print("="*60)
    
    # Print hyperparameters
    print("\nHyperparameters:")
    for key, value in HYPERPARAMS.items():
        print(f"  {key}: {value}")
    print()
    
    # Create directories
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # Setup device
    device = torch.device(HYPERPARAMS['device'])
    print(f"Using device: {device}\n")
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_root='./data/chest_xray',
        batch_size=HYPERPARAMS['batch_size'],
        image_size=HYPERPARAMS['image_size'],
        num_workers=8
    )
    
    # Initialize model
    model = VAE(
        image_size=HYPERPARAMS['image_size'],
        latent_dim=HYPERPARAMS['latent_dim']
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel has {total_params:,} parameters")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=HYPERPARAMS['learning_rate'])
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_bce': [], 'val_bce': [],
        'train_kld': [], 'val_kld': []
    }
    
    # Training loop
    print("\nStarting training...\n")
    best_val_loss = float('inf')
    
    for epoch in range(1, HYPERPARAMS['num_epochs'] + 1):
        print(f"Epoch {epoch}/{HYPERPARAMS['num_epochs']}")
        print("-" * 60)
        
        # Train
        train_loss, train_bce, train_kld = train_epoch(
            model, train_loader, optimizer, device, HYPERPARAMS['beta']
        )
        
        # Validate
        val_loss, val_bce, val_kld = validate(
            model, val_loader, device, HYPERPARAMS['beta']
        )
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_bce'].append(train_bce)
        history['val_bce'].append(val_bce)
        history['train_kld'].append(train_kld)
        history['val_kld'].append(val_kld)
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train - Loss: {train_loss:.4f}, BCE: {train_bce:.4f}, KLD: {train_kld:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, BCE: {val_bce:.4f}, KLD: {val_kld:.4f}")
        
        # Save reconstructions and generated samples every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            save_reconstructions(model, val_loader, epoch, device)
            generate_samples(model, epoch, device)
            print(f"  Saved reconstructions and generated samples")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'hyperparams': HYPERPARAMS
            }, 'checkpoints/best_model.pth')
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
        
        print()
    
    # Save final model
    torch.save({
        'epoch': HYPERPARAMS['num_epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'hyperparams': HYPERPARAMS,
        'history': history
    }, 'checkpoints/final_model.pth')
    
    # Plot final loss curves
    plot_losses(history)
    
    print("="*60)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("="*60)

if __name__ == '__main__':
    main()