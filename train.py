import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import csv
from tqdm import tqdm
from vae_model import VAE, vae_loss
from dataset import get_dataloaders

# ---------------- CONFIGURATION ---------------- #
HYPERPARAMS = {
    'image_size': 128,
    'latent_dim': 256,
    'batch_size': 64,         
    'learning_rate': 1e-4,
    'num_epochs': 500,         
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'anneal_epochs': 100,      
    'max_beta': 0.5          
}
# ----------------------------------------------- #

class CSVLogger:
    def __init__(self, filename, fieldnames):
        self.filename = filename
        self.fieldnames = fieldnames
        
        # Create file and write header
        with open(self.filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def log(self, data):
        with open(self.filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(data)

def get_current_beta(epoch, config):
    if epoch < config['anneal_epochs']:
        return (epoch / config['anneal_epochs']) * config['max_beta']
    return config['max_beta']

def train_epoch(model, train_loader, optimizer, device, beta):
    model.train()
    train_loss = 0
    train_bce = 0
    train_kld = 0
    
    pbar = tqdm(train_loader, desc=f'Train (Beta={beta:.5f})')
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss, bce, kld = vae_loss(recon_batch, data, mu, logvar, beta)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_bce += bce.item()
        train_kld += kld.item()
        pbar.set_postfix({'loss': loss.item()/len(data)})
    
    n = len(train_loader.dataset)
    return train_loss/n, train_bce/n, train_kld/n

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
    n = len(val_loader.dataset)
    return val_loss/n, val_bce/n, val_kld/n

def generate_samples(model, epoch, device, num_samples=16):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim).to(device)
        samples = model.decode(z)
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        axes = axes.flatten()
        for i in range(num_samples):
            axes[i].imshow(samples[i].cpu().squeeze(), cmap='gray')
            axes[i].axis('off')
        plt.tight_layout()
        plt.savefig(f'outputs/gen_epoch_{epoch}.png')
        plt.close()

def main():
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    device = torch.device(HYPERPARAMS['device'])
    
    # Initialize Logger
    logger = CSVLogger('outputs/training_log.csv', ['epoch', 'total_loss', 'bce', 'kld', 'val_loss', 'val_bce', 'val_kld', 'beta'])
    
    print("Loading Data...")
    train_loader, _, test_loader = get_dataloaders(
        data_root='./data/chest_xray',
        batch_size=HYPERPARAMS['batch_size'],
        image_size=HYPERPARAMS['image_size']
    )
    
    model = VAE(image_size=HYPERPARAMS['image_size'], latent_dim=HYPERPARAMS['latent_dim']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=HYPERPARAMS['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    print(f"Starting training on {device}...")
    
    for epoch in range(1, HYPERPARAMS['num_epochs'] + 1):
        current_beta = get_current_beta(epoch, HYPERPARAMS)
        
        # Train & Validate
        loss, bce, kld = train_epoch(model, train_loader, optimizer, device, current_beta)
        val_loss, val_bce, val_kld = validate(model, test_loader, device, current_beta)
        
        scheduler.step(val_loss)
        
        # LOG TO CSV (The Magic Line)
        logger.log({
            'epoch': epoch,
            'total_loss': loss,
            'bce': bce,
            'kld': kld,
            'val_loss': val_loss,
            'val_bce': val_bce,
            'val_kld': val_kld,
            'beta': current_beta
        })
        
        print(f"Epoch {epoch} | Loss: {loss:.1f} | BCE: {bce:.1f} | KLD: {kld:.1f} | Val Loss: {val_loss:.1f}")
        
        if epoch % 5 == 0 or epoch == 1:
            generate_samples(model, epoch, device)
            torch.save({'model': model.state_dict(), 'config': HYPERPARAMS}, 'checkpoints/best_model.pth')

if __name__ == '__main__':
    main()