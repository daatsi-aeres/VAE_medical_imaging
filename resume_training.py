import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
from vae_model import VAE, vae_loss
from dataset import get_dataloaders

# ================= CONFIGURATION =================
RESUME_CONFIG = {
    'checkpoint_path': 'checkpoints/best_model.pth', # Path to the file you want to resume from
    'start_epoch': 51,                               # The epoch to START at (e.g., if you finished 50, start at 51)
    'total_epochs': 200,                             # The NEW final epoch goal
    
    # Keep these same as your original training to match architecture
    'image_size': 128,
    'latent_dim': 256,
    'batch_size': 64,         
    'learning_rate': 1e-3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Annealing (Beta) Config
    'anneal_epochs': 20,      
    'max_beta': 0.01          
}
# =================================================

class CSVLogger:
    def __init__(self, filename, fieldnames, resume=False):
        self.filename = filename
        self.fieldnames = fieldnames
        
        # If resuming, we append ('a'). If new, we write ('w')
        mode = 'a' if resume else 'w'
        
        # Only write header if we are NOT resuming or if file is empty
        write_header = not resume or not os.path.exists(filename)
        
        self.file = open(self.filename, mode, newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        
        if write_header:
            self.writer.writeheader()
            self.file.flush()

    def log(self, data):
        self.writer.writerow(data)
        self.file.flush()
        
    def close(self):
        self.file.close()

def get_current_beta(epoch, config):
    # Calculates Beta based on the current global epoch
    if epoch < config['anneal_epochs']:
        return (epoch / config['anneal_epochs']) * config['max_beta']
    return config['max_beta']

def train_epoch(model, train_loader, optimizer, device, beta):
    model.train()
    train_loss = 0
    train_bce = 0
    train_kld = 0
    
    pbar = tqdm(train_loader, desc=f'Train (Beta={beta:.5f})')
    for batch_idx, data in enumerate(pbar):
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
    device = torch.device(RESUME_CONFIG['device'])
    
    # 1. Initialize Model & Optimizer
    print(f"Initializing model on {device}...")
    model = VAE(image_size=RESUME_CONFIG['image_size'], latent_dim=RESUME_CONFIG['latent_dim']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=RESUME_CONFIG['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # 2. Load Checkpoint
    if os.path.exists(RESUME_CONFIG['checkpoint_path']):
        print(f"Loading checkpoint from {RESUME_CONFIG['checkpoint_path']}...")
        checkpoint = torch.load(RESUME_CONFIG['checkpoint_path'])
        
        # Load weights
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        elif 'model_state_dict' in checkpoint: # Handle different saving conventions
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint) # Try loading direct dictionary
            
        print("Model weights loaded successfully.")
        
        # Optional: Load optimizer if it was saved (The previous script didn't save it, so this handles that case)
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("Optimizer state loaded.")
        else:
            print("Warning: Optimizer state not found in checkpoint. Re-initializing optimizer.")
            
    else:
        print(f"Error: Checkpoint {RESUME_CONFIG['checkpoint_path']} not found!")
        return

    # 3. Setup Logger (Append mode)
    logger = CSVLogger('outputs/training_log.csv', 
                      ['epoch', 'total_loss', 'bce', 'kld', 'val_loss', 'val_bce', 'val_kld', 'beta'],
                      resume=True)
    
    print("Loading Data...")
    train_loader, _, test_loader = get_dataloaders(
        data_root='./data/chest_xray',
        batch_size=RESUME_CONFIG['batch_size'],
        image_size=RESUME_CONFIG['image_size']
    )

    print(f"Resuming training from Epoch {RESUME_CONFIG['start_epoch']} to {RESUME_CONFIG['total_epochs']}...")
    
    # 4. Resume Loop
    for epoch in range(RESUME_CONFIG['start_epoch'], RESUME_CONFIG['total_epochs'] + 1):
        # Calculate Beta based on GLOBAL epoch (so it stays at 0.01 if epoch > 20)
        current_beta = get_current_beta(epoch, RESUME_CONFIG)
        
        # Train
        loss, bce, kld = train_epoch(model, train_loader, optimizer, device, current_beta)
        val_loss, val_bce, val_kld = validate(model, test_loader, device, current_beta)
        
        scheduler.step(val_loss)
        
        # Log
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
        
        # Save
        if epoch % 5 == 0 or epoch == RESUME_CONFIG['total_epochs']:
            generate_samples(model, epoch, device)
            # We save optimizer state now for future resuming
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': RESUME_CONFIG,
                'epoch': epoch
            }, 'checkpoints/best_model.pth')

    logger.close()
    print("Resumed Training Complete.")

if __name__ == '__main__':
    main()