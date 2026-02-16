import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import os
import pandas as pd
from scipy import linalg
import torch.nn.functional as F
from torchvision.models import inception_v3

# Import your model and data loader
from vae_model import VAE
from dataset import get_dataloaders

# ================= CONFIGURATION =================
CONFIG = {
    'batch_size': 32,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'checkpoint_path': 'checkpoints/best_model.pth',
    'output_dir': 'evaluation_results_final',
    'log_file': 'outputs/training_log.csv' # Path to your training log
}
# =================================================

def load_model_and_data():
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    print("Loading Data...")
    # Get test loader (returns images AND labels)
    _, _, test_loader = get_dataloaders(
        data_root='./data/chest_xray',
        batch_size=CONFIG['batch_size'],
        image_size=128
    )
    
    print("Loading Model...")
    checkpoint = torch.load(CONFIG['checkpoint_path'])
    model = VAE(image_size=128, latent_dim=256).to(CONFIG['device'])
    
    # Handle different checkpoint saving formats
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    return model, test_loader

# ------------------------------------------------------------------
# 1. Loss Graphs (The Foundation)
# ------------------------------------------------------------------
def plot_loss_curves(log_file):
    print("\n[1/6] Generating Loss Curves...")
    if not os.path.exists(log_file):
        print(f"Warning: {log_file} not found. Skipping loss plots.")
        return

    df = pd.read_csv(log_file)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"] 

    # Plot 1: Log Scale Loss
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['total_loss'], label='Train Loss', color=colors[0], linewidth=2)
    plt.plot(df['epoch'], df['val_loss'], label='Val Loss', color=colors[1], linestyle='--', linewidth=2)
    plt.yscale('log')
    plt.title('Training Loss Dynamics (Log Scale)', fontsize=16, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Log Scale)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{CONFIG['output_dir']}/1_loss_log_scale.png", dpi=300)
    plt.close()

    # Plot 2: Dual Axis (Recon vs KL)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = colors[0]
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Reconstruction Loss', color=color, fontweight='bold')
    ax1.plot(df['epoch'], df['bce'], color=color, linewidth=2, label='Recon Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx() 
    color = colors[2]
    ax2.set_ylabel('KL Divergence', color=color, fontweight='bold')
    ax2.plot(df['epoch'], df['kld'], color=color, linewidth=2, linestyle='--', label='KL Loss')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Loss Components Trade-off', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{CONFIG['output_dir']}/2_loss_components.png", dpi=300)
    plt.close()
    print("Loss curves saved.")

# ------------------------------------------------------------------
# 2. FID & IS (The Standard Metrics)
# ------------------------------------------------------------------
class InceptionEvaluator:
    def __init__(self, device='cuda'):
        self.device = device
        self.inception = inception_v3(pretrained=True, transform_input=False).to(device)
        self.inception.eval()
        self.inception.fc = torch.nn.Identity()

    def preprocess(self, images):
        # Resize to 299x299 and duplicate channels to RGB
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        return images

    def get_features(self, images, batch_size=32):
        features = []
        n_batches = (len(images) + batch_size - 1) // batch_size
        with torch.no_grad():
            for i in range(n_batches):
                batch = images[i*batch_size : (i+1)*batch_size].to(self.device)
                batch = self.preprocess(batch)
                pred = self.inception(batch)
                features.append(pred.cpu().numpy())
        return np.concatenate(features, axis=0)

    def calculate_fid(self, real_images, fake_images):
        print("Extracting features for FID...")
        act1 = self.get_features(real_images)
        act2 = self.get_features(fake_images)
        mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if np.iscomplexobj(covmean): covmean = covmean.real
        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return fid

    def calculate_inception_score(self, images, splits=10):
        print("Calculating Inception Score...")
        model = inception_v3(pretrained=True, transform_input=False).to(self.device)
        model.eval()
        preds = []
        n_batches = (len(images) + 32 - 1) // 32
        with torch.no_grad():
            for i in range(n_batches):
                batch = images[i*32 : (i+1)*32].to(self.device)
                batch = self.preprocess(batch)
                pred = F.softmax(model(batch), dim=1)
                preds.append(pred.cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        scores = []
        for i in range(splits):
            part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
        return np.mean(scores), np.std(scores)

def calculate_standard_metrics(model, dataloader):
    print("\n[2/7] Calculating FID and Inception Score...")
    
    # Collect real images (limit to 1000 for speed)
    real_imgs = []
    for imgs, _ in dataloader:
        real_imgs.append(imgs)
        if len(real_imgs) * CONFIG['batch_size'] >= 1000: break
    real_imgs = torch.cat(real_imgs)[:1000]
    
    # Generate fake images
    model.eval()
    with torch.no_grad():
        z = torch.randn(1000, 256).to(CONFIG['device'])
        fake_imgs = model.decode(z).cpu()
    
    evaluator = InceptionEvaluator(CONFIG['device'])
    fid = evaluator.calculate_fid(real_imgs, fake_imgs)
    is_mean, is_std = evaluator.calculate_inception_score(fake_imgs)
    
    print(f"FID Score: {fid:.4f}")
    print(f"Inception Score: {is_mean:.4f} +/- {is_std:.4f}")
    
    with open(f"{CONFIG['output_dir']}/metrics_standard.txt", "w") as f:
        f.write(f"FID: {fid:.4f}\n")
        f.write(f"IS: {is_mean:.4f} +/- {is_std:.4f}\n")

# ------------------------------------------------------------------
# 3. Latent Space Visualization (t-SNE)
# ------------------------------------------------------------------
def plot_tsne(model, dataloader):
    print("\n[3/7] Generating t-SNE Visualization...")
    latent_vectors = []
    labels = []
    
    # Limit to 500 samples to keep t-SNE fast and clear
    count = 0
    with torch.no_grad():
        for imgs, lbls in dataloader:
            if count >= 500: break
            imgs = imgs.to(CONFIG['device'])
            mu, _ = model.encode(imgs)
            latent_vectors.append(mu.cpu().numpy())
            labels.extend(lbls.numpy())
            count += len(imgs)
            
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    labels = np.array(labels)
    
    pca = PCA(n_components=50)
    latent_pca = pca.fit_transform(latent_vectors)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    latent_2d = tsne.fit_transform(latent_pca)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='coolwarm', alpha=0.6, s=20)
    plt.legend(handles=scatter.legend_elements()[0], labels=['Normal', 'Pneumonia'], title="Condition")
    plt.title('Latent Space Projection (t-SNE)', fontsize=14)
    plt.savefig(f"{CONFIG['output_dir']}/3_tsne_latent_space.png", dpi=300)
    plt.close()

# ------------------------------------------------------------------
# 4. Anomaly Detection (Histograms)
# ------------------------------------------------------------------
def plot_anomaly_histograms(model, dataloader):
    print("\n[4/7] Generating Anomaly Histograms...")
    normal_errors = []
    pneumonia_errors = []
    loss_fn = nn.MSELoss(reduction='none') 
    
    # Limit samples
    count = 0
    with torch.no_grad():
        for imgs, lbls in dataloader:
            if count >= 500: break
            imgs = imgs.to(CONFIG['device'])
            recon, _, _ = model(imgs)
            error = loss_fn(recon, imgs).view(imgs.size(0), -1).mean(dim=1).cpu().numpy()
            
            for i, label in enumerate(lbls):
                if label == 0: normal_errors.append(error[i])
                else: pneumonia_errors.append(error[i])
            count += len(imgs)
                    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(normal_errors, fill=True, color="blue", label="Normal", alpha=0.3)
    sns.kdeplot(pneumonia_errors, fill=True, color="red", label="Pneumonia", alpha=0.3)
    plt.title('Reconstruction Error Distribution', fontsize=14)
    plt.legend()
    plt.savefig(f"{CONFIG['output_dir']}/4_anomaly_histogram.png", dpi=300)
    plt.close()

# ------------------------------------------------------------------
# 5. Interpolation
# ------------------------------------------------------------------
def plot_interpolation(model, dataloader):
    print("\n[5/7] Generating Interpolation...")
    normal_img = None
    pneumonia_img = None
    
    for imgs, lbls in dataloader:
        if normal_img is not None and pneumonia_img is not None: break
        for i in range(len(imgs)):
            if lbls[i] == 0 and normal_img is None:
                normal_img = imgs[i].unsqueeze(0).to(CONFIG['device'])
            if lbls[i] == 1 and pneumonia_img is None:
                pneumonia_img = imgs[i].unsqueeze(0).to(CONFIG['device'])
    
    model.eval()
    with torch.no_grad():
        z_normal, _ = model.encode(normal_img)
        z_pneumonia, _ = model.encode(pneumonia_img)
        alphas = np.linspace(0, 1, 10)
        z_steps = torch.cat([(1 - a) * z_normal + a * z_pneumonia for a in alphas], dim=0)
        generated = model.decode(z_steps)
        
    fig, axes = plt.subplots(1, 10, figsize=(20, 3))
    for i in range(10):
        img = generated[i].cpu().squeeze().numpy()
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
    plt.suptitle("Latent Space Interpolation (Normal -> Pneumonia)", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{CONFIG['output_dir']}/5_interpolation.png", dpi=300)
    plt.close()

# ------------------------------------------------------------------
# 6. SSIM
# ------------------------------------------------------------------
def calculate_ssim(model, dataloader):
    print("\n[6/7] Calculating SSIM...")
    ssim_scores = []
    count = 0
    with torch.no_grad():
        for imgs, _ in dataloader:
            if count >= 200: break # Limit samples
            imgs = imgs.to(CONFIG['device'])
            recon, _, _ = model(imgs)
            
            imgs_np = imgs.cpu().numpy().squeeze()
            recon_np = recon.cpu().numpy().squeeze()
            
            if len(imgs_np.shape) == 2: # Batch size 1 fix
                imgs_np = imgs_np[None, ...]
                recon_np = recon_np[None, ...]
                
            for i in range(imgs_np.shape[0]):
                score = ssim(imgs_np[i], recon_np[i], data_range=1.0)
                ssim_scores.append(score)
            count += len(imgs)
                
    avg_ssim = np.mean(ssim_scores)
    print(f"Average SSIM: {avg_ssim:.4f}")
    with open(f"{CONFIG['output_dir']}/metrics_advanced.txt", "w") as f:
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")

# =================================================

def plot_reconstruction_vs_generation(model, dataloader):
    print("\n[7/7] Generating Reconstruction vs Generation Grids...")
    
    # Create an iterator to fetch different batches
    data_iter = iter(dataloader)
    
    # Generate 4 different plots
    for plot_idx in range(4):
        try:
            real_batch, _ = next(data_iter)
        except StopIteration:
            break # Stop if we run out of data
            
        real_batch = real_batch[:4].to(CONFIG['device']) # Take 4 images
        
        # 2. Reconstruct
        model.eval()
        with torch.no_grad():
            recon_batch, _, _ = model(real_batch)
        
        # 3. Generate Random Dreams
        with torch.no_grad():
            z_random = torch.randn(4, 256).to(CONFIG['device'])
            gen_batch = model.decode(z_random)
            
        # 4. Plotting
        # Tight layout with no gaps
        fig, axes = plt.subplots(3, 4, figsize=(10, 7.5), 
                               gridspec_kw={'wspace': 0.05, 'hspace': 0.05})
        
        rows = [real_batch, recon_batch, gen_batch]
        row_names = ["Real Input", "Reconstructed", "Random\nGenerated"]
        
        for r in range(3):
            for c in range(4):
                ax = axes[r, c]
                img = rows[r][c].cpu().squeeze().numpy()
                ax.imshow(img, cmap='gray')
                ax.axis('off') # Hide axes completely
                
                # Add row labels only to the first column (Left aligned)
                if c == 0:
                    ax.text(-0.1, 0.5, row_names[r], transform=ax.transAxes, 
                            fontsize=14, va='center', ha='right', fontweight='bold', rotation=90)

        # Save with tight bounding box to remove white borders
        save_path = f"{CONFIG['output_dir']}/6_recon_vs_gen_{plot_idx}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {save_path}")


def main():
    model, test_loader = load_model_and_data()
    
    # 1. Standard Training Plots
    plot_loss_curves(CONFIG['log_file'])
    
    # 2. Standard GAN/VAE Metrics
    calculate_standard_metrics(model, test_loader)
    
    # 3. Advanced Analysis
    plot_tsne(model, test_loader)
    plot_anomaly_histograms(model, test_loader)
    plot_interpolation(model, test_loader)
    calculate_ssim(model, test_loader)
    
    plot_reconstruction_vs_generation(model, test_loader)

    print(f"\nEvaluation Complete! Results saved to {CONFIG['output_dir']}")

if __name__ == "__main__":
    main()