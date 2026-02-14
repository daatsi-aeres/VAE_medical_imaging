import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm

from vae_model import VAE
from dataset import get_dataloaders

# For Inception Score and FID
from torchvision.models import inception_v3
import torch.nn.functional as F
from scipy import linalg

class InceptionScoreCalculator:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = inception_v3(pretrained=True, transform_input=False).to(device)
        self.model.eval()
        
    def calculate(self, images, batch_size=32, splits=10):
        """
        Calculate Inception Score
        images: tensor of shape (N, C, H, W) in range [0, 1]
        """
        N = len(images)
        
        # Resize images to 299x299 for Inception
        if images.shape[2] != 299 or images.shape[3] != 299:
            resize = transforms.Resize((299, 299))
            images = torch.stack([resize(img) for img in images])
        
        # Convert grayscale to RGB
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        
        # Get predictions
        preds = []
        for i in range(0, N, batch_size):
            batch = images[i:i+batch_size].to(self.device)
            with torch.no_grad():
                pred = F.softmax(self.model(batch), dim=1)
            preds.append(pred.cpu().numpy())
        
        preds = np.concatenate(preds, axis=0)
        
        # Calculate score
        split_scores = []
        for k in range(splits):
            part = preds[k * (N // splits): (k+1) * (N // splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(np.sum(pyx * np.log(pyx / py)))
            split_scores.append(np.exp(np.mean(scores)))
        
        return np.mean(split_scores), np.std(split_scores)

class FIDCalculator:
    def __init__(self, device='cuda'):
        self.device = device
        # Load InceptionV3 for feature extraction
        self.inception = inception_v3(pretrained=True, transform_input=False).to(device)
        self.inception.eval()
        
        # Remove the final classification layer
        self.inception.fc = torch.nn.Identity()
        
    def get_activations(self, images, batch_size=32):
        """Get Inception activations (2048-dim features)"""
        N = len(images)
        
        # Resize and convert to RGB
        if images.shape[2] != 299 or images.shape[3] != 299:
            resize = transforms.Resize((299, 299))
            images = torch.stack([resize(img) for img in images])
        
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        
        # Get activations
        activations = []
        for i in tqdm(range(0, N, batch_size), desc="Extracting features"):
            batch = images[i:i+batch_size].to(self.device)
            with torch.no_grad():
                # Forward pass through inception
                pred = self.inception(batch)
            activations.append(pred.cpu().numpy())
        
        activations = np.concatenate(activations, axis=0)
        
        # Should be shape (N, 2048)
        print(f"Feature shape: {activations.shape}")
        return activations
    
    def calculate_fid(self, real_images, fake_images, batch_size=32):
        """Calculate FID score"""
        # Get activations
        print("Extracting features from real images...")
        act_real = self.get_activations(real_images, batch_size)
        
        print("Extracting features from generated images...")
        act_fake = self.get_activations(fake_images, batch_size)
        
        # Calculate statistics
        mu_real = np.mean(act_real, axis=0)
        sigma_real = np.cov(act_real, rowvar=False)
        
        mu_fake = np.mean(act_fake, axis=0)
        sigma_fake = np.cov(act_fake, rowvar=False)
        
        # Calculate FID
        diff = mu_real - mu_fake
        
        # Add small value to diagonal for numerical stability
        offset = np.eye(sigma_real.shape[0]) * 1e-6
        sigma_real += offset
        sigma_fake += offset
        
        covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2*covmean)
        
        return fid

def generate_samples(model, num_samples, device):
    """Generate samples from VAE"""
    model.eval()
    samples = []
    
    batch_size = 64
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc="Generating samples"):
            batch_size_actual = min(batch_size, num_samples - i)
            z = torch.randn(batch_size_actual, model.latent_dim).to(device)
            generated = model.decode(z)
            samples.append(generated.cpu())
    
    return torch.cat(samples, dim=0)

def get_real_images(data_loader, num_images):
    """Get real images from dataset"""
    images = []
    for batch in data_loader:
        images.append(batch)
        if len(torch.cat(images, dim=0)) >= num_images:
            break
    
    images = torch.cat(images, dim=0)[:num_images]
    return images

def main():
    print("="*60)
    print("Evaluating VAE with Inception Score and FID")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load trained model
    print("Loading trained model...")
    checkpoint = torch.load('checkpoints/best_model.pth')
    
    model = VAE(
        image_size=checkpoint['hyperparams']['image_size'],
        latent_dim=checkpoint['hyperparams']['latent_dim']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully!\n")
    
    # Load test data
    print("Loading test data...")
    _, _, test_loader = get_dataloaders(
        data_root='./data/chest_xray',
        batch_size=32,
        image_size=checkpoint['hyperparams']['image_size'],
        num_workers=4
    )
    
    # Generate samples
    num_samples = 1000  # Use 1000 samples for evaluation
    print(f"\nGenerating {num_samples} samples from VAE...")
    fake_images = generate_samples(model, num_samples, device)
    
    # Get real images
    print(f"\nCollecting {num_samples} real images...")
    real_images = get_real_images(test_loader, num_samples)
    
    # Calculate Inception Score
    print("\n" + "="*60)
    print("Calculating Inception Score...")
    print("="*60)
    is_calculator = InceptionScoreCalculator(device)
    is_mean, is_std = is_calculator.calculate(fake_images)
    print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")
    
    # Calculate FID
    print("\n" + "="*60)
    print("Calculating FID Score...")
    print("="*60)
    fid_calculator = FIDCalculator(device)
    fid_score = fid_calculator.calculate_fid(real_images, fake_images)
    print(f"FID Score: {fid_score:.4f}")
    
    # Save results
    results = {
        'inception_score_mean': is_mean,
        'inception_score_std': is_std,
        'fid_score': fid_score
    }
    
    # Save to file
    with open('outputs/evaluation_results.txt', 'w') as f:
        f.write("VAE Evaluation Results\n")
        f.write("="*60 + "\n")
        f.write(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}\n")
        f.write(f"FID Score: {fid_score:.4f}\n")
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print(f"Results saved to outputs/evaluation_results.txt")
    print("="*60)

if __name__ == '__main__':
    main()