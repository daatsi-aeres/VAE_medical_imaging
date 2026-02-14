import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, image_size=128, latent_dim=256, hidden_dim=256):
        super(VAE, self).__init__()
        
        self.image_size = image_size
        self.latent_dim = latent_dim
        
        # Encoder: Image -> Latent distribution (mu, logvar)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),   # 128->64
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 64->32
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 32->16
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), # 16->8
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, stride=2, padding=1), # 8->4
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate flattened size (for 128x128 input)
        self.flat_size = 512 * 4 * 4  # 8192
        
        # Latent space projection
        self.fc_mu = nn.Linear(self.flat_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_size, latent_dim)
        
        # Decoder: Latent -> Image
        self.decoder_input = nn.Linear(latent_dim, self.flat_size)
        
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (512, 4, 4)),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), # 4->8
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # 8->16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 16->32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 32->64
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),    # 64->128
            nn.Sigmoid()  # Output in [0,1]
        )
        
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        # Reparameterization trick: z = mu + sigma * epsilon
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.decoder_input(z)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss function
def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    # Reconstruction loss (Binary Cross Entropy)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + beta * KLD, BCE, KLD