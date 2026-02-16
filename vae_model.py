import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.leaky_relu(out, 0.2)
        return out

class VAE(nn.Module):
    def __init__(self, image_size=128, latent_dim=256):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # ================= ENCODER =================
        # Input: 1 x 128 x 128
        self.enc_conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1) # -> 64 x 64 x 64
        self.enc_bn1 = nn.BatchNorm2d(64)
        
        self.enc_res1 = ResidualBlock(64, 128, stride=2) # -> 128 x 32 x 32
        self.enc_res2 = ResidualBlock(128, 256, stride=2) # -> 256 x 16 x 16
        self.enc_res3 = ResidualBlock(256, 512, stride=2) # -> 512 x 8 x 8
        self.enc_res4 = ResidualBlock(512, 1024, stride=2) # -> 1024 x 4 x 4
        
        self.flat_size = 1024 * 4 * 4
        
        # Latent Space
        self.fc_mu = nn.Linear(self.flat_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_size, latent_dim)
        
        # ================= DECODER =================
        self.decoder_input = nn.Linear(latent_dim, self.flat_size)
        
        self.dec_res1 = ResidualBlock(1024, 512) # 4x4 -> 4x4 (channels change)
        self.dec_res2 = ResidualBlock(512, 256)
        self.dec_res3 = ResidualBlock(256, 128)
        self.dec_res4 = ResidualBlock(128, 64)
        
        # Final reconstruction layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = F.leaky_relu(self.enc_bn1(self.enc_conv1(x)), 0.2)
        x = self.enc_res1(x)
        x = self.enc_res2(x)
        x = self.enc_res3(x)
        x = self.enc_res4(x)
        x = x.flatten(start_dim=1)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 1024, 4, 4)
        
        # Upsampling + Residual Block
        # We use nearest neighbor interpolation + convolution to avoid checkerboard artifacts
        x = F.interpolate(x, scale_factor=2, mode='nearest') # 4->8
        x = self.dec_res1(x)
        
        x = F.interpolate(x, scale_factor=2, mode='nearest') # 8->16
        x = self.dec_res2(x)
        
        x = F.interpolate(x, scale_factor=2, mode='nearest') # 16->32
        x = self.dec_res3(x)
        
        x = F.interpolate(x, scale_factor=2, mode='nearest') # 32->64
        x = self.dec_res4(x)
        
        x = F.interpolate(x, scale_factor=2, mode='nearest') # 64->128
        x = self.final_conv(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# # This was missing in the previous step:
# def vae_loss(recon_x, x, mu, logvar, kld_weight=1.0):
#     # BCE = Reconstruction Loss (How blurry is it?)
#     # reduction='sum' sums over all pixels in the batch
#     BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
#     # KLD = Regularization Loss (How organized is the latent space?)
#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
#     return BCE + (kld_weight * KLD), BCE, KLD

def vae_loss(recon_x, x, mu, logvar, kld_weight=1.0):
    # MSE = Reconstruction Loss (Squared Error)
    # reduction='sum' sums the error across all pixels and batch items
    MSE = F.mse_loss(recon_x, x, reduction='sum')

    # KLD = Regularization Loss (KL Divergence)
    # Measures how much the learned distribution diverges from a standard normal distribution
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + (kld_weight * KLD), MSE, KLD