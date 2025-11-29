import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path

# --- CONFIGURATION ---
BATCH_SIZE = 128
LEARNING_RATE = 2e-4    # Slightly slower to be careful
EMBEDDING_DIM = 64      
NUM_EMBEDDINGS = 512    # Doubled vocabulary size
EPOCHS = 20             # Give it time to learn the hard parts
DECAY = 0.99            
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- PATH SETUP ---
current_path = Path(__file__).parent.resolve()
root_path = current_path.parent
sys.path.append(str(root_path))
DATA_PATH = root_path / "data" / "episodes"
ARTIFACTS_PATH = root_path / "data" / "artifacts"
ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)

# --- 1. DATASET ---
class CommonsDataset(Dataset):
    def __init__(self, data_dir):
        self.files = sorted(list(Path(data_dir).glob("*.npz")))
        print(f"Loading {len(self.files)} episodes into memory...")
        self.data = []
        for f in self.files:
            with np.load(f) as data:
                self.data.append(data["frames"])
        self.data = np.concatenate(self.data, axis=0)
        print(f"Dataset Loaded: {self.data.shape} frames (uint8).")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frame = self.data[idx]
        tensor = torch.from_numpy(frame).float() / 255.0
        return tensor.permute(2, 0, 1) 

# --- 2. ROBUST LAYERS ---
class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=num_residual_hiddens, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_residual_hiddens, out_channels=num_hiddens, kernel_size=1, stride=1, bias=False)
        )
    
    def forward(self, x):
        return x + self._block(x)

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_embedding', torch.empty(num_embeddings, embedding_dim))
        self._embedding.data.normal_()
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)
        
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.t()))
            
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self._embedding).view(input_shape)
        
        if self.training:
            _ema_cluster_size = self._ema_cluster_size * self._decay + \
                                (1 - self._decay) * torch.sum(encodings, 0)
            
            n = torch.sum(_ema_cluster_size.data)
            _ema_cluster_size = (
                (_ema_cluster_size + self._epsilon) /
                (n + self._num_embeddings * self._epsilon) * n)
            
            self._ema_cluster_size = _ema_cluster_size
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding = self._ema_w / self._ema_cluster_size.unsqueeze(1)
        
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), encoding_indices

# --- 3. HIGH-RES VQ-VAE ---
class VQVAE(nn.Module):
    def __init__(self):
        super(VQVAE, self).__init__()
        
        # Encoder: 64 -> 32 -> 16
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1), # Increased filters
            Residual(128, 128, 64),
            Residual(128, 128, 64),
            nn.Conv2d(128, EMBEDDING_DIM, 3, 1, 1) 
        )
        
        self.vq_layer = VectorQuantizerEMA(NUM_EMBEDDINGS, EMBEDDING_DIM, decay=DECAY)
        
        # Decoder: 16 -> 32 -> 64
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(EMBEDDING_DIM, 128, 3, 1, 1), nn.ReLU(),
            Residual(128, 128, 64),
            Residual(128, 128, 64),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        z = self.encoder(x)
        loss, quantized, _ = self.vq_layer(z)
        x_recon = self.decoder(quantized)
        return loss, x_recon, _

# --- 4. TRAINING LOOP WITH WEIGHTED LOSS ---
def main():
    print(f"Running on device: {DEVICE} (Weighted Loss Mode)")
    
    dataset = CommonsDataset(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    model = VQVAE().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Starting VQ-VAE Training...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_recon = 0
        
        for batch_idx, data in enumerate(dataloader):
            data = data.to(DEVICE)
            
            optimizer.zero_grad()
            vq_loss, data_recon, _ = model(data)
            
            # --- THE FIX: WEIGHTED LOSS ---
            # Calculate raw error per pixel
            raw_loss = torch.abs(data_recon - data) # L1 Error
            
            # Create a weight map:
            # If the pixel in Ground Truth is NOT black (> 0.05), give it 20x importance.
            # This covers the Green Apples, Blue Agents, and Gray Background.
            # The vast Black Void gets 1x importance.
            weights = torch.ones_like(raw_loss)
            weights[data > 0.05] = 20.0 
            
            recon_loss = (raw_loss * weights).mean()
            
            loss = recon_loss + vq_loss
            
            loss.backward()
            optimizer.step()
            
            total_recon += recon_loss.item()
            
            if batch_idx % 100 == 0:
                 print(f"  Batch {batch_idx} Weighted Loss: {recon_loss.item():.4f}")

        avg_recon = total_recon / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Avg Weighted Loss: {avg_recon:.4f}")

    print("Saving model...")
    torch.save(model.state_dict(), ARTIFACTS_PATH / "vqvae.pth")
    
    # Visualization
    model.eval()
    with torch.no_grad():
        sample = next(iter(dataloader))[:8].to(DEVICE)
        _, reconstruction, _ = model(sample)
        
        sample = sample.cpu().permute(0, 2, 3, 1).numpy()
        reconstruction = reconstruction.cpu().permute(0, 2, 3, 1).numpy()
        
        fig, axes = plt.subplots(2, 8, figsize=(16, 4))
        for i in range(8):
            axes[0, i].imshow(sample[i])
            axes[0, i].axis('off')
            axes[0, i].set_title("Original")
            
            axes[1, i].imshow(reconstruction[i])
            axes[1, i].axis('off')
            axes[1, i].set_title("Recon")
            
        plt.tight_layout()
        plt.savefig(ARTIFACTS_PATH / "reconstruction_grid.png")
        print(f"Saved visualization to {ARTIFACTS_PATH / 'reconstruction_grid.png'}")

if __name__ == "__main__":
    main()