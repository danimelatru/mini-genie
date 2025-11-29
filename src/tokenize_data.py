import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

# --- CONFIG (Must match Weighted Loss VQ-VAE) ---
EMBEDDING_DIM = 64      
NUM_EMBEDDINGS = 512 # Match the 'Nuclear' version
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- PATH SETUP ---
current_path = Path(__file__).parent.resolve()
root_path = current_path.parent
sys.path.append(str(root_path))
DATA_PATH = root_path / "data" / "episodes"
ARTIFACTS_PATH = root_path / "data" / "artifacts"
OUT_PATH = root_path / "data" / "tokens"
OUT_PATH.mkdir(parents=True, exist_ok=True)

# --- ARCHITECTURE (Exact copy of Weighted Loss VQ-VAE) ---
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
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)

    def forward_indices(self, inputs):
        # Helper to just get indices
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        flat_input = inputs.view(-1, self._embedding_dim)
        
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        return encoding_indices

class VQVAE(nn.Module):
    def __init__(self):
        super(VQVAE, self).__init__()
        # Encoder matching Weighted Loss Version (128 filters)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1), # The 128 layer
            Residual(128, 128, 64),
            Residual(128, 128, 64),
            nn.Conv2d(128, EMBEDDING_DIM, 3, 1, 1) 
        )
        self.vq_layer = VectorQuantizerEMA(NUM_EMBEDDINGS, EMBEDDING_DIM)

    def encode(self, x):
        z = self.encoder(x)
        indices = self.vq_layer.forward_indices(z) 
        return indices

# --- MAIN TOKENIZATION LOOP ---
def main():
    print(f"Running on device: {DEVICE}")
    
    print("Loading VQ-VAE weights...")
    model = VQVAE().to(DEVICE)
    try:
        # Load weights
        state_dict = torch.load(ARTIFACTS_PATH / "vqvae.pth", map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR loading weights: {e}")
        return
        
    model.eval()
    
    files = sorted(list(Path(DATA_PATH).glob("*.npz")))
    print(f"Found {len(files)} episodes to tokenize.")
    
    with torch.no_grad():
        for i, f in enumerate(files):
            with np.load(f) as data:
                frames = data["frames"]
                actions = data["actions"] 
            
            # Batch processing
            frame_tensor = torch.from_numpy(frames).float() / 255.0
            frame_tensor = frame_tensor.permute(0, 3, 1, 2).to(DEVICE)
            
            # Encode -> Get Indices
            indices = model.encode(frame_tensor)
            
            # Reshape [T*16*16] -> [T, 16, 16]
            indices = indices.view(frames.shape[0], 16, 16).cpu().numpy().astype(np.uint16)
            
            # Save
            save_path = OUT_PATH / f.name
            np.savez_compressed(save_path, tokens=indices, actions=actions)
            
            if i % 10 == 0:
                print(f"Tokenized episode {i}/{len(files)}")
                
    print(f"Done! Tokens saved to {OUT_PATH}")

if __name__ == "__main__":
    main()