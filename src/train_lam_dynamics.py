import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path

# --- CONFIGURATION ---
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 10             # 10 Epochs is enough for the RAM-limited dataset
NUM_LATENT_ACTIONS = 8  # Discover 8 distinct behaviors
TOKEN_VOCAB = 512       # Must match the 'Nuclear' VQ-VAE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- PATH SETUP ---
current_path = Path(__file__).parent.resolve()
root_path = current_path.parent
sys.path.append(str(root_path))
DATA_PATH = root_path / "data" / "tokens"
ARTIFACTS_PATH = root_path / "data" / "artifacts"

# --- 1. DATASET (RAM SAFE VERSION) ---
class TokenTransitionsDataset(Dataset):
    def __init__(self, data_dir, limit=2000):
        self.files = sorted(list(Path(data_dir).glob("*.npz")))
        
        # Limit to 2000 episodes to prevent OOM on the login/gpu node
        if len(self.files) > limit:
            print(f"Dataset too large ({len(self.files)}). Limiting to {limit} episodes to save RAM.")
            self.files = self.files[:limit]
        
        print(f"Loading {len(self.files)} token files...")
        
        self.transitions = []
        self.real_actions = [] 
        
        for f in self.files:
            with np.load(f) as data:
                tokens = data["tokens"]   # (T, 16, 16)
                actions = data["actions"] # (T, 2)
                
                # Flatten tokens for the MLP
                flat_tokens = tokens.reshape(tokens.shape[0], -1)
                
                current_steps = flat_tokens[:-1]
                next_steps = flat_tokens[1:]
                
                # We track Agent 0's real action just for the Confusion Matrix
                agent_0_actions = actions[:-1, 0] 
                
                self.transitions.append(np.stack([current_steps, next_steps], axis=1))
                self.real_actions.append(agent_0_actions)
                
        self.transitions = np.concatenate(self.transitions, axis=0)
        self.real_actions = np.concatenate(self.real_actions, axis=0)
        print(f"Loaded {len(self.transitions)} transitions. RAM Usage is safe.")

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        trans = torch.from_numpy(self.transitions[idx]).long()
        real_act = torch.tensor(self.real_actions[idx]).long()
        return trans[0], trans[1], real_act

# --- 2. LATENT ACTION MODEL ---
class LatentActionModel(nn.Module):
    def __init__(self):
        super(LatentActionModel, self).__init__()
        
        self.embedding = nn.Embedding(TOKEN_VOCAB, 32)
        
        # --- ACTION INFERENCE (The Detective) ---
        self.encoder = nn.Sequential(
            nn.Linear(256 * 32 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_LATENT_ACTIONS) # Logits for actions
        )
        
        # --- DYNAMICS PREDICTION (The Simulator) ---
        self.decoder_mlp = nn.Sequential(
            nn.Linear(256 * 32 + NUM_LATENT_ACTIONS, 512),
            nn.ReLU(),
            nn.Linear(512, 256 * TOKEN_VOCAB) # Predict next tokens
        )

    def forward(self, z_t, z_next):
        # Embed
        emb_t = self.embedding(z_t).view(z_t.size(0), -1)
        emb_next = self.embedding(z_next).view(z_next.size(0), -1)
        
        # Infer Action
        combined = torch.cat([emb_t, emb_next], dim=1)
        action_logits = self.encoder(combined) 
        
        # Gumbel-Softmax for differentiable discrete sampling
        action_soft = F.gumbel_softmax(action_logits, tau=1.0, hard=False)
        
        # Predict Next State
        decoder_input = torch.cat([emb_t, action_soft], dim=1)
        pred_next_logits = self.decoder_mlp(decoder_input)
        
        # Reshape for CrossEntropy
        pred_next_logits = pred_next_logits.view(z_t.size(0), 256, TOKEN_VOCAB)
        
        return pred_next_logits, action_logits

# --- 3. TRAINING LOOP (WITH ENTROPY FIX) ---
def main():
    print(f"Running LAM training on {DEVICE}")
    
    dataset = TokenTransitionsDataset(DATA_PATH, limit=2000)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = LatentActionModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    print("Starting Unsupervised Action Discovery (Entropy Regularized)...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        total_recon = 0
        total_entropy = 0
        
        for curr_z, next_z, _ in dataloader:
            curr_z, next_z = curr_z.to(DEVICE), next_z.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred_next_logits, action_logits = model(curr_z, next_z)
            
            # 1. Reconstruction Loss (Predicting the future)
            recon_loss = criterion(pred_next_logits.view(-1, TOKEN_VOCAB), next_z.view(-1))
            
            # 2. Entropy Regularization (THE FIX)
            # Calculate probabilities of actions
            probs = F.softmax(action_logits, dim=1)
            # Calculate average usage of each action in this batch
            avg_probs = torch.mean(probs, dim=0) 
            # Calculate Entropy: H(p) = - sum(p * log(p))
            # High entropy means we are using many actions. Low entropy means mode collapse.
            entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-8))
            
            # We want to MAXIMIZE entropy, so we SUBTRACT it from the loss.
            # 0.1 is the regularization strength (hyperparameter)
            loss = recon_loss - 0.1 * entropy 
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_entropy += entropy.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(dataloader):.4f} | Recon: {total_recon/len(dataloader):.4f} | Entropy: {total_entropy/len(dataloader):.4f}")

    print("Saving LAM...")
    torch.save(model.state_dict(), ARTIFACTS_PATH / "lam.pth")
    
    # --- 4. EVALUATION ---
    print("Generating Confusion Matrix...")
    model.eval()
    
    all_pred = []
    all_real = []
    
    with torch.no_grad():
        for curr_z, next_z, real_act in dataloader:
            curr_z, next_z = curr_z.to(DEVICE), next_z.to(DEVICE)
            
            _, action_logits = model(curr_z, next_z)
            pred_act = torch.argmax(action_logits, dim=1).cpu().numpy()
            
            all_pred.extend(pred_act)
            all_real.extend(real_act.numpy())
            
    # Plotting
    cm = confusion_matrix(all_real, all_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Discovered Latent Action (0-7)')
    plt.ylabel('Ground Truth Action (0=Noop, 1=Up, ...)')
    plt.title('Unsupervised Action Discovery (With Entropy Reg)')
    plt.savefig(ARTIFACTS_PATH / "action_confusion_matrix.png")
    print(f"SUCCESS! Saved matrix to {ARTIFACTS_PATH / 'action_confusion_matrix.png'}")

if __name__ == "__main__":
    main()