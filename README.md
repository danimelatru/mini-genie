# Mini-Genie: Unsupervised Action Discovery in Sequential Social Dilemmas

## Project Overview
This project is a technical implementation of a Generative World Model (GWM) applied to Multi-Agent Reinforcement Learning environments, specifically the **Commons Game** (Sequential Social Dilemma).

Inspired by DeepMind's **Genie** (2024), the goal is to learn a latent action space and environment dynamics purely from unlabeled video data, effectively discovering concepts like "Harvesting" or "Zapping" without supervision.

## 🚀 The Pipeline

The architecture follows a strict 3-stage pipeline designed to handle the complexity of small objects (apples) and mode collapse in action discovery.

### 1. Data Collection (`collect_data.py`)
- **Environment:** Harvest (Commons Game).
- **Agent Policy:** Implemented a custom **Heuristic Greedy Agent** using computer vision logic to force causal interactions (resource consumption). Random agents failed to trigger scarcity dynamics.
- **Volume:** 50,000 frames (RGB, 64x64).

### 2. Spatiotemporal Compression (`train_vqvae.py`)
- **Model:** VQ-VAE with Residual Blocks.
- **Challenge:** "Posterior Collapse" where small objects (apples) vanished during compression.
- **Solution:** - Implemented a **High-Res Latent Space (16x16)** instead of the standard 8x8.
    - Designed a **Spatial Weighted L1 Loss** that penalizes errors on non-background pixels by a factor of 20x.
- **Result:** High-fidelity reconstruction of small, single-pixel resources.

### 3. Latent Action Discovery (`train_lam_dynamics.py`)
- **Model:** Latent Action Model (LAM) + Dynamics Predictor.
- **Challenge:** "Mode Collapse". The model initially mapped all transitions to a single latent action to minimize loss safely.
- **Solution:** Introduced **Entropy Regularization** ($\lambda=0.1$) to the loss function, maximizing the entropy of the average action distribution.
- **Result:** Successfully disentangled distinct latent behaviors unsupervised.

## 🛠️ Installation & Usage

### Prerequisites
The project uses specific legacy versions to ensure compatibility with the `gym==0.21` environment.

```bash
pip install "gym==0.21.0" "numpy<1.24" "scikit-image<0.20"
pip install torch torchvision
pip install "pandas<2.0" "seaborn<0.12" "scikit-learn<1.3" "matplotlib<3.8"