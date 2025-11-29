import os
import sys
import numpy as np
from pathlib import Path

# --- SETUP PATHS ---
current_path = Path(__file__).parent.resolve()
root_path = current_path.parent
sys.path.append(str(root_path))

try:
    from env.CommonsGame.CommonsGame.envs.env import CommonsGame
    from env.CommonsGame.CommonsGame.constants import bigMap
except ImportError as e:
    print("Error importing CommonsGame. Ensure the 'env' folder is in the project root.")
    print(f"Searching in: {root_path}")
    raise e

# --- AGENT LOGIC ---

class Agent:
    def get_action(self, obs, agent_index):
        raise NotImplementedError

class RandomAgent(Agent):
    def __init__(self, action_space):
        self.action_space = action_space
    
    def get_action(self, obs, agent_index):
        return self.action_space.sample()

class SimpleGreedyAgent(Agent):
    def __init__(self, action_space, agent_id):
        self.action_space = action_space
        self.agent_id = agent_id 
        
    def get_action(self, obs, agent_index):
        # Handle Death/Zap (Observation is None)
        if obs is None:
            return self.action_space.sample()

        # 1. Define green color range (Apples)
        # Heuristic for pure green: Low Red, High Green, Low Blue
        green_mask = (obs[:, :, 0] < 100) & (obs[:, :, 1] > 100) & (obs[:, :, 2] < 100)
        
        # 2. Get coordinates of all apples
        apple_coords = np.argwhere(green_mask)
        
        # If no apples are visible, explore randomly
        if len(apple_coords) == 0:
            return self.action_space.sample()
            
        # 3. Find "My" Position (Center of local view)
        agent_pos = np.array([obs.shape[0]//2, obs.shape[1]//2])
        
        # 4. Find the nearest apple
        distances = np.linalg.norm(apple_coords - agent_pos, axis=1)
        nearest_apple_idx = np.argmin(distances)
        target = apple_coords[nearest_apple_idx]
        
        # 5. Decide movement
        dy = target[0] - agent_pos[0]
        dx = target[1] - agent_pos[1]
        
        if abs(dy) > abs(dx):
            if dy < 0: return 1 # Up
            else:      return 2 # Down
        else:
            if dx < 0: return 3 # Left
            else:      return 4 # Right

# --- ENVIRONMENT SETUP ---

def convert_map(map_list):
    return [list("".join(row)) for row in map_list]

def collect_episode(env, agents, max_steps=1000):
    obs = env.reset()
    
    frames = []
    actions = []
    rewards = []

    for t in range(max_steps):
        all_actions = []
        
        for i, agent in enumerate(agents):
            current_obs = obs[i] if isinstance(obs, list) else obs
            act = agent.get_action(current_obs, i)
            all_actions.append(act)

        next_obs, reward, done, info = env.step(all_actions)

        # --- PROCESS FRAME FOR STORAGE ---
        raw_frame = obs[0] if isinstance(obs, list) else obs
        
        # Handle Death/Zap in Data Collection
        # If agent is dead, raw_frame is None. We must create a dummy black frame
        # to maintain the (64, 64, 3) shape consistency for the VQ-VAE.
        if raw_frame is None:
            save_frame = np.zeros((64, 64, 3), dtype=np.uint8)
        else:
            save_frame = raw_frame
            
            # Crop to 64x64 if necessary
            if save_frame.shape[0] > 64:
                save_frame = save_frame[:64, :64, :]

            # Ensure uint8
            if save_frame.dtype != np.uint8:
                if save_frame.max() <= 1.0:
                     save_frame = (save_frame * 255).astype(np.uint8)
                else:
                     save_frame = save_frame.astype(np.uint8)
            
        frames.append(save_frame)
        actions.append(all_actions)
        rewards.append(reward)

        obs = next_obs

        if isinstance(done, list):
            if any(done): break
        elif done:
            break

    return {
        "frames": np.array(frames),
        "actions": np.array(actions, dtype=np.uint8),
        "rewards": np.array(rewards, dtype=np.float32),
    }

def main():
    out_dir = root_path / "data" / "episodes"
    
    print(f"Saving data to: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    mapSketch = convert_map(bigMap)

    print("Initializing CommonsGame Environment...")
    env = CommonsGame(
        numAgents=2,
        visualRadius=32, 
        mapSketch=mapSketch,
        fullState=False 
    )

    agents = [
        SimpleGreedyAgent(env.action_space, agent_id=0),
        RandomAgent(env.action_space)
    ]

    print("Starting data collection...")
    total_frames = 0
    
    for i in range(50): 
        ep = collect_episode(env, agents)
        
        if i == 0:
            print(f"[DEBUG] Frame Shape: {ep['frames'].shape}")
            print(f"[DEBUG] Actions Shape: {ep['actions'].shape}")
            print(f"[DEBUG] Data Type: {ep['frames'].dtype}")
        
        np.savez_compressed(out_dir / f"episode_{i:04d}.npz", **ep)
        
        total_frames += len(ep['frames'])
        print(f"[OK] Saved episode {i:04d}. Total frames accumulated: {total_frames}")

if __name__ == "__main__":
    main()