import pandas as pd
import numpy as np
from Environment.low_level_env import LowLevelEnv

# ✅ Load preprocessed HDFCBANK dataset
data = pd.read_csv('../macroHFT_indian_equity/Data/HDFCBANK_train.csv')

# ✅ Initialize Low-Level Environment
env = LowLevelEnv(data)

# 🎯 **Parameters**
num_episodes = 5
num_steps = 50  # Steps per episode

for episode in range(num_episodes):
    print(f"\n🎯 Episode {episode + 1} -----------------------------")
    
    # ✅ Reset environment at the start of each episode
    state = env.reset()

    total_reward = 0  # Track cumulative reward per episode
    
    for step in range(num_steps):
        # ✅ Sample random action (0 = No action, 1 = Partial Buy, 2 = Partial Sell)
        action = env.action_space.sample()
        
        # ✅ Execute action
        next_state, reward, done, _ = env.step(action)
        
        # ✅ Accumulate reward
        total_reward += reward

        # ✅ Display environment state
        env.render()
        
        if done:
            print("✅ Episode finished.")
            break

    print(f"🏁 Episode {episode + 1} finished with Total Reward: {total_reward:.2f}")
