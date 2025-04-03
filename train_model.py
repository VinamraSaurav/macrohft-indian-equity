import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import os
import pandas as pd
from Environment.high_level_env import HighLevelEnv
from dqn_agent import DQNAgent

# 🛠️ Load Preprocessed Data with 11 Indicators
data = pd.read_csv("../macroHFT_indian_equity/Data/HDFCBANK_train.csv")

# 🎯 Hyperparameters
NUM_EPISODES = 1000
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.0005
TARGET_UPDATE = 10      # Update target network every 10 episodes
MEMORY_SIZE = 10000
SAVE_INTERVAL = 100      # Save model every 100 episodes

# ✅ Initialize environment and agent with 11 indicators
env = HighLevelEnv(data)
agent = DQNAgent(state_size=env.observation_space.shape[0], action_size=env.action_space.n)
optimizer = optim.Adam(agent.q_network.parameters(), lr=LEARNING_RATE)

# 📊 Experience Replay Buffer
memory = deque(maxlen=MEMORY_SIZE)

# 📈 Tracking performance metrics
rewards_per_episode = []
net_worths = []

# 🔥 Training Loop
epsilon = EPSILON_START

for episode in range(1, NUM_EPISODES + 1):
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    total_reward = 0
    done = False
    
    while not done:
        # ε-greedy action selection
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = agent.act(state)

        # Take step in the environment
        next_state, reward, done, _ = env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

        # Store experience in memory
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        # Experience Replay
        if len(memory) >= BATCH_SIZE:
            batch = random.sample(memory, BATCH_SIZE)
            
            states, actions, rewards, next_states, dones = zip(*batch)
            
            states = torch.cat(states)
            actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
            next_states = torch.cat(next_states)
            dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

            # Q-Learning update
            q_values = agent.q_network(states).gather(1, actions)
            
            with torch.no_grad():
                max_next_q_values = agent.target_network(next_states).max(1)[0].unsqueeze(1)
                target_q_values = rewards + (GAMMA * max_next_q_values * (1 - dones))

            # Loss calculation
            loss = F.mse_loss(q_values, target_q_values)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Decay epsilon
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    # Sync target network every TARGET_UPDATE episodes
    if episode % TARGET_UPDATE == 0:
        agent.target_network.load_state_dict(agent.q_network.state_dict())

    # Save the model periodically
    if episode % SAVE_INTERVAL == 0:
        model_dir = "../macroHFT_indian_equity/models/"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"dqn_model_ep{episode}.pth")
        torch.save(agent.q_network.state_dict(), model_path)
        print(f"💾 Model saved at {model_path}")

    # Store episode metrics
    rewards_per_episode.append(total_reward)
    net_worths.append(env.net_worth)

    # Display progress
    print(f"Episode: {episode}/{NUM_EPISODES} | Reward: {total_reward:.2f} | Net Worth: {env.net_worth:.2f} | Epsilon: {epsilon:.4f}")

# Save final model
final_model_path = "../macroHFT_indian_equity/models/dqn_final_model.pth"
torch.save(agent.q_network.state_dict(), final_model_path)
print(f"✅ Training complete. Model saved as {final_model_path}")
