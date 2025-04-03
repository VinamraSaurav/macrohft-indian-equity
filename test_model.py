import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from Environment.high_level_env import HighLevelEnv
from Environment.low_level_env import LowLevelEnv
from dqn_agent import DQNAgent

# 🔥 Load Test Data
test_data = pd.read_csv("Data/HDFCBANK_test.csv")

# 🎯 Initialize High-Level and Low-Level Environments
high_env = HighLevelEnv(test_data)
low_env = LowLevelEnv(test_data)

# ✅ Load the single DQN model
model_path = "./models/dqn_final_model.pth"

# Load the model into both agents
high_agent = DQNAgent(state_size=high_env.observation_space.shape[0], action_size=high_env.action_space.n)
low_agent = DQNAgent(state_size=low_env.observation_space.shape[0], action_size=low_env.action_space.n)

# Load the same model for both high- and low-level agents
high_agent.q_network.load_state_dict(torch.load(model_path))
low_agent.q_network.load_state_dict(torch.load(model_path))

# Set models to evaluation mode
high_agent.q_network.eval()
low_agent.q_network.eval()

# 📈 Metrics to track
net_worths = []
rewards = []
high_rewards = []
low_rewards = []
win_count = 0
loss_count = 0
initial_balance = 10000  # Initial balance for the environment

# 🚀 Testing Loop
NUM_EPISODES = 10  # Run multiple episodes for better evaluation

for episode in range(1, NUM_EPISODES + 1):
    
    # Reset environments
    high_state = high_env.reset()
    low_state = low_env.reset()

    # Convert states to tensors
    high_state = torch.tensor(high_state, dtype=torch.float32).unsqueeze(0)
    low_state = torch.tensor(low_state, dtype=torch.float32).unsqueeze(0)

    total_reward = 0
    high_episode_reward = 0
    low_episode_reward = 0
    done = False

    while not done:
        # 🔥 High-level agent action
        high_action = high_agent.act(high_state)
        high_next_state, high_reward, high_done, _ = high_env.step(high_action)
        high_episode_reward += high_reward

        # 🔥 Low-level agent action
        low_action = low_agent.act(low_state)
        low_next_state, low_reward, low_done, _ = low_env.step(low_action)
        low_episode_reward += low_reward

        # Update states
        high_state = torch.tensor(high_next_state, dtype=torch.float32).unsqueeze(0)
        low_state = torch.tensor(low_next_state, dtype=torch.float32).unsqueeze(0)

        # Calculate combined reward with weighted importance
        total_reward += (0.7 * high_reward + 0.3 * low_reward)

        # Check if either environment signals termination
        done = high_done or low_done

    # Store metrics
    final_net_worth = high_env.net_worth  # Track the high-level agent's final net worth
    net_worths.append(final_net_worth)
    rewards.append(total_reward)
    high_rewards.append(high_episode_reward)
    low_rewards.append(low_episode_reward)

    # Calculate win/loss
    if final_net_worth > initial_balance:
        win_count += 1
    else:
        loss_count += 1

    print(f"Episode: {episode}/{NUM_EPISODES} | Net Worth: {final_net_worth:.2f} | "
          f"High-Level Reward: {high_episode_reward:.2f} | Low-Level Reward: {low_episode_reward:.2f} | "
          f"Total Reward: {total_reward:.2f}")

# ✅ Final Statistics
print("\n📊 Evaluation Results:")
print(f"✅ Episodes: {NUM_EPISODES}")
print(f"📈 Average Net Worth: {np.mean(net_worths):.2f}")
print(f"💰 Max Net Worth: {np.max(net_worths):.2f}")
print(f"📉 Min Net Worth: {np.min(net_worths):.2f}")
print(f"🏆 Wins: {win_count} | ❌ Losses: {loss_count}")
print(f"⚡ Win Rate: {100 * win_count / NUM_EPISODES:.2f}%")

# 📊 Plot Net Worth over Episodes
plt.figure(figsize=(14, 6))
plt.plot(net_worths, label='Net Worth', color='blue', marker='o')
plt.axhline(initial_balance, color='r', linestyle='--', label='Initial Balance')
plt.title('Net Worth Over Test Episodes')
plt.xlabel('Episode')
plt.ylabel('Net Worth')
plt.legend()
plt.grid(True)
plt.show()

# 📊 Plot High-Level vs Low-Level Rewards
plt.figure(figsize=(14, 6))
plt.plot(high_rewards, label='High-Level Rewards', color='green', marker='o')
plt.plot(low_rewards, label='Low-Level Rewards', color='orange', marker='x')
plt.title('High-Level vs Low-Level Rewards')
plt.xlabel('Episode')
plt.ylabel('Rewards')
plt.legend()
plt.grid(True)
plt.show()
