import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.0005):
        self.state_size = state_size
        self.action_size = action_size

        # 🛠️ Q-Networks
        self.q_network = self.build_model()
        self.target_network = self.build_model()
        
        # Copy initial weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

    def build_model(self):
        """Create the neural network model."""
        model = nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )
        return model

    def act(self, state, epsilon=0.0):
        """Epsilon-greedy action selection."""
        if np.random.rand() < epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                q_values = self.q_network(state)
                return torch.argmax(q_values).item()

    def update_target_network(self):
        """Copy weights from Q-network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
