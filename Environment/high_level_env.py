import numpy as np
import gym
from gym import spaces
from Environment.low_level_env import LowLevelEnv

class HighLevelEnv(gym.Env):
    def __init__(self, data, low_level_steps=5, initial_balance=10000):
        """High-level environment managing the low-level agent."""
        super(HighLevelEnv, self).__init__()

        self.data = data
        self.low_level_steps = low_level_steps
        self.current_step = 0

        # ✅ Observation space (11 features)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(11,), 
            dtype=np.float32
        )

        # Actions: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)

        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.net_worth = self.balance
        self.positions = []

        # ✅ Low-level agent
        self.low_level_env = LowLevelEnv(data, initial_balance)

    def reset(self):
        """Reset the high-level environment and the low-level agent."""
        self.current_step = 0
        self.balance = self.initial_balance
        self.net_worth = self.balance
        self.positions = []
        self.low_level_env.reset()

        return self._create_observation()

    def _create_observation(self):
        """Create the observation vector."""
        row = self.data.iloc[self.current_step]

        # OHLCV
        obs = [
            row['Open'], row['High'], row['Low'], row['Close'], row['Volume'],
            row['SMA_20'], row['EMA_20']
        ]

        # Price (Low + High) / 2
        price = (row['Low'] + row['High']) / 2
        obs.append(price)

        # Balance, Net worth, and Position count
        obs.extend([self.balance, self.net_worth, len(self.positions)])

        return np.array(obs, dtype=np.float32)

    def step(self, action):
        """Execute the high-level action."""
        done = False
        reward = 0

        # High-level actions
        if action == 1:  # Buy
            self.positions.append(self.low_level_env.position)
            self.balance -= self.low_level_env.position * self.data['Close'].iloc[self.current_step]

        elif action == 2:  # Sell
            if self.positions:
                position = self.positions.pop(0)
                self.balance += position * self.data['Close'].iloc[self.current_step]

        # Run low-level agent for `low_level_steps`
        for _ in range(self.low_level_steps):
            low_action = self.low_level_env.action_space.sample()
            _, low_reward, low_done, _ = self.low_level_env.step(low_action)
            reward += low_reward
            if low_done:
                done = True
                break

        # Update net worth
        self.net_worth = self.balance + sum(self.positions) * self.data['Close'].iloc[self.current_step]
        
        # Move to the next step
        self.current_step += 1
        if self.current_step >= len(self.data):
            done = True

        next_state = self._create_observation() if not done else np.zeros((11,), dtype=np.float32)
        
        return next_state, reward, done, {}

    def render(self):
        """Render the current state."""
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Net Worth: {self.net_worth:.2f}, Positions: {len(self.positions)}")
