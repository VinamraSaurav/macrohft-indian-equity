import numpy as np
import gym
from gym import spaces

class LowLevelEnv(gym.Env):
    def __init__(self, data, initial_balance=10000):
        super(LowLevelEnv, self).__init__()
        
        self.data = data
        self.current_step = 0
        
        # ✅ Observation space (11 features)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(11,), 
            dtype=np.float32
        )
        
        # Actions: 0 = No action, 1 = Partial Buy, 2 = Partial Sell
        self.action_space = spaces.Discrete(3)
        
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0
        self.net_worth = self.balance

    def reset(self):
        """Reset the environment to the initial state."""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.net_worth = self.balance
        return self._next_observation()
    
    def _next_observation(self):
        """Create the observation vector (11 features)."""
        row = self.data.iloc[self.current_step]

        # OHLCV
        obs = [
            row['Open'], row['High'], row['Low'], row['Close'], row['Volume'],
            row['SMA_20'], row['EMA_20']
        ]

        # Price (Low + High) / 2
        price = (row['Low'] + row['High']) / 2
        obs.append(price)

        # Balance, Net worth, and Position
        obs.extend([self.balance, self.net_worth, self.position])

        return np.array(obs, dtype=np.float32)

    def step(self, action):
        """Execute the given action and update the environment."""
        done = False
        reward = 0
        prev_worth = self.net_worth
        
        # Current price for trading
        price = self.data['Close'].iloc[self.current_step]

        # Execute action
        if action == 1:  # Partial Buy
            if self.balance >= 0.5 * price:
                self.position += 0.5
                self.balance -= 0.5 * price

        elif action == 2:  # Partial Sell
            if self.position >= 0.5:
                self.position -= 0.5
                self.balance += 0.5 * price

        # Update net worth
        self.net_worth = self.balance + (self.position * price)
        reward = self.net_worth - prev_worth
        
        # Move to the next step
        self.current_step += 1
        if self.current_step >= len(self.data):
            done = True
        
        # Next observation
        next_state = self._next_observation() if not done else np.zeros((11,), dtype=np.float32)

        return next_state, reward, done, {}

    def render(self):
        """Display the environment state."""
        print(f'Step: {self.current_step}, Balance: {self.balance:.2f}, Net Worth: {self.net_worth:.2f}, Position: {self.position:.2f}')
