import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json

class LLMAuditingEnv(gym.Env):
    """Custom Environment that follows the gym interface for LLM Auditing"""
    def __init__(self):
        super(LLMAuditingEnv, self).__init__()
        
        # Load the dataset you just built
        with open("data/eval_subset.json", "r") as f:
            self.dataset = json.load(f)
            
        self.current_file_idx = 0
        self.attempt_count = 0
        self.max_attempts = 5 # The strict budget from your proposal
        
        # Action Space: 5 different extraction prompts (0 to 4)
        # e.g., 0: "Complete this code:", 1: "License:", 2: "Boilerplate:", etc.
        self.action_space = spaces.Discrete(5)
        
        # State Space: We observe the current file and how many attempts we've used
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Pick a random code file from our 150 samples
        self.current_file_idx = np.random.randint(0, len(self.dataset))
        self.attempt_count = 0
        return self._get_obs(), {}

    def _get_obs(self):
        # Normalize the state values between 0 and 1 for better neural network training
        return np.array([
            self.current_file_idx / len(self.dataset), 
            self.attempt_count / self.max_attempts
        ], dtype=np.float32)

    def step(self, action):
        self.attempt_count += 1
        
        # MVP LLM Simulation: 
        # Instead of loading a massive model that crashes your 8GB RAM, we assign 
        # statistical success probabilities to each of the 5 prompts. 
        # Let's say action 2 ("Boilerplate:") is statistically the most effective.
        success_probabilities = [0.1, 0.2, 0.75, 0.05, 0.15]
        chance = success_probabilities[action]
        
        # Did the LLM leak the exact training data?
        extracted = np.random.random() < chance
        
        # The exact Reward logic from your proposal:
        if extracted:
            reward = 10.0  # Massive reward for successfully triggering memorization
            terminated = True
        else:
            reward = -1.0  # Penalty for wasting an attempt
            terminated = False
            
        # Stop if we hit our 5-attempt budget
        truncated = self.attempt_count >= self.max_attempts
        if truncated:
            terminated = True
            
        return self._get_obs(), reward, terminated, truncated, {}