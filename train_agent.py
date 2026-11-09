import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from llm_env import LLMAuditingEnv

# Hardware acceleration: Route calculations to Apple's Metal GPU for massive speedups
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Hardware Acceleration Status: Training on {device.type.upper()}")

# The Policy Network: Takes the current state (file + attempts used) and outputs prompt probabilities
class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 5) # 5 possible extraction prompts (actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=-1)

def train_reinforce():
    env = LLMAuditingEnv()
    policy = PolicyNetwork().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=0.01)

    num_episodes = 500
    gamma = 0.99  # Discount factor for future rewards

    print("Initializing Adaptive Prompt Selector Training...")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        
        terminated = False
        truncated = False
        
        # Play one full episode (max 5 attempts based on your budget)
        while not (terminated or truncated):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            # Get prompt probabilities from our network
            probs = policy(state_tensor)
            m = Categorical(probs)
            
            # Sample an action (prompt) based on the probabilities
            action = m.sample()
            
            # Step the environment
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            
            # Store the log probability and reward for the REINFORCE math
            log_probs.append(m.log_prob(action))
            rewards.append(reward)
            
            state = next_state
            
        # Calculate discounted returns (the core math of REINFORCE)
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns).to(device)
        
        # Standardize returns for training stability
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        # Calculate policy loss: -log_prob * return
        loss = []
        for log_prob, G in zip(log_probs, returns):
            loss.append(-log_prob * G)
        loss = torch.cat(loss).sum()
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{num_episodes} | Latest Episode Total Reward: {sum(rewards)}")

    print("\nTraining complete! The agent has successfully learned the optimal extraction prompt strategy.")
    
    # Save the trained model weights
    torch.save(policy.state_dict(), "adaptive_prompt_policy.pth")
    print("Model successfully saved to 'adaptive_prompt_policy.pth'")

if __name__ == "__main__":
    train_reinforce()