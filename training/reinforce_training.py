import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from stable_baselines3.common.vec_env import DummyVecEnv
from environment.custom_env import EnergyEnv
import numpy as np
import os

class PolicyNetwork(nn.Module):
    def __init__(self, input_shape, num_actions, hidden_size=64):
        super(PolicyNetwork, self).__init__()
        # Flatten the grid input
        self.flatten = nn.Flatten()
        input_dim = input_shape[0] * input_shape[1]
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_actions)
        
    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=-1)
    
    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

def train_reinforce():
    # Create environment
    env = DummyVecEnv([lambda: EnergyEnv()])
    obs_shape = env.observation_space.shape
    num_actions = env.action_space.n
    
    # Initialize policy
    policy = PolicyNetwork(obs_shape, num_actions)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    
    # Training parameters
    max_episodes = 100
    gamma = 0.99
    
    # Create save directory
    os.makedirs("models/reinforce", exist_ok=True)
    
    for episode in range(max_episodes):
        state = env.reset()
        done = False
        log_probs = []
        rewards = []
        
        while not done:
            action, log_prob = policy.act(state)
            state, reward, done, _ = env.step([action])
            log_probs.append(log_prob)
            rewards.append(reward)
        
        # Calculate discounted returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        
        # Normalize returns
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate policy loss
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        # Update policy
        optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()
        
        # Print training progress
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {sum(rewards):.1f}")
            
            # Save checkpoint
            if episode % 100 == 0:
                torch.save(policy.state_dict(), 
                          f"models/reinforce/reinforce_energy_{episode}.pt")
    
    # Save final model
    torch.save(policy.state_dict(), "models/reinforce/reinforce_energy_final.pt")
    env.close()

if __name__ == "__main__":
    train_reinforce()