import argparse
import os
import numpy as np
import pygame
import imageio
from environment.custom_env import EnergyEnv
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# REINFORCE Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, input_shape, num_actions, hidden_size=64):
        super(PolicyNetwork, self).__init__()
        self.flatten = nn.Flatten()
        input_dim = input_shape[0] * input_shape[1]
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_actions)
        
    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=-1)
    
    def act(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.forward(state)
        if deterministic:
            action = torch.argmax(probs)
            return action.item()
        else:
            m = Categorical(probs)
            action = m.sample()
            return action.item(), m.log_prob(action)

# Training Functions
def train_dqn():
    env = make_vec_env(lambda: EnergyEnv(), n_envs=1)
    os.makedirs("models/dqn", exist_ok=True)
    
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-3,
        buffer_size=70000,
        learning_starts=1000,
        batch_size=32,
        gamma=0.99,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        tensorboard_log="logs/dqn"
    )
    
    model.learn(total_timesteps=150000)
    model.save("models/dqn/dqn_energy_final")
    env.close()

def train_ppo():
    env = make_vec_env(lambda: EnergyEnv(), n_envs=1)
    os.makedirs("models/ppo", exist_ok=True)
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log="logs/ppo"
    )
    
    model.learn(total_timesteps=200000)
    model.save("models/ppo/ppo_energy_final")
    env.close()

def train_a2c():
    env = make_vec_env(lambda: EnergyEnv(), n_envs=4)
    os.makedirs("models/a2c", exist_ok=True)
    
    model = A2C(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=7e-4,
        n_steps=5,
        gamma=0.99,
        gae_lambda=1.0,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="logs/a2c"
    )
    
    model.learn(total_timesteps=200000)
    model.save("models/a2c/a2c_energy_final")
    env.close()

def train_reinforce():
    env = DummyVecEnv([lambda: EnergyEnv()])
    os.makedirs("models/reinforce", exist_ok=True)
    
    obs_shape = env.observation_space.shape
    num_actions = env.action_space.n
    policy = PolicyNetwork(obs_shape, num_actions)
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)
    gamma = 0.99
    
    for episode in range(1000):
        state = env.reset()
        done = [False]
        log_probs = []
        rewards = []
        
        while not done[0]:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state[0]).unsqueeze(0)
                probs = policy.forward(state_tensor)
                probs = torch.clamp(probs, min=1e-8, max=1.0-1e-8)
            
            action, log_prob = policy.act(state[0])
            state, reward, done, _ = env.step([action])
            log_probs.append(log_prob)
            rewards.append(float(reward[0]))
        
        # Calculate returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(np.array(returns, dtype=np.float32))
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Update policy
        policy_loss = torch.stack([-log_prob * R for log_prob, R in zip(log_probs, returns)]).sum()
        
        optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {sum(rewards):.1f}")
            
    torch.save(policy.state_dict(), "models/reinforce/reinforce_energy_final.pt")
    env.close()

# Testing Functions
def test_random_agent():
    env = EnergyEnv(render_mode="human")
    frames = []
    obs, _ = env.reset()
    clock = pygame.time.Clock()
    
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        frame = np.transpose(
            pygame.surfarray.array3d(pygame.display.get_surface()), 
            (1, 0, 2)
        )
        frames.append(frame)
        
        if done:
            obs, _ = env.reset()
            
        clock.tick(5)
    
    os.makedirs("results", exist_ok=True)
    imageio.mimsave("results/random_agent.gif", frames, fps=5)
    env.close()

def test_trained_agent(model_path, algorithm):
    env = EnergyEnv(render_mode="human")
    frames = []
    episode_rewards = []
    
    try:
        if algorithm == "dqn":
            model = DQN.load(f"{model_path}.zip")
        elif algorithm == "ppo":
            model = PPO.load(f"{model_path}.zip")
        elif algorithm == "a2c":
            model = A2C.load(f"{model_path}.zip")
        elif algorithm == "reinforce":
            policy = PolicyNetwork(env.observation_space.shape, env.action_space.n)
            policy.load_state_dict(torch.load(model_path))
            model = policy
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    except FileNotFoundError:
        print(f"\nError: Model not found at {model_path}")
        print("Please train the model first using:")
        print(f"python main.py --train {algorithm}")
        env.close()
        return
    
    clock = pygame.time.Clock()
    
    for episode in range(3):  # Now testing 10 episodes
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            if algorithm == "reinforce":
                action = model.act(obs, deterministic=True)
            else:
                action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            frame = np.transpose(
                pygame.surfarray.array3d(pygame.display.get_surface()), 
                (1, 0, 2)
            )
            frames.append(frame)
            
            clock.tick(5)
        
        episode_rewards.append(total_reward)
        print(f"Episode {episode+1}: Reward = {total_reward:.1f}")
    
    # Print summary statistics
    print("\n=== Testing Summary ===")
    print(f"Average Reward: {np.mean(episode_rewards):.2f}")
    print(f"Standard Deviation: {np.std(episode_rewards):.2f}")
    print(f"Minimum Reward: {np.min(episode_rewards):.2f}")
    print(f"Maximum Reward: {np.max(episode_rewards):.2f}")
    
    os.makedirs("results", exist_ok=True)
    imageio.mimsave(f"results/{algorithm}_agent.gif", frames, fps=5)
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL Training and Testing for Energy Environment")
    parser.add_argument("--train", choices=["dqn", "ppo", "a2c", "reinforce", "all"], 
                       help="Train models (specify algorithm or 'all')")
    parser.add_argument("--test", choices=["random", "dqn", "ppo", "a2c", "reinforce"], 
                       help="Test a trained model")
    args = parser.parse_args()
    
    # Training
    if args.train in ["dqn", "all"]:
        print("\n=== Training DQN ===")
        train_dqn()
    
    if args.train in ["ppo", "all"]:
        print("\n=== Training PPO ===")
        train_ppo()
    
    if args.train in ["a2c", "all"]:
        print("\n=== Training A2C ===")
        train_a2c()
    
    if args.train in ["reinforce", "all"]:
        print("\n=== Training REINFORCE ===")
        train_reinforce()
    
    # Testing
    if args.test == "random":
        print("\n=== Testing Random Agent ===")
        test_random_agent()
    elif args.test == "dqn":
        print("\n=== Testing DQN ===")
        test_trained_agent("models/dqn/dqn_energy_final", "dqn")
    elif args.test == "ppo":
        print("\n=== Testing PPO ===")
        test_trained_agent("models/ppo/ppo_energy_final", "ppo")
    elif args.test == "a2c":
        print("\n=== Testing A2C ===")
        test_trained_agent("models/a2c/a2c_energy_final", "a2c")
    elif args.test == "reinforce":
        print("\n=== Testing REINFORCE ===")
        test_trained_agent("models/reinforce/reinforce_energy_final.pt", "reinforce")