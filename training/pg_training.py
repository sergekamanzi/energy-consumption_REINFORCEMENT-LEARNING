from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from environment.custom_env import EnergyEnv
import os

def train_ppo():
    # Create vectorized environment
    env = make_vec_env(lambda: EnergyEnv(), n_envs=1)
    
    # Create save directory
    os.makedirs("models/pg", exist_ok=True)
    
    # Callback to save model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="models/pg/",
        name_prefix="ppo_energy"
    )
    
    # Create model
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
    
    # Train model
    model.learn(
        total_timesteps=200000,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Save final model
    model.save("models/pg/ppo_energy_final")
    env.close()

if __name__ == "__main__":
    train_ppo()