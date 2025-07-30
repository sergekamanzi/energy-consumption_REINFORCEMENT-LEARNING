from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from environment.custom_env import EnergyEnv
import os

def train_dqn():
    # Create vectorized environment
    env = make_vec_env(lambda: EnergyEnv(), n_envs=1)
    
    # Create save directory
    os.makedirs("models/dqn", exist_ok=True)
    
    # Callback to save model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="models/dqn/",
        name_prefix="dqn_energy"
    )
    
    # Create model
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
    
    # Train model
    model.learn(
        total_timesteps=150000,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Save final model
    model.save("models/dqn/dqn_energy_final")
    env.close()

if __name__ == "__main__":
    train_dqn()