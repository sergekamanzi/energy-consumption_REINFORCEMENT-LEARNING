from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from environment.custom_env import EnergyEnv
import os

def train_actor_critic():
    # Create vectorized environment
    env = make_vec_env(lambda: EnergyEnv(), n_envs=4)  # Multiple environments for faster training
    
    # Create save directory
    os.makedirs("models/actor_critic", exist_ok=True)
    
    # Callback to save model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="models/actor_critic/",
        name_prefix="a2c_energy"
    )
    
    # Create Actor-Critic model (A2C implementation)
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
        tensorboard_log="logs/actor_critic"
    )
    
    # Train model
    model.learn(
        total_timesteps=200000,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Save final model
    model.save("models/actor_critic/a2c_energy_final")
    env.close()

if __name__ == "__main__":
    train_actor_critic()