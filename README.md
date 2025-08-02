# Energy Usage Optimization - Reinforcement Learning Project

This project implements a reinforcement learning environment for optimizing household energy usage. The agent navigates a 5x5 grid representing a smart home with various appliances, each with different energy impacts. The goal is to learn policies that maximize energy efficiency while maintaining appliance functionality.

The environment features discrete movement actions (up, down, left, right) and includes appliances like AC, fridge, TV, oven, lights, PC, washer, and solar panels. Each appliance affects the energy balance differently, with some draining energy (oven: -4) and others generating it (solar: +2). The meter tracks current consumption, while reaching the goal cell completes the episode with bonus rewards.

Four reinforcement learning algorithms are implemented: DQN, PPO, A2C, and REINFORCE. Each has been tuned with specific hyperparameters for this environment. DQN uses experience replay and ε-greedy exploration, while the policy gradient methods (PPO, A2C, REINFORCE) employ different approaches to policy optimization.

The project includes comprehensive training scripts, testing functionality, and visualization tools. Performance metrics track cumulative rewards, training stability, and generalization across different initial states. The best-performing model (DQN) achieved an average reward of 14.60, demonstrating effective energy optimization strategies.

Future improvements could include implementing Double DQN, prioritized experience replay, or hybrid approaches combining the strengths of different algorithms. The modular design allows for easy extension with additional appliances or more complex energy dynamics.
