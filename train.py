import ale_py
import gymnasium as gym
import time
import numpy as np
import torch # For checking device
import os
import matplotlib.pyplot as plt
from collections import deque
import gymnasium.wrappers as gym_wrappers

from agents.ppo import PPOAgent

gym.register_envs(ale_py)

# --- Hyperparameters ---
ENV_NAME = 'ALE/Breakout-v5' # 'ALE/DonkeyKong-v5'
RENDER_MODE = None # 'human' or None
FRAMESKIP = 4

NUM_EPISODES = 10000  # Number of episodes to train for
MAX_STEPS_PER_EPISODE = 10000 # Max steps before truncating an episode
EPSILON_START = 1.0    # Starting value of epsilon
EPSILON_END = 0.005     # Minimum value of epsilon
EPSILON_DECAY = 0.999  # Multiplicative factor for decaying epsilon
SAVE_EVERY = 100 # Save the model and plot every N episodes

# --- Hyperparameters for Interval LR Annealing ---
ANNEAL_LR_ON_INTERVAL = True     # Enable LR annealing at regular intervals
LR_DECAY_FACTOR = 0.95           # Factor to multiply LR by (e.g., 0.95 means 5% decay)
LR_DECAY_ROLLOUTS = 200          # Decay LR every N rollouts (agent updates)
MIN_LR = 1e-6                    # Minimum learning rate

# --- Directory Setup ---
MODEL_DIR = "models"
PLOT_DIR = "plots"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
# Update model and plot paths to be specific to the environment
ENV_NAME_LOWER = ENV_NAME.split('/')[-1].split('-')[0].lower() # e.g., "donkeykong"
MODEL_PATH = os.path.join(MODEL_DIR, f"{ENV_NAME_LOWER}_ppo.pth")
PLOT_PATH_PREFIX = os.path.join(PLOT_DIR, f"{ENV_NAME_LOWER}_scores_ppo")

# --- Environment Setup ---
env = gym_wrappers.AtariPreprocessing(
    gym.make(
        ENV_NAME, 
        render_mode=RENDER_MODE, 
        frameskip=1
    ),
    screen_size=84,
    grayscale_obs=True,
    grayscale_newaxis=False, # Results in (H, W) for grayscale, not (H, W, 1)
    frame_skip=FRAMESKIP,    # Stacks and max-pools last 4 frames
    scale_obs=True           # Normalizes observations to [0, 1] and converts to float32
)
env = gym_wrappers.FrameStackObservation(env, stack_size=4) # Apply FrameStack

state_shape = env.observation_space.shape
action_size = env.action_space.n

print(f"State shape: {state_shape}")
print(f"Action size: {action_size}")

# --- Agent Setup ---
# agent = DQNAgent(state_space_shape=state_shape, action_space_size=action_size,
#                  buffer_size=10000, batch_size=64, gamma=0.99, lr=1e-4, tau=1e-3, update_every=4)

agent = PPOAgent(state_space_shape=state_shape, action_space_size=action_size,
                 lr=2.5e-4,             # Initial PPO learning rate
                 gamma=0.99,
                 gae_lambda=0.95,
                 clip_epsilon=0.1,
                 n_steps=512,
                 ppo_epochs=8,
                 mini_batch_size=64,
                 entropy_coeff=0.01,
                 value_loss_coeff=0.5,
                 max_grad_norm=0.5,
                 fc_units=512,
                 anneal_lr_on_interval=ANNEAL_LR_ON_INTERVAL,
                 lr_decay_factor=LR_DECAY_FACTOR,
                 lr_decay_rollouts=LR_DECAY_ROLLOUTS,
                 min_lr=MIN_LR)

# Try to load a pre-trained model
# try:
#     agent.load(MODEL_PATH)
#     print(f"Loaded model from {MODEL_PATH}")
#     # If loading a model, you might want to adjust epsilon if you're evaluating
#     EPSILON_START = EPSILON_END # Start with low epsilon if evaluating
# except FileNotFoundError:
#     print(f"No pre-trained model found at {MODEL_PATH}, starting from scratch.")


# --- Training Loop ---
epsilon = EPSILON_START
total_steps = 0
scores_window = deque(maxlen=100)
all_episode_scores = []
all_average_scores = []

print(f"Training on {agent.device}")

for i_episode in range(1, NUM_EPISODES + 1):
    observation, info = env.reset()
    # The observation from env.reset() is a LazyFrames object for Atari if not using wrappers.
    # Convert to numpy array for the agent.
    if not isinstance(observation, np.ndarray):
        observation = np.array(observation)

    current_episode_reward = 0

    for t in range(MAX_STEPS_PER_EPISODE):
        if RENDER_MODE == 'human':
            env.render()

        action = agent.act(observation, epsilon)
        next_observation, reward, terminated, truncated, info = env.step(action)

        if not isinstance(next_observation, np.ndarray):
            next_observation = np.array(next_observation)

        # Store experience in replay memory
        agent.learn(observation, action, reward, next_observation, terminated or truncated)

        observation = next_observation
        current_episode_reward += reward
        total_steps += 1

        if terminated or truncated:
            break

    scores_window.append(current_episode_reward)
    all_episode_scores.append(current_episode_reward)
    current_avg_score = np.mean(scores_window)
    all_average_scores.append(current_avg_score)
    epsilon = max(EPSILON_END, EPSILON_DECAY * epsilon) # Decay epsilon

    print(f"Episode {i_episode}\tTotal Steps: {total_steps}\tRollouts: {agent.rollouts_processed_for_lr_decay}\tScore: {current_episode_reward:.2f}\tAvg Score (last 100): {current_avg_score:.2f}\tEpsilon: {epsilon:.4f}\tLR: {agent.optimizer.param_groups[0]['lr']:.2e}")

    if i_episode % SAVE_EVERY == 0:
        agent.save(MODEL_PATH)
        print(f"Model saved at episode {i_episode} to {MODEL_PATH}")

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(all_episode_scores, label='Episode Score', alpha=0.6)
        plt.plot(all_average_scores, label='Avg Score (Window 100)', color='red', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.title(f'Training Scores up to Episode {i_episode}')
        plt.legend()
        plt.grid(True)
        plot_save_path = f"{PLOT_PATH_PREFIX}.png"
        plt.savefig(plot_save_path)
        plt.close()
        print(f"Plot saved to {plot_save_path}")

# --- Cleanup ---
env.close()
agent.save(MODEL_PATH) # Save the final model
print("Training finished and model saved.")
print(f"To evaluate, you can load '{MODEL_PATH}' and set epsilon to a small value.")