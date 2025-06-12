# Atari RL Project

Exploring Reinforcement Learning (RL) using environments from Atari games. This project is built with [Gymnasium](https://gymnasium.farama.org/) and [PyTorch](https://pytorch.org/).

## Implemented Agents

Currently, the following RL agents are implemented:

*   **Proximal Policy Optimization (PPO)**: Implemented in [`agents/ppo.py`](agents/ppo.py). This is an on-policy actor-critic agent that uses a clipped surrogate objective and Generalized Advantage Estimation (GAE).
*   **Deep Q-Network (DQN)**: Implemented in [`agents/dqn.py`](agents/dqn.py). This is an off-policy agent that uses a Q-network to estimate action values and a replay buffer for experience replay.

## Features

*   **Environment Preprocessing**: Uses `AtariPreprocessing` from Gymnasium for standard Atari game preprocessing, including screen resizing, grayscale conversion, frame skipping, and observation scaling. See [`train.py`](train.py).
*   **Frame Stacking**: Employs `FrameStackObservation` to stack consecutive frames, providing the agent with information about motion. See [`train.py`](train.py).
*   **Training Loop**: A flexible training script (`train.py`) allows for training agents on various Atari environments.
*   **Model Saving/Loading**: Agents can save their learned models (e.g., to the `models/` directory) and load them for continued training or evaluation. See [`DQNAgent.save()`](agents/dqn.py), [`DQNAgent.load()`](agents/dqn.py), [`PPOAgent.save()`](agents/ppo.py), and [`PPOAgent.load()`](agents/ppo.py).
*   **Performance Plotting**: Training progress (episode scores and EMA scores) is plotted and saved to the `plots/` directory during training. See [`train.py`](train.py).
*   **Learning Rate Annealing**: The PPO agent supports learning rate annealing based on a decay factor over a specified number of rollouts. See [`PPOAgent`](agents/ppo.py) and hyperparameters in [`train.py`](train.py).

## Project Structure

```
.
├── agents/             # Contains agent implementations (ppo.py, dqn.py)
├── models/             # Stores trained model checkpoints (e.g., .pth files)
├── plots/              # Stores generated plots of training progress
├── train.py            # Main script for training agents
├── README.md           # This file
└── .gitignore
```

## Getting Started

1.  **Install Dependencies**: Ensure you have Python, Gymnasium, PyTorch, ale-py, Matplotlib, and tqdm installed.
    ```bash
    pip install gymnasium[atari] ale-py torch matplotlib tqdm
    ```
2.  **Configure Training**: Modify hyperparameters in [`train.py`](train.py) as needed, such as `ENV_NAME`, `NUM_EPISODES`, agent-specific parameters, and LR annealing settings.
3.  **Run Training**:
    ```bash
    python train.py
    ```
    Trained models will be saved in the `models/` directory (e.g., `models/breakout_ppo.pth`), and plots will be saved in the `plots/` directory.
    