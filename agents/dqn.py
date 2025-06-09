import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque, namedtuple

from .agent import Agent

class QNetwork(nn.Module):
    def __init__(self, input_shape, action_size, fc_units=512):
        """
        Initialize parameters and build model.
        Args:
            input_shape (tuple): Shape of the input state (C, H, W).
            action_size (int): Dimension of each action.
            fc_units (int): Number of nodes in the fully connected hidden layer.
        """
        super(QNetwork, self).__init__()
        channels, height, width = input_shape

        # Convolutional layers
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Linear layers
        self.fc1 = nn.LazyLinear(fc_units)
        self.fc2 = nn.Linear(fc_units, action_size)

    def forward(self, state):
        """
        Build a network that maps state -> action values.
        Args:
            state (torch.Tensor): The input state (batch_size, C, H, W).
        Returns:
            torch.Tensor: The Q-values for each action.
        """
        x = torch.relu(self.conv1(state))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Define the Replay Buffer
Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        # Ensure states are concrete numpy arrays (float32 due to scale_obs=True)
        # This converts LazyFrames from FrameStack to np.array
        state_np = np.array(state, dtype=np.float32)
        next_state_np = np.array(next_state, dtype=np.float32)
        e = Experience(state_np, action, reward, next_state_np, done)
        self.memory.append(e)

    def sample(self, batch_size):
        return random.sample(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)

class DQNAgent(Agent):
    def __init__(self, state_space_shape, action_space_size,
                 buffer_size=10000, batch_size=64, gamma=0.99,
                 lr=1e-4, tau=1e-3, update_every=4):
        super().__init__(state_space_shape, action_space_size)

        # state_space_shape is expected to be (C, H, W) from the wrapped environment
        self.original_state_shape = state_space_shape
        
        if len(self.original_state_shape) != 3:
            raise ValueError(
                f"Expected state_space_shape to be (C, H, W) representing (Channels, Height, Width), "
                f"but got {self.original_state_shape}"
            )

        # Input shape for QNetwork is directly (C, H, W)
        q_network_input_shape = self.original_state_shape 

        self.action_size = action_space_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.update_every = update_every

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Q-Network
        self.qnetwork_local = QNetwork(q_network_input_shape, self.action_size).to(self.device)
        self.qnetwork_target = QNetwork(q_network_input_shape, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        # Replay memory
        self.memory = ReplayBuffer(self.buffer_size)
        self.t_step = 0

    def _preprocess_state(self, state):
        """
        Converts state (typically (C,H,W) numpy array/LazyFrames from FrameStack) 
        to a preprocessed tensor (1, C, H, W).
        Assumes state is float32 and normalized due to wrappers.
        """
        # state is LazyFrames or np.ndarray from FrameStack, shape (C,H,W)
        state_np = np.array(state, dtype=np.float32) # Ensure numpy array, float32

        # state_np is (C, H, W)
        state_tensor = torch.from_numpy(state_np).to(self.device)
        
        # Add batch dimension if it's not there yet
        if state_tensor.ndim == 3: # (C, H, W)
            state_tensor = state_tensor.unsqueeze(0) # (1, C, H, W)
        elif state_tensor.ndim == 4 and state_tensor.shape[0] == 1: # Already (1,C,H,W)
            pass 
        else:
            raise ValueError(f"Unexpected state tensor dimension: {state_tensor.shape}. Expected (C,H,W) or (1,C,H,W)")
        
        # Normalization (e.g. / 255.0) is done by wrappers (AtariPreprocessing with scale_obs=True)
        return state_tensor

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Args:
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state_tensor = self._preprocess_state(state)
        self.qnetwork_local.eval() # Set network to evaluation mode
        with torch.no_grad():
            action_values = self.qnetwork_local(state_tensor)
        self.qnetwork_local.train() # Set network back to train mode

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return np.random.choice(self.action_size)

    def learn(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(self.batch_size)
                self._learn_from_experiences(experiences)

    def _learn_from_experiences(self, experiences):
        states, actions, rewards, next_states_raw, dones = zip(*experiences)

        states_tensor = torch.from_numpy(np.array(states, dtype=np.float32)).to(self.device)
        next_states_tensor = torch.from_numpy(np.array(next_states_raw, dtype=np.float32)).to(self.device)

        actions_tensor = torch.from_numpy(np.vstack(actions)).long().to(self.device)
        rewards_tensor = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
        dones_tensor = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(self.device)

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states_tensor).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards_tensor + (self.gamma * Q_targets_next * (1 - dones_tensor))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states_tensor).gather(1, actions_tensor)

        # Compute loss
        loss = nn.MSELoss()(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def _soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def save(self, path):
        """Saves the local Q-network weights."""
        torch.save(self.qnetwork_local.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path):
        """Loads weights into the local Q-network."""
        self.qnetwork_local.load_state_dict(torch.load(path, map_location=self.device))
        self.qnetwork_target.load_state_dict(torch.load(path, map_location=self.device)) # Also load to target
        self.qnetwork_local.eval() # Set to eval mode after loading
        self.qnetwork_target.eval()
        print(f"Model loaded from {path}")
