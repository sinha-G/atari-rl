import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

from .agent import Agent # Assuming agent.py is in the same directory

# --- Helper: Initialize weights ---
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCritic(nn.Module):
    def __init__(self, input_shape, action_size, fc_units=512):
        super(ActorCritic, self).__init__()
        channels, height, width = input_shape

        # Convolutional layers (shared)
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(channels, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            flattened_size = self.conv(dummy_input).shape[1]
        
        # Actor head
        self.actor_fc = layer_init(nn.Linear(flattened_size, fc_units))
        self.actor_output = layer_init(nn.Linear(fc_units, action_size), std=0.01)

        # Critic head
        self.critic_fc = layer_init(nn.Linear(flattened_size, fc_units))
        self.critic_output = layer_init(nn.Linear(fc_units, 1), std=1.0)

    def forward(self, state):
        """
        Passes state through the network.
        Returns action logits and state value.
        """
        x = self.conv(state)
        
        actor_hidden = torch.relu(self.actor_fc(x))
        action_logits = self.actor_output(actor_hidden)
        
        critic_hidden = torch.relu(self.critic_fc(x))
        state_value = self.critic_output(critic_hidden)
        
        return action_logits, state_value

    def get_action_and_log_prob(self, state, action=None):
        """
        Returns an action sampled from the policy, and its log probability.
        If action is provided, returns log_prob for that action.
        """
        action_logits, _ = self.forward(state)
        probs = Categorical(logits=action_logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action)

    def get_value(self, state):
        """
        Returns the state value estimated by the critic.
        """
        _, state_value = self.forward(state)
        return state_value


class PPOAgent(Agent):
    def __init__(self, state_space_shape, action_space_size,
                 lr=3e-4,          # Learning rate for the optimizer
                 gamma=0.99,       # Discount factor
                 gae_lambda=0.95,  # Lambda for GAE
                 clip_epsilon=0.2, # PPO clip parameter
                 n_steps=2048,     # Steps to collect per environment before update (rollout buffer size)
                 ppo_epochs=10,    # Number of epochs to update policy per rollout
                 mini_batch_size=64, # Mini-batch size for PPO updates
                 entropy_coeff=0.01, # Entropy coefficient for actor loss
                 value_loss_coeff=0.5, # Value loss coefficient
                 max_grad_norm=0.5, # Gradient clipping
                 fc_units=512,
                 anneal_lr_on_interval: bool = False,
                 lr_decay_factor: float = 0.9, # Factor to multiply LR by
                 lr_decay_rollouts: int = 100, # Number of rollouts (updates) before decaying LR
                 min_lr: float = 1e-6):       # Minimum learning rate):
        super().__init__(state_space_shape, action_space_size)

        self.original_state_shape = state_space_shape # (C, H, W)
        self.action_size = action_space_size
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.n_steps = n_steps 
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff
        self.max_grad_norm = max_grad_norm

        self.anneal_lr_on_interval = anneal_lr_on_interval
        self.lr_decay_factor = lr_decay_factor
        self.lr_decay_rollouts = lr_decay_rollouts
        self.min_lr = min_lr
        self.rollouts_processed_for_lr_decay = 0 # Counter for LR decay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor_critic = ActorCritic(self.original_state_shape, self.action_size, fc_units).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.lr, eps=1e-5)

        # Storage for rollouts
        self.memory = {
            'states': torch.zeros((self.n_steps, *self.original_state_shape), dtype=torch.float32, device=self.device),
            'actions': torch.zeros((self.n_steps,), dtype=torch.long, device=self.device),
            'log_probs_old': torch.zeros((self.n_steps,), dtype=torch.float32, device=self.device),
            'rewards': torch.zeros((self.n_steps,), dtype=torch.float32, device=self.device),
            'dones': torch.zeros((self.n_steps,), dtype=torch.float32, device=self.device),
            'values_old': torch.zeros((self.n_steps,), dtype=torch.float32, device=self.device)
        }
        self.current_step_in_rollout = 0

    def _preprocess_state(self, state):
        state_np = np.array(state, dtype=np.float32)
        state_tensor = torch.from_numpy(state_np).to(self.device)
        if state_tensor.ndim == 3: # (C, H, W)
            state_tensor = state_tensor.unsqueeze(0) # (1, C, H, W)
        return state_tensor

    def act(self, state, eps=0.):
        state_tensor = self._preprocess_state(state)
        self.actor_critic.eval()
        with torch.no_grad():
            action, _ = self.actor_critic.get_action_and_log_prob(state_tensor)
        self.actor_critic.train()
        return action.item()

    def learn(self, state, action, reward, next_state, done):
        state_np = np.array(state, dtype=np.float32) # Should be (C,H,W)
        
        state_tensor_for_storage = self._preprocess_state(state_np) # (1,C,H,W)
        action_tensor = torch.tensor([action], dtype=torch.long, device=self.device)

        self.actor_critic.eval()
        with torch.no_grad():
            _, log_prob_old = self.actor_critic.get_action_and_log_prob(state_tensor_for_storage, action=action_tensor)
            value_old = self.actor_critic.get_value(state_tensor_for_storage)
        self.actor_critic.train()

        self.memory['states'][self.current_step_in_rollout] = torch.from_numpy(state_np).to(self.device) # Store (C,H,W)
        self.memory['actions'][self.current_step_in_rollout] = action_tensor.squeeze()
        self.memory['log_probs_old'][self.current_step_in_rollout] = log_prob_old.squeeze()
        self.memory['rewards'][self.current_step_in_rollout] = torch.tensor(reward, dtype=torch.float32).to(self.device)
        self.memory['dones'][self.current_step_in_rollout] = torch.tensor(done, dtype=torch.float32).to(self.device)
        self.memory['values_old'][self.current_step_in_rollout] = value_old.squeeze()

        self.current_step_in_rollout += 1

        if self.current_step_in_rollout == self.n_steps:
            self._update(next_state, done)
            self.current_step_in_rollout = 0

    def _compute_gae_and_returns(self, last_next_state, last_done):
        advantages = torch.zeros_like(self.memory['rewards']).to(self.device)
        last_gae_lam = 0

        with torch.no_grad():
            if last_done:
                next_value_final = torch.tensor(0.0).to(self.device)
            else:
                next_state_final_tensor = self._preprocess_state(last_next_state)
                next_value_final = self.actor_critic.get_value(next_state_final_tensor).reshape(1)

        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_values_t = next_value_final
            else:
                next_values_t = self.memory['values_old'][t+1]
            
            current_done_t = self.memory['dones'][t]
            delta = self.memory['rewards'][t] + self.gamma * next_values_t * (1.0 - current_done_t) - self.memory['values_old'][t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * (1.0 - current_done_t) * last_gae_lam
        
        returns = advantages + self.memory['values_old']
        return advantages, returns

    def _update(self, last_next_state, last_done):
        # Annealing
        if self.anneal_lr_on_interval:
            self.rollouts_processed_for_lr_decay += 1
            if self.rollouts_processed_for_lr_decay % self.lr_decay_rollouts == 0:
                old_lr = self.optimizer.param_groups[0]['lr']
                new_lr = max(old_lr * self.lr_decay_factor, self.min_lr)
                if new_lr < old_lr:
                    self.optimizer.param_groups[0]['lr'] = new_lr
                    print(f"LR decayed at rollout {self.rollouts_processed_for_lr_decay}: {old_lr:.2e} -> {new_lr:.2e}")
                elif old_lr <= self.min_lr:
                     print(f"LR already at/below minimum {self.min_lr:.2e} at rollout {self.rollouts_processed_for_lr_decay}.")

        advantages, returns_target = self._compute_gae_and_returns(last_next_state, last_done)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Flatten batch for mini-batch processing
        b_states = self.memory['states'] # (n_steps, C, H, W)
        b_actions = self.memory['actions'] # (n_steps,)
        b_log_probs_old = self.memory['log_probs_old'] # (n_steps,)
        b_values_old = self.memory['values_old'] # (n_steps,) - not directly used in loss, but for GAE
        b_advantages = advantages # (n_steps,)
        b_returns_target = returns_target # (n_steps,) - these are V_targets

        self.actor_critic.train()
        indices = np.arange(self.n_steps)

        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, self.n_steps, self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_indices = indices[start:end]

                mb_states = b_states[mb_indices]
                mb_actions = b_actions[mb_indices]
                mb_log_probs_old = b_log_probs_old[mb_indices]
                mb_advantages = b_advantages[mb_indices]
                mb_returns_target = b_returns_target[mb_indices]

                # Get new log_probs, values, and entropy from current policy
                new_logits, new_values = self.actor_critic(mb_states)
                new_values = new_values.squeeze(-1) # Shape: (mini_batch_size)
                
                dist = Categorical(logits=new_logits)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                # Ratio (pi_theta / pi_theta_old)
                log_ratio = new_log_probs - mb_log_probs_old
                ratio = torch.exp(log_ratio)

                # Clipped surrogate objective
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value loss (critic loss)
                value_loss = nn.MSELoss()(new_values, mb_returns_target)
                
                # Total loss
                loss = actor_loss + self.value_loss_coeff * value_loss - self.entropy_coeff * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

    def save(self, path):
        torch.save(self.actor_critic.state_dict(), path)
        print(f"PPO model saved to {path}")

    def load(self, path):
        self.actor_critic.load_state_dict(torch.load(path, map_location=self.device))
        self.actor_critic.eval()
        print(f"PPO model loaded from {path}")
