import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import string
from collections import deque
import metrics
from data_wrappers.code import CodeSearchNetDataset


class KeyboardEnvironment:
    """Environment for keyboard layout optimization."""

    def __init__(self, dataset):
        self.symbols = list(string.ascii_lowercase) + [';', ',', '.', '/']
        self.num_positions = 30  # 3 rows × 10 positions
        self.dataset = dataset
        self.reset()

    def reset(self):
        """Reset environment to initial state."""
        self.current_layout = [[''] * 10 for _ in range(3)]
        self.available_symbols = self.symbols.copy()
        self.placed_symbols = []
        self.current_position = 0
        return self.get_state()

    def get_state(self):
        """
        Get current state representation.
        Returns a feature vector containing:
        - Current partial layout encoding
        - Remaining symbols to place
        - Position information
        """
        flat_layout = []
        for row in self.current_layout:
            for pos in row:
                if pos == '':
                    flat_layout.append(0)
                else:
                    flat_layout.append(self.symbols.index(pos) + 1)

        available_mask = [1 if s in self.available_symbols else 0 for s in self.symbols]

        row = self.current_position // 10
        col = self.current_position % 10
        position_features = [row / 2.0, col / 9.0, self.current_position / 29.0]

        state = flat_layout + available_mask + position_features
        return np.array(state, dtype=np.float32)

    def step(self, action):
        """
        Place a symbol at the current position.
        Action is the index of the symbol to place.
        """
        if action >= len(self.available_symbols):
            # Invalid action - return large penalty
            return self.get_state(), -10.0, False, {}

        symbol = self.available_symbols[action]
        row = self.current_position // 10
        col = self.current_position % 10

        self.current_layout[row][col] = symbol
        self.placed_symbols.append(symbol)
        self.available_symbols.remove(symbol)
        self.current_position += 1

        done = self.current_position >= 30 or len(self.available_symbols) == 0

        # Calculate reward
        if done:
            score = metrics.evaluate_keyboard(self.current_layout, self.dataset)['Composite Score']
            reward = -score * 100  # Scale for better gradients
        else:
            reward = self.calculate_intermediate_reward(symbol, row, col)

        return self.get_state(), reward, done, {'layout': self.current_layout}

    def calculate_intermediate_reward(self, symbol, row, col):
        reward = 0

        common_letters = 'etaoinshrdlu'
        if symbol in common_letters and row == 1:
            reward += 0.1

        if symbol in 'qxzj' and row == 1:
            reward -= 0.1

        if symbol in 'aeiou' and (row == 1 or (row in [0, 2] and 3 <= col <= 6)):
            reward += 0.05

        return reward


class PolicyNetwork(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)

    def forward(self, state):
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))

        action_probs = F.softmax(self.actor(x), dim=-1)

        state_value = self.critic(x)

        return action_probs, state_value


class PPOMemory:

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def add(self, state, action, reward, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def get_batch(self):
        return (
            torch.tensor(np.array(self.states), dtype=torch.float32),
            torch.tensor(self.actions, dtype=torch.long),
            torch.tensor(self.rewards, dtype=torch.float32),
            torch.tensor(self.log_probs, dtype=torch.float32),
            torch.tensor(self.values, dtype=torch.float32),
            torch.tensor(self.dones, dtype=torch.float32)
        )


class ProximalPolicyOptimization:
    def __init__(self,
                 state_dim,
                 action_dim,
                 lr=3e-4,
                 gamma=0.99,
                 gae_lambda=0.95,
                 clip_epsilon=0.2,
                 c1=0.5,  # Value loss coefficient
                 c2=0.01,  # Entropy bonus
                 epochs_per_update=10,
                 batch_size=64):

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.c1 = c1
        self.c2 = c2
        self.epochs_per_update = epochs_per_update
        self.batch_size = batch_size

        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Memory buffer
        self.memory = PPOMemory()

        self.best_layout = None
        self.best_score = float('inf')

    def select_action(self, state, training=True):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            action_probs, value = self.policy(state_tensor)

        # mask invalid actions (already placed symbols)
        # this is handled by the environment returning valid actions only

        if training:
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        else:
            action = torch.argmax(action_probs)
            log_prob = torch.log(action_probs[0, action])

        return action.item(), log_prob.item(), value.item()

    def compute_returns_and_advantages(self, rewards, values, dones):
        returns = []
        advantages = []

        rewards = rewards.numpy()
        values = values.numpy()
        dones = dones.numpy()

        next_value = 0
        next_advantage = 0

        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0
                next_advantage = 0

            td_error = rewards[t] + self.gamma * next_value - values[t]

            # GAE advantage
            advantage = td_error + self.gamma * self.gae_lambda * next_advantage
            next_advantage = advantage

            return_ = rewards[t] + self.gamma * next_value
            next_value = values[t]

            advantages.insert(0, advantage)
            returns.insert(0, return_)

        return torch.tensor(returns, dtype=torch.float32), torch.tensor(advantages, dtype=torch.float32)

    def update(self):
        states, actions, rewards, old_log_probs, old_values, dones = self.memory.get_batch()

        returns, advantages = self.compute_returns_and_advantages(rewards, old_values, dones)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # advnorm

        for _ in range(self.epochs_per_update):
            indices = np.arange(len(states))
            np.random.shuffle(indices)

            for start in range(0, len(states), self.batch_size):
                end = min(start + self.batch_size, len(states))
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]

                action_probs, values = self.policy(batch_states)
                dist = Categorical(action_probs)

                new_log_probs = dist.log_prob(batch_actions)

                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                values = values.squeeze()
                value_loss = F.mse_loss(values, batch_returns)

                entropy = dist.entropy().mean()

                loss = actor_loss + self.c1 * value_loss - self.c2 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

        self.memory.clear()

    def train(self, env, num_episodes=1000, update_frequency=10):
        """Train PPO agent."""
        episode_rewards = deque(maxlen=100)
        episode_scores = deque(maxlen=100)

        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0

            while True:
                action, log_prob, value = self.select_action(state, training=True)

                next_state, reward, done, info = env.step(action)

                self.memory.add(state, action, reward, log_prob, value, done)

                episode_reward += reward
                state = next_state

                if done:
                    layout = info['layout']
                    score = metrics.evaluate_keyboard(layout, env.dataset)['Composite Score']

                    if score < self.best_score:
                        self.best_score = score
                        self.best_layout = layout

                    episode_rewards.append(episode_reward)
                    episode_scores.append(score)
                    break

            if (episode + 1) % update_frequency == 0:
                self.update()

            # Logging
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards) if episode_rewards else 0
                avg_score = np.mean(episode_scores) if episode_scores else 0
                print(f"Episode {episode + 1}: Avg Reward: {avg_reward:.2f}, "
                      f"Avg Score: {avg_score:.4f}, Best Score: {self.best_score:.4f}")

        return self.best_layout, self.best_score


def main():
    print("Loading dataset...")
    dataset = CodeSearchNetDataset(language="python", split="train")[:500]
    print("Dataset loaded.")

    env = KeyboardEnvironment(dataset)

    state_dim = 30 + 30 + 3  # Layout encoding + available symbols + position features
    action_dim = 30  # Maximum possible symbols

    ppo = ProximalPolicyOptimization(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        c1=0.5,
        c2=0.01,
        epochs_per_update=10,
        batch_size=32
    )

    # Train
    print("\nStarting PPO training...")
    best_layout, best_score = ppo.train(
        env,
        num_episodes=200,
        update_frequency=10
    )

    print("\nBest layout found by PPO:")
    for row in best_layout:
        print(row)
    print(f"Score: {best_score:.4f}")

    print("\nMetrics for best layout:")
    detailed_scores = metrics.evaluate_keyboard(best_layout, dataset)
    for metric, score in detailed_scores.items():
        print(f"  {metric}: {score:.4f}")


if __name__ == "__main__":
    main()