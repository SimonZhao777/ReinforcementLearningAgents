import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.0005, discount_factor=0.99, epsilon=1.0):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = 0.999
        self.min_epsilon = 0.01
        self.batch_size = 128
        self.memory = deque(maxlen=10000)

        # Check if GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Build the neural network models
        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Loss tracking
        self.losses = []

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
        return model

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state, training=True):
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Explore
        # Exploit during training or evaluation - gradients not needed for action selection
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
        return torch.argmax(q_values, dim=1).item()  # Exploit

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn_from_replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([experience[0] for experience in minibatch]).to(self.device)
        actions = torch.LongTensor([experience[1] for experience in minibatch]).to(self.device)
        rewards = torch.FloatTensor([experience[2] for experience in minibatch]).to(self.device)
        next_states = torch.FloatTensor([experience[3] for experience in minibatch]).to(self.device)
        dones = torch.FloatTensor([experience[4] for experience in minibatch]).to(self.device)

        # Get Q-values for the next states using the target model
        with torch.no_grad():
            next_q_values = self.target_model(next_states)
            target_q_values = rewards + (1 - dones) * self.discount_factor * next_q_values.max(1)[0]

        # Get current Q-values and update them
        current_q_values = self.model(states)
        action_indices = actions.unsqueeze(1)
        q_values_for_actions = current_q_values.gather(1, action_indices).squeeze()

        # Compute loss and backpropagate
        loss = nn.MSELoss()(q_values_for_actions, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Store the loss value
        self.losses.append(loss.item())

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename, map_location=self.device))
        self.update_target_model()

    def get_avg_loss(self):
        if len(self.losses) == 0:
            return 0.0
        avg_loss = sum(self.losses) / len(self.losses)
        # print(f"Episode: {episode}, Average Loss: {avg_loss:.4f}")
        self.losses.clear()  # Clear the losses after printing
        return avg_loss