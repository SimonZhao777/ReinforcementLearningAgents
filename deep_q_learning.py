import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.99, epsilon=1.0):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        self.batch_size = 32
        self.memory = deque(maxlen=10000)

        # Build the neural network models
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

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
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.model(state_tensor)
        return torch.argmax(q_values, dim=1).item()  # Exploit

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn_from_replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([experience[0] for experience in minibatch])
        actions = torch.LongTensor([experience[1] for experience in minibatch])
        rewards = torch.FloatTensor([experience[2] for experience in minibatch])
        next_states = torch.FloatTensor([experience[3] for experience in minibatch])
        dones = torch.FloatTensor([experience[4] for experience in minibatch])

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

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename))
        self.update_target_model()