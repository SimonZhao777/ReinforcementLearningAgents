import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

class DQNAgent:
    def __init__(self, state_size, action_size, frame_stack=4, learning_rate=0.0005, discount_factor=0.99, epsilon=1.0):
        self.state_size = state_size  # 图像大小 (84, 84, 3)
        self.action_size = action_size
        self.frame_stack = frame_stack  # 使用连续的帧数
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = 0.999
        self.min_epsilon = 0.01
        self.batch_size = 128
        self.memory = deque(maxlen=10000)

        # 检查 GPU 是否可用
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 构建 CNN 模型
        self.model = self._build_cnn_model().to(self.device)
        self.target_model = self._build_cnn_model().to(self.device)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Loss tracking
        self.losses = []

    def _build_cnn_model(self):
        """构建卷积神经网络"""
        model = nn.Sequential(
            nn.Conv2d(self.frame_stack * 3, 32, kernel_size=8, stride=4),  # 输入通道数为 frame_stack * 3
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),  # 根据卷积输出调整线性层输入大小
            nn.ReLU(),
            nn.Linear(512, self.action_size)
        )
        return model

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def preprocess_state(self, state_frames):
        """将多帧图像堆叠为单个张量"""
        # 将多帧图像堆叠在一起 (frame_stack * 3 通道)
        stacked_frames = np.concatenate(state_frames, axis=2)  # Shape: (84, 84, frame_stack * 3)

        # 转换为 PyTorch 格式 (batch_size, channels, height, width)
        state_tensor = torch.FloatTensor(np.transpose(stacked_frames, (2, 0, 1))).unsqueeze(0).to(self.device)
        return state_tensor

    def get_action(self, state, training=True):
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Explore
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values, dim=1).item()  # Exploit

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn_from_replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.cat([experience[0] for experience in minibatch]).to(self.device)
        actions = torch.LongTensor([experience[1] for experience in minibatch]).to(self.device)
        rewards = torch.FloatTensor([experience[2] for experience in minibatch]).to(self.device)
        next_states = torch.cat([experience[3] for experience in minibatch]).to(self.device)
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
            return 0
        avg_loss = sum(self.losses) / len(self.losses)
        self.losses.clear()  # 清空损失记录
        return avg_loss