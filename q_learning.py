import numpy as np
from collections import defaultdict, deque
import random

class QLearningAgent:
    def __init__(self, learning_rate=0.05, discount_factor=0.99, epsilon=0.1):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.initial_epsilon = epsilon
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        self.num_actions = 2
        self.q_table = defaultdict(lambda: np.zeros(self.num_actions))  # 4 possible actions
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 32
        
    def get_action(self, state, training=True):
        if np.all(self.q_table[state] == self.q_table[state][0]):
            return np.random.randint(0, self.num_actions)  # Random action
        elif training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.num_actions)  # Random action
        else:
            return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state):
        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])
        
        # Q-learning formula
        new_value = (1 - self.learning_rate) * old_value + \
                   self.learning_rate * (reward + self.discount_factor * next_max)
        
        self.q_table[state][action] = new_value
    
    def save_q_table(self, filename):
        # Convert defaultdict to regular dict for saving
        q_dict = dict(self.q_table)
        np.save(filename, q_dict)
    
    def load_q_table(self, filename):
        q_dict = np.load(filename, allow_pickle=True).item()
        self.q_table = defaultdict(lambda: np.zeros(self.num_actions), q_dict)
    
    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay) 
    
    def store_experience(self, state, action, reward, next_state):
        self.replay_buffer.append((state, action, reward, next_state))
    
    def learn_from_replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return
            
        batch = random.sample(self.replay_buffer, self.batch_size)
        for state, action, reward, next_state in batch:
            self.learn(state, action, reward, next_state) 