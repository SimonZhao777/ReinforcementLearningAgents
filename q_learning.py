import numpy as np
from collections import defaultdict

class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(4))  # 4 possible actions
        
    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 4)  # Random action
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
        self.q_table = defaultdict(lambda: np.zeros(4), q_dict) 