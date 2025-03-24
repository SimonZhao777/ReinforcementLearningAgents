import pygame
from game import CatchGame
from q_learning import QLearningAgent
import numpy as np

def train_agent(episodes=1000, render=False):
    # Initialize game and agent
    game = CatchGame()
    agent = QLearningAgent()
    
    # Training loop
    for episode in range(episodes):
        state = game.reset()
        total_reward = 0
        done = False
        
        while not done:
            if render:
                game.render()
                pygame.time.wait(5)  # Slow down rendering for visualization
            
            # Get action from agent
            action = agent.get_action(state)
            
            # Take action and get next state
            next_state, reward, done = game.step(action)
            total_reward += reward
            
            # Store experience and learn from replay
            # agent.learn(state, action, reward, next_state)
            agent.store_experience(state, action, reward, next_state)
            agent.learn_from_replay()
            
            state = next_state
        
        # Decay epsilon after each episode
        agent.decay_epsilon()
        
        if episode % 100 == 0:
            print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")
    
    # Save the trained Q-table
    agent.save_q_table('q_table.npy')
    game.close()

def play_game(episodes=5):
    # Load the trained agent
    game = CatchGame()
    agent = QLearningAgent()
    agent.load_q_table('q_table.npy')
    agent.epsilon = 0  # No exploration during play
    
    for episode in range(episodes):
        state = game.reset()
        total_reward = 0
        done = False
        
        while not done:
            game.render()
            pygame.time.wait(10)  # Slow down for visualization
            
            action = agent.get_action(state, training=False)
            state, reward, done = game.step(action)
            total_reward += reward
        
        print(f"Play Episode: {episode + 1}, Total Reward: {total_reward}")
    
    game.close()

if __name__ == "__main__":
    # Train the agent
    print("Training the agent...")
    train_agent(episodes=10000, render=False)
    
    # Play with the trained agent
    print("\nPlaying with the trained agent...")
    play_game(episodes=15)