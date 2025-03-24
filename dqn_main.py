import pygame
from game import CatchGame
from q_learning import QLearningAgent
from deep_q_learning import DQNAgent
import numpy as np

def train_agent(episodes=1000, update_target_model_episodes=100, render=False, use_dqn=False):
    # Initialize game
    game = CatchGame()

    # Initialize agent based on user choice
    if use_dqn:
        state_size = len(game.reset())
        action_size = 2  # Left and Right
        agent = DQNAgent(state_size, action_size)
    else:
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
            action = agent.get_action(state, training=True)

            # Take action and get next state
            next_state, reward, done = game.step(action)
            total_reward += reward

            # Store experience and learn from replay
            if use_dqn:
                agent.store_experience(state, action, reward, next_state, done)
                agent.learn_from_replay()
            else:
                agent.store_experience(state, action, reward, next_state)
                agent.learn_from_replay()

            state = next_state

        # Decay epsilon after each episode
        agent.decay_epsilon()

        # Print total reward and loss every 100 episodes
        if episode % 100 == 0:
            if use_dqn:
                print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}, Average Loss: {agent.get_avg_loss():.4f}")
            else:
                print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")

        # Update target model every C episodes
        if episode % update_target_model_episodes == 0:
            agent.update_target_model()
            agent.save_model('dqn_model.pth')

    # Save the trained model or Q-table
    if use_dqn:
        agent.save_model('dqn_model.pth')
    else:
        agent.save_q_table('q_table.npy')
    game.close()

def play_game(episodes=5, use_dqn=False):
    # Load the trained agent
    game = CatchGame()

    if use_dqn:
        state_size = len(game.reset())
        action_size = 4  # Left and Right
        agent = DQNAgent(state_size, action_size)
        agent.load_model('dqn_model.pth')
    else:
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
    # Choose between Q-learning and DQN
    USE_DQN = True  # Set to False to use Q-learning instead

    # Train the agent
    print("Training the agent...")
    train_agent(episodes=10000, update_target_model_episodes=100, render=False, use_dqn=USE_DQN)

    # Play with the trained agent
    print("\nPlaying with the trained agent...")
    play_game(episodes=15, use_dqn=USE_DQN)