from collections import deque

import pygame
from game import CatchGame
from cnn_deep_q_learning import DQNAgent
from q_learning import QLearningAgent
import numpy as np

def train_agent(episodes=1000, update_target_model_steps=200000, decay_epsilon_steps=1000, use_dqn=True, action_size=3, frame_stack=4):
    game = CatchGame()
    # frame_stack = 4  # 使用 4 帧作为一个状态

    if use_dqn:
        state_size = (84, 84, 3)  # RGB 图像大小
        agent = DQNAgent(state_size, action_size, frame_stack=frame_stack)

        # 初始化帧队列
        state_frames = deque([np.zeros((84, 84, 3)) for _ in range(frame_stack)], maxlen=frame_stack)
        state_frames.append(game.get_screen())  # 添加第一帧
        state = agent.preprocess_state(state_frames)
    else:
        agent = QLearningAgent()

    # 训练循环
    total_steps = 0
    for episode in range(episodes):
        game.reset()
        game.render()
        if use_dqn:
            state_frames.clear()
            state_frames.extend([np.zeros((84, 84, 3)) for _ in range(frame_stack)])
            state_frames.append(game.get_screen())
            state = agent.preprocess_state(state_frames)
        else:
            state = game.reset()

        total_reward = 0
        done = False
        episode_steps = 0

        while not done:
            game.render()
            # pygame.time.wait(5)

            # 获取动作
            action = agent.get_action(state, training=True)

            # 执行动作并获取下一状态
            reward, done = game.step(action)[1:]  # 先执行动作再获取最新的state
            # # --- Reward Clipping ---
            # reward = np.clip(reward, -1.0, 1.0)
            reward = reward / 200.0

            game.render()  # 需要先render才能get_screen
            next_screen = game.get_screen()
            state_frames.append(next_screen)
            next_state = agent.preprocess_state(state_frames)

            # 存储经验和学习
            if use_dqn:
                agent.store_experience(state, action, reward, next_state, done)
                agent.learn_from_replay()
            else:
                agent.store_experience(state, action, reward, next_state)
                agent.learn_from_replay()

            state = next_state
            total_reward += reward
            total_steps += 1
            episode_steps += 1

            # Optional: Add a max steps per episode check
            # if episode_steps > max_steps_per_episode:
            #     done = True

            if total_steps % decay_epsilon_steps == 0:  # decay epsilon for every 100 steps
                agent.decay_epsilon()

            # Update target model every C episodes
            if total_steps % update_target_model_steps == 0:
                agent.update_target_model()
                agent.save_model('cnn_dqn_model.pth')

        if episode % 10 == 0:
            if use_dqn:
                print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}, Average Loss: {agent.get_avg_loss():.4f}")
            else:
                print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")



    if use_dqn:
        agent.save_model('cnn_dqn_model.pth')
    else:
        agent.save_q_table('q_table.npy')
    game.close()

def play_game(episodes=5, use_dqn=True, action_size=3, frame_stack=4):
    game = CatchGame()
    # frame_stack = 4  # 使用 4 帧作为一个状态

    if use_dqn:
        state_size = (84, 84, 3)  # RGB 图像大小
        agent = DQNAgent(state_size, action_size, frame_stack=frame_stack)
        agent.load_model('cnn_dqn_model.pth')

        # 初始化帧队列
        state_frames = deque([np.zeros((84, 84, 3)) for _ in range(frame_stack)], maxlen=frame_stack)
        state_frames.append(game.get_screen())  # 添加第一帧
        state = agent.preprocess_state(state_frames)
    else:
        agent = QLearningAgent()
        agent.load_q_table('q_table.npy')
        agent.epsilon = 0  # No exploration during play

    for episode in range(episodes):
        if use_dqn:
            state_frames.clear()
            state_frames.extend([np.zeros((84, 84, 3)) for _ in range(frame_stack)])
            state_frames.append(game.get_screen())
            state = agent.preprocess_state(state_frames)
        else:
            state = game.reset()

        total_reward = 0
        done = False

        while not done:
            game.render()
            pygame.time.wait(10)

            # 获取动作
            action = agent.get_action(state, training=False)

            # 执行动作并获取下一状态
            reward, done = game.step(action)[1:]  # 先执行动作再获取最新的state
            game.render()  # 需要先render才能get_screen
            next_screen = game.get_screen()
            state_frames.append(next_screen)
            next_state = agent.preprocess_state(state_frames)

            state = next_state
            total_reward += reward

        print(f"Play Episode: {episode + 1}, Total Reward: {total_reward}")

    game.close()

if __name__ == "__main__":
    USE_DQN = True  # 设置为 False 使用 Q-learning
    action_size = 3  # Set action space to 3 (0: stay still, 1: move left, 2: move right)
    frame_stack = 4  # 使用 4 帧作为一个状态

    print("Training the agent...")
    train_agent(episodes=20000, update_target_model_steps=200000, decay_epsilon_steps=1000, use_dqn=USE_DQN, action_size=action_size, frame_stack=frame_stack)

    print("\nPlaying with the trained agent...")
    play_game(episodes=15, use_dqn=USE_DQN, action_size=action_size, frame_stack=frame_stack)