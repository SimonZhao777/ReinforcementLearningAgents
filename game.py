import pygame
import numpy as np
import random
import cv2

class CatchGame:
    def __init__(self, width=800, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Catch the Target")
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 255, 0)
        
        # Game parameters
        self.player_size = 20
        self.target_size = 20
        self.obstacle_size = 20
        self.player_speed = 10
        self.target_speed = 6
        self.obstacle_speed = 8
        self.grid_size = 20
        
        self.reset()
    
    def reset(self):
        # Initialize player position (bottom center)
        self.player_x = self.width // 2
        self.player_y = self.height - 50
        
        # Initialize target position (top)
        self.target_x = random.randint(0, self.width - self.target_size)
        self.target_y = 50
        
        # Initialize obstacles
        self.obstacles = []
        for _ in range(3):
            self.obstacles.append({
                'x': random.randint(0, self.width - self.obstacle_size),
                'y': random.randint(100, self.height - 100)
            })
        
        return self.get_state()
    
    def get_state(self):
        # Discretize the state space for Q-learning
        # State includes: player_x, target_x, target_y, and positions of obstacles
        state = [
            self.player_x // self.grid_size,  # Discretize player x position
            self.player_y // self.grid_size,  # Discretize player x position
            # (self.player_x - self.target_x) // self.grid_size,  # Discretize target x position
            self.target_x // self.grid_size,  # Discretize target x position
            self.target_y // self.grid_size,  # Discretize target y position
        ]
        # Add obstacle positions
        for obs in self.obstacles:
            # state.extend([(self.player_x - obs['x']) // self.grid_size, obs['y'] // self.grid_size])
            state.extend([obs['x'] // self.grid_size, obs['y'] // self.grid_size])
        return tuple(state)
    
    def step(self, action):
        # Actions: 0 (left), 1 (right), 2 (up), 3 (down)
        reward = 0
        done = False
        
        # Move player based on action
        if action == 0:  # Left
            self.player_x = max(0, self.player_x - self.player_speed)
        elif action == 1:  # Right
            self.player_x = min(self.width - self.player_size, self.player_x + self.player_speed)
        elif action == 2:  # Up
            self.player_y = max(0, self.player_y - self.player_speed)
        elif action == 3:  # Down
            self.player_y = min(self.height - self.player_size, self.player_y + self.player_speed)
        
        # Move target
        self.target_y += self.target_speed
        
        # Move obstacles
        for obs in self.obstacles:
            obs['y'] += self.obstacle_speed
            if obs['y'] > self.height:
                obs['y'] = 0
                obs['x'] = random.randint(0, self.width - self.obstacle_size)
        
        # Check for collisions
        if self.check_collision(self.player_x, self.player_y, self.player_size,
                             self.target_x, self.target_y, self.target_size):
            reward = 200
            done = True
            self.reset()
        
        # Check for obstacle collisions
        for obs in self.obstacles:
            if self.check_collision(self.player_x, self.player_y, self.player_size,
                                 obs['x'], obs['y'], self.obstacle_size):
                reward = -50
                done = True
                self.reset()
                break
        
        # Check if target is missed
        if self.target_y > self.height:
            reward = -10
            self.target_y = 50
            self.target_x = random.randint(0, self.width - self.target_size)
        
        return self.get_state(), reward, done
    
    def check_collision(self, x1, y1, size1, x2, y2, size2):
        return (x1 < x2 + size2 and
                x1 + size1 > x2 and
                y1 < y2 + size2 and
                y1 + size1 > y2)
    
    def render(self):
        self.screen.fill(self.WHITE)
        
        # Draw player
        pygame.draw.rect(self.screen, self.BLUE,
                        (self.player_x, self.player_y, self.player_size, self.player_size))
        
        # Draw target
        pygame.draw.rect(self.screen, self.RED,
                        (self.target_x, self.target_y, self.target_size, self.target_size))
        
        # Draw obstacles
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, self.GREEN,
                           (obs['x'], obs['y'], self.obstacle_size, self.obstacle_size))
        
        pygame.display.flip()

    def get_screen(self):
        """获取当前游戏画面并缩放到固定大小"""
        screen = pygame.surfarray.array3d(pygame.display.get_surface())
        screen = np.transpose(screen, (1, 0, 2))  # 转置以匹配 PyTorch 格式
        screen = cv2.resize(screen, (84, 84))  # 缩放到 84x84
        screen = screen / 255.0  # 归一化到 [0, 1]
        return screen
    
    def close(self):
        pygame.quit() 