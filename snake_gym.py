import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces

class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}
    
    def __init__(self, size=30, fpos=None):
        super(SnakeEnv, self).__init__()
        self.size = size
        self.state_space = (size, size, 2)
        self.action_space = spaces.Discrete(4)  # 4 possible moves: up, down, left, right
        self.observation_space = spaces.Box(low=0, high=255, shape=(size, size, 2), dtype=np.int16)
        self.fpos = fpos or [(5,5),(3,3),(6,6),(5,6),(6,7),(5,5)]
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(self.state_space, dtype=np.int16)
        self.snake = [(self.size // 2, self.size // 2)]  # Initial snake position
        self.direction = 0  # Default direction: Up
        self.spawn_food()
        return self.board, {}
    
    def spawn_food(self):
        empty_cells = [(i, j) for i in range(self.size) for j in range(self.size) if (i, j) not in self.snake]
        self.food = random.choice(empty_cells)
        self.board[self.food] = [0, 255]
    
    def step(self, action):
        x, y = self.snake[0]
        if action == 0: x -= 1  # Up
        elif action == 1: x += 1  # Down
        elif action == 2: y -= 1  # Left
        elif action == 3: y += 1  # Right
        
        new_head = (x, y)
        done = new_head in self.snake or x < 0 or y < 0 or x >= self.size or y >= self.size
        reward = -10 if done else -1
        
        if not done:
            self.snake.insert(0, new_head)
            if new_head == self.food:
                reward = 10
                self.spawn_food()
            else:
                self.snake.pop()
        
        self.update_board()
        return self.board, reward, done, False, {}
    
    def update_board(self):
        self.board.fill(0)
        for x, y in self.snake:
            self.board[x, y] = [255, 0]
        self.board[self.food] = [0, 255]
    
    def render(self):
        print("\n".join(" ".join("H" if (i, j) in self.snake else "F" if (i, j) == self.food else "." for j in range(self.size)) for i in range(self.size)))
    
    def close(self):
        pass

# Example usage
if __name__ == "__main__":
    env = SnakeEnv()
    obs, _ = env.reset()
    done = False
    while not done:
        action = random.choice([0, 1, 2, 3])  # Random action
        obs, reward, done, _, _ = env.step(action)
        env.render()
        print("Reward:", reward)