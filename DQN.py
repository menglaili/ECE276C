import numpy as np
import gym
import random
from tqdm import tqdm
import os
from PIL import Image
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import warnings
warnings.filterwarnings("ignore")


class Replay():
    def __init__(self, buffer_size, init_length, state_dim, action_dim, env):
        """
        A function to initialize the replay buffer.

        param: buffer_size: Size of replay buffer
        param: init_length : Initial number of transitions to collect
        param: state_dim : Size of the state space
        param: action_dim : Size of the action space
        param: env : gym environment object
        """
        self.buffer_size = buffer_size
        self.init_length = init_length
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.env = env

        self.cntr = 0
        self.buffer = defaultdict()
        self.buffer['state'] = np.zeros((self.buffer_size, self.state_dim))
        self.buffer['next_state'] = np.zeros((self.buffer_size, self.state_dim))
        self.buffer['action'] = np.zeros((self.buffer_size, self.action_dim))
        self.buffer['reward'] = np.zeros(self.buffer_size)
        self.buffer['done'] = np.zeros(self.buffer_size, dtype=np.bool)

        self._init_buffer()

    def _init_buffer(self):

        while self.cntr < self.init_length:
            done = False
            state = self.env.reset()
            step = 1
            while not done:
                action = self.env.action_space.sample()
                next_state, reward, done, _ = self.env.step(action)
                step += 1
                exp = (state, action, reward, next_state, done)
                self.buffer_add(exp)

                state = next_state
                if self.cntr >= self.init_length:
                    break

    def buffer_add(self, exp):
        """
        A function to add a dictionary to the buffer
        param: exp : A dictionary consisting of state, action, reward , next state and done flag
        """

        idx = self.cntr % self.buffer_size
        self.buffer['state'][idx, :] = exp[0]
        self.buffer['action'][idx, :] = exp[1]
        self.buffer['reward'][idx] = exp[2]
        self.buffer['next_state'][idx, :] = exp[3]
        self.buffer['done'][idx] = exp[4]

        self.cntr += 1


class DQN(nn.Module):
    """Deep Q-learning Network"""
    def __init__(self, input_channels, img_height, img_width, num_actions):
        super(DQN, self).__init__()
        # input 3*160*160
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=4, padding=2) # -> 32x40x40
        # self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=1) # -> 64x19x19
        # self.bn1 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1) # -> 64x19x19
        # self.bn1 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(in_features=19*19*64, out_features=512)
        self.fc1 = nn.Linear(in_features=512, out_features=num_actions)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


env = gym.make('MsPacman-v0').unwrapped
resize = T.Compose([T.ToPILImage(),
                    T.Resize((166,160), interpolation=Image.CUBIC),
                    T.ToTensor()])

def get_screen():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, :int(screen_height * 0.8),:]

    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0)


if __name__ == '__main__':