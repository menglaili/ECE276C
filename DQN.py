import numpy as np
import gym
import random
from tqdm import tqdm
import os
# import matplotlib
# import matplotlib.pyplot as plt
from PIL import Image
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from collections import namedtuple
from torch.utils.tensorboard import SummaryWriter
from itertools import count
import cv2
import imageio
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='DQN_Pacman')
parser.add_argument('--load', default=0, type=int, metavar='N',
                    help='0: Dont load model; 1: load model')


print(torch.cuda.is_available())
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESIZE = T.Compose([T.ToPILImage(),
                        T.Resize((160, 160), interpolation=Image.CUBIC),
                        T.ToTensor()])


def edit_screen2state(screen):
    # input size (210,160,3) -> (3, 210, 160)
    screen = screen.transpose((2, 0, 1))
    _, screen_height, screen_width = screen.shape
    # edited -> (3, 172, 160)
    screen = screen[:, :int(screen_height * 0.82),:]

    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return RESIZE(screen).unsqueeze(0).to(DEVICE)


Transition = namedtuple('Transition',('state', 'next_state', 'action', 'reward'))


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
        self.fc2 = nn.Linear(in_features=512, out_features=num_actions)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class Replay():
    def __init__(self, buffer_size, init_length, env):
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
        self.env = env
        self.fullMemory_flag = False
        self.cntr = 0
        self.buffer = []
        self._init_buffer()

    def _init_buffer(self):

        while self.cntr < self.init_length:
            done = False
            screen = self.env.render(mode='rgb_array')
            state = edit_screen2state(screen)
            while not done:
                action = torch.tensor([[self.env.action_space.sample()]], device=DEVICE, dtype=torch.long)
                _, reward, done, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=DEVICE)

                if not done:
                    screen = self.env.render(mode='rgb_array')
                    next_state = edit_screen2state(screen)
                    self.buffer_add(state, next_state, action, reward)
                else:
                    self.buffer_add(state, None, action, reward)

                state = next_state
                if self.cntr >= self.init_length:
                    break

    def buffer_add(self, *args):
        """
        A function to add a dictionary to the buffer
        param: exp : A dictionary consisting of state, action, reward , next state and done flag
        """
        if (not self.fullMemory_flag) and self.cntr < self.buffer_size:
            self.buffer.append(Transition(*args))
        else:
            self.cntr = self.cntr % self.buffer_size
            self.fullMemory_flag = True
            self.buffer[self.cntr] = Transition(*args)
        self.cntr += 1

    def buffer_sample(self, N):
        """
        A function to sample N points from the buffer
        param: N : Number of samples to obtain from the buffer
        """
        return random.sample(self.buffer, N)


class DQNAgent():
    def __init__(
            self,
            env,
            learning_rate=3e-4,
            gamma=0.999,
            batch_size=256,
            buffer_size=50000,
            init_length=5000,
            EPS_START=0.9,
            EPS_END=0.05,
            EPS_DECAY=160,
            Target_update=10
    ):
        """
        param: env: An gym environment
        param: learning_rate: Learning rate
        param: gamma: The discount factor
        param: batch_size: The batch size for training
        param: buffer_size: Size of replay buffer
        param: init_length : Initial number of transitions to collect
        """
        self.gamma = gamma
        self.batch_size = batch_size
        self.env = env
        self.n_actions = self.env.action_space.n
        self.test_env = gym.make('MsPacman-v0').unwrapped
        self.eps_s = EPS_START
        self.eps_e = EPS_END
        self.eps_decay = EPS_DECAY
        self.target_update = Target_update
        self.num_evals = 1
        # For updating the epsilon
        self.eps_step = 0

        # get screen size/state size so that we can initialize layers
        _ = self.env.reset()
        screen = self.env.render(mode='rgb_array')
        state = edit_screen2state(screen)
        _, _, h, w = state.shape

        # Create a actor and actor_target
        self.main_net = DQN(3, h, w, self.n_actions).to(DEVICE)
        self.target_net = DQN(3, h, w, self.n_actions).to(DEVICE)
        self.target_net.load_state_dict(self.main_net.state_dict())
        self.target_net.eval()
        # Make sure that both networks have the same initial weights

        # Define the optimizer for the main network
        #         self.optimizer= optim.Adam(self.main_net.parameters(), lr=learning_rate)
        self.optimizer = optim.RMSprop(self.main_net.parameters(), lr=learning_rate)

        # define a replay buffer
        self.ReplayBuffer = Replay(buffer_size, init_length, self.env)

    def select_action_train(self, state):
        p = random.random()
        threshold = self.eps_e + (self.eps_s - self.eps_e) * \
                    np.exp(-1.0 * self.eps_step / self.eps_decay)
        self.eps_step += 1
        if p > threshold:
            with torch.no_grad():
                return self.main_net.forward(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=DEVICE, dtype=torch.long)

    #     def update_target_networks(self):
    #         """
    #         A function to update the target networks
    #         """
    #         weighSync(self.actor_target, self.actor)
    #         weighSync(self.critic_target, self.critic)

    def update_network(self):
        """
        A function to update the function just once
        """
        packages = self.ReplayBuffer.buffer_sample(self.batch_size)
        batch = Transition(*zip(*packages))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=DEVICE, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)

        curr_Q = self.main_net(states).gather(1, actions)
        max_Q = torch.zeros(self.batch_size, device=DEVICE)
        max_Q[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_Q = (max_Q * self.gamma) + rewards

        self.optimizer.zero_grad()
        loss = F.mse_loss(curr_Q, expected_Q.unsqueeze(1))
        loss.backward()
        # Limiting the extent of network updates can greatly improve the effectiveness of training
        for param in self.main_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()

    def train(self, last_i, num_games, log_dir):
        """
        Train the policy for the given number of iterations
        :param num_games:The number of steps to train the policy for
        """
        writer = SummaryWriter(log_dir)
        for i in tqdm(range(last_i+1, num_games)):

            #             # obtain action
            #             action_mean = self.actor.forward(torch.FloatTensor(state)).cpu().detach().numpy()
            #             noise = np.random.multivariate_normal(mean=[0, 0], cov=np.diag([0.1, 0.1]))
            #             action = np.clip(action_mean+noise, -1, 1)
            _ = self.env.reset()
            screen = self.env.render(mode='rgb_array')
            state = edit_screen2state(screen)

            #         writer.add_graph(self.main_net, state)
            #         writer.add_graph(self.target_net, state)

            rewards = 0
            for t in count():
                # obtain action
                action = self.select_action_train(state)
                # step
                next_state, reward, done, _ = self.env.step(action.item())
                # duration

                # Reward
                rewards = rewards + reward
                reward = torch.tensor([reward], device=DEVICE)

                if not done:
                    next_state = edit_screen2state(next_state)
                    self.ReplayBuffer.buffer_add(state, next_state, action, reward)
                    state = next_state
                else:
                    self.ReplayBuffer.buffer_add(state, None, action, reward)
#                     cum_rewards = sum(rewards)
                    writer.add_scalar("Number of moves Per Game", (t+1), i)
                    writer.add_scalar("total of rewards Per Game", rewards, i)

                    print("Episodes-{}| scores={}| durations={}".format(i, rewards, (t+1)))

                # update the network
                loss = self.update_network()
                # writer.add_scalar("Loss", loss, i)
                if done:
                    break
            if i % self.target_update == 0 or i == num_games - 1:
                self.target_net.load_state_dict(self.main_net.state_dict())
                durations, total_scores = self.evaluation()
                writer.add_scalar("Evaluation Per 10 Trainings: Scores", total_scores, self.num_evals)
                writer.add_scalar("Evaluation Per 10 Trainings: Durations", durations, self.num_evals)
                torch.save({
                    'episode': i,
                    'step_done': self.eps_step,
                    'num_eval': self.num_evals,
                    'model_state_dict_target': self.target_net.state_dict(),
                    'model_state_dict_policy': self.main_net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    }, 'models/dqn_pacman_b256.pth')
                self.num_evals += 1
        writer.close()
        self.env.close()
        self.gif_generator()

    def evaluation(self):
        _ = self.test_env.reset()
        screen = self.test_env.render(mode='rgb_array')
        state = edit_screen2state(screen)
        step = 1
        total_rewards = 0
        done = False
        while not done:
            action = self.main_net.forward(state).max(1)[1].view(1, 1)
            next_state, reward, done, _ = self.test_env.step(action.item())

            state = edit_screen2state(next_state)
            step += 1
            total_rewards += reward

        return step, total_rewards
    
    def gif_generator(self):
        steps = 5000
        _ = self.test_env.reset()
        state = self.test_env.render(mode='rgb_array')
        
        imgs = []
        done = False
        for step in range(steps):
        #     here you should generate your action
        #     env.render(mode="rgb_array")
            state = edit_screen2state(state)
            action = self.main_net.forward(state).max(1)[1].view(1, 1)
            state, _, done, _ = self.test_env.step(action.item())

            if done:
                state = self.test_env.reset()
            img_tmp = np.asarray(state)
        #     img_tmp = img_tmp.swapaxes(0,1).swapaxes(1,2)
            imgs.append(img_tmp)

        self.test_env.close()
        imageio.mimwrite('myfile.gif', imgs)

        
def main():
    global args
    args = parser.parse_args()
    
    env = gym.make('MsPacman-v0').unwrapped
 
    pacman = DQNAgent(
        env,
        learning_rate=1e-2,
        gamma=0.999,
        batch_size=256,
        buffer_size=20000,
        init_length=2000
    )
    # Train the policy
    result_dir = "data/"
    log_dir = os.path.join(result_dir, "logfinal")
    print(log_dir)
    os.makedirs(log_dir, exist_ok=True)
#     os.makedirs('models/dqn_pacman', exist_ok=True)
    last_i = -1
    if args.load == 1:
        checkpoint = torch.load(os.path.join('models','dqn_pacman_b256.pth'), map_location=DEVICE)
        last_i = checkpoint['episode']
        pacman.eps_decay = 160
        print("pacman eps_decay:{}".format(pacman.eps_decay))
        pacman.eps_step = checkpoint['step_done']
        pacman.main_net.load_state_dict(checkpoint['model_state_dict_policy'])
        pacman.target_net.load_state_dict(checkpoint['model_state_dict_target'])
        pacman.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('loaded')
    pacman.train(int(last_i), int(200), log_dir)
    env.close()


if __name__ == "__main__":
    main()

