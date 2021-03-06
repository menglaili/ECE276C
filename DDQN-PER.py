import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import random
import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
dtype = torch.float
dir = 'save_files/'

class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()
        
        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

class Atari_Wrapper(gym.Wrapper):
    # env wrapper to resize images, grey scale and frame stacking and other misc.
    def __init__(self, env, env_name, k, dsize=(84, 84), use_add_done=False):
        super(Atari_Wrapper, self).__init__(env)
        self.dsize = dsize
        self.k = k
        self.use_add_done = use_add_done
        self.frame_cutout_h = (0, -1)
        self.frame_cutout_w = (0, -1)

    def reset(self):
        self.Return = 0
        self.last_life_count = 0

        ob = self.env.reset()
        ob = self.preprocess_observation(ob)

        # stack k times the reset ob
        self.frame_stack = np.stack([ob for i in range(self.k)])

        return self.frame_stack

    def step(self, action):
        # do k frameskips, same action for every intermediate frame
        # stacking k frames
        reward = 0
        done = False
        additional_done = False

        # k frame skips or end of episode
        frames = []
        for i in range(self.k):
            ob, r, d, info = self.env.step(action)

            # insert a (additional) done, when agent loses a life (Games with lives)
            if self.use_add_done:
                if info['ale.lives'] < self.last_life_count:
                    additional_done = True
                self.last_life_count = info['ale.lives']

            ob = self.preprocess_observation(ob)
            frames.append(ob)

            # add reward
            reward += r

            if d:  # env done
                done = True
                break

        # build the observation
        self.step_frame_stack(frames)

        # add info, get return of the completed episode
        self.Return += reward
        if done:
            info["return"] = self.Return

        # clip reward
        if reward > 0:
            reward = 1
        elif reward == 0:
            reward = 0
        else:
            reward = -1

        return self.frame_stack, reward, done, info, additional_done

    def step_frame_stack(self, frames):
        num_frames = len(frames)

        if num_frames == self.k:
            self.frame_stack = np.stack(frames)
        elif num_frames > self.k:
            self.frame_stack = np.array(frames[-self.k::])
            self.frame_stack = np.array(frames[-self.k::])
        else:  # mostly used when episode ends
            # shift the existing frames in the framestack to the front=0 (0->k, index is time)
            self.frame_stack[0: self.k - num_frames] = self.frame_stack[num_frames::]
            # insert the new frames into the stack
            self.frame_stack[self.k - num_frames::] = np.array(frames)

    def preprocess_observation(self, ob):
        # resize and grey and cutout image
        ob = cv2.cvtColor(ob[self.frame_cutout_h[0]:self.frame_cutout_h[1],
                          self.frame_cutout_w[0]:self.frame_cutout_w[1]], cv2.COLOR_BGR2GRAY)
        ob = cv2.resize(ob, dsize=self.dsize)

        return ob


class DQN(nn.Module):
    # nature paper architecture
    def __init__(self, in_channels, num_actions):
        super().__init__()

        network = [
            torch.nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        ]

        self.network = nn.Sequential(*network)

    def forward(self, x):
        actions = self.network(x)
        return actions


class Agent(nn.Module):
    def __init__(self, in_channels, num_actions, epsilon):
        super().__init__()
        self.in_channels = in_channels
        self.num_actions = num_actions
        self.network = DQN(in_channels, num_actions)
        self.eps = epsilon

    def forward(self, x):
        actions = self.network(x)
        return actions

    def e_greedy(self, x):
        actions = self.forward(x)
        greedy = torch.rand(1)
        if self.eps < greedy:
            return torch.argmax(actions)
        else:
            return (torch.rand(1) * self.num_actions).type('torch.LongTensor')[0]

    def greedy(self, x):
        actions = self.forward(x)
        return torch.argmax(actions)

    def set_epsilon(self, epsilon):
        self.eps = epsilon


class Logger:
    def __init__(self, filename):
        self.filename = filename
        f = open(f"{self.filename}.csv", "w")
        f.close()

    def log(self, msg):
        f = open(f"{self.filename}.csv", "a+")
        f.write(f"{msg}\n")
        f.close()


class PER():
    def __init__(self, capacity):
        self.memory = Memory(capacity)

    def insert(self, agent, obs, actions, rewards, dones):
        dones = [int(not dones[i]) for i in range(len(dones))][:-1]
        actions = actions[:-1]; rewards = rewards[:-1];
        minibatch_size = len(obs) - 1
        
        obsO = obs[:-1]; obs_nO = obs[1:]; actionsO = actions; rewardsO = rewards; donesO = dones
        
        
        obs = ((torch.stack(obs).to(dtype)) / 255)
        next_obs = obs[1:].to(device)
        obs = obs[:-1].to(device)
        actions = np.stack(actions)
        rewards = torch.tensor(rewards).to(device)
        dones = torch.tensor(dones).to(device)
        
        
        
        Qs = agent(torch.cat([obs, next_obs])).detach()
        obs_Q, next_obs_Q = torch.split(Qs, minibatch_size, dim=0)
        obs_Q = obs_Q[range(minibatch_size), actions]

        # target
        next_obs_Q_max = torch.max(next_obs_Q, 1)[1].detach()
        target_Q = target_agent(next_obs)[range(minibatch_size), next_obs_Q_max].detach()

        target = rewards + gamma * target_Q * dones
        
        error = torch.abs(target - obs_Q).cpu().numpy()
        
        for t in range(minibatch_size):
            self.memory.add(error[t], (obsO[t], actionsO[t], rewardsO[t], obs_nO[t], donesO[t]))
        
    def get(self, batch_size):
        sL = []; aL = []; rL = []; nL = []; dL = []
        mini_batch, idxs, is_weight = self.memory.sample(batch_size)
        for data in mini_batch:
            state, action, reward, next_state, done = data
            sL.append(state); aL.append(action); rL.append(reward); nL.append(next_state); dL.append(done)
        return [sL, aL, rL, nL, dL], idxs, is_weight
    
    def update(self, idx, errors):
        self.memory.update(idx, errors)
    
    def __len__(self):
        return self.memory.tree.n_entries


class Experience_Replay():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def insert(self, transitions):

        for i in range(len(transitions)):
            if len(self.memory) < self.capacity:
                self.memory.append(None)
            self.memory[self.position] = transitions[i]
            self.position = (self.position + 1) % self.capacity

    def get(self, batch_size):
        # return random.sample(self.memory, batch_size)
        indexes = (np.random.rand(batch_size) * (len(self.memory) - 1)).astype(int)
        return [self.memory[i] for i in indexes]

    def __len__(self):
        return len(self.memory)


class Env_Runner:

    def __init__(self, env, agent):
        super().__init__()

        self.env = env
        self.agent = agent

        self.logger = Logger(dir + "training_info")
        self.logger.log("training_step, return")

        self.ob = self.env.reset()
        self.total_steps = 0

    def run(self, steps):

        obs = []
        actions = []
        rewards = []
        dones = []

        for step in range(steps):

            self.ob = torch.tensor(self.ob)  # uint8
            action = self.agent.e_greedy(
                self.ob.to(device).to(dtype).unsqueeze(0) / 255)  # float32+norm ob.to(device).to(dtype).unsqueeze(0) / 255)
                # self.ob.to(dtype).unsqueeze(0) / 255)  # float32+norm ob.to(device).to(dtype).unsqueeze(0) / 255)
            action = action.detach().cpu().numpy()

            obs.append(self.ob)
            actions.append(action)

            self.ob, r, done, info, additional_done = self.env.step(action)

            if done:  # real environment reset, other add_dones are for q learning purposes
                self.ob = self.env.reset()
                if "return" in info:
                    self.logger.log(f'{self.total_steps + step},{info["return"]}')

            rewards.append(r)
            dones.append(done or additional_done)

        self.total_steps += steps

        return obs, actions, rewards, dones


def make_transitions(obs, actions, rewards, dones):
    # observations are in uint8 format
    tuples = []
    steps = len(obs) - 1
    for t in range(steps):
        tuples.append((obs[t],
                       actions[t],
                       rewards[t],
                       obs[t + 1],
                       int(not dones[t])))

    return tuples


if __name__ == '__main__':

    env_name = 'MsPacman-v0'

    # hyperparameter
    N = 5
    num_stacked_frames = 4
    replay_memory_size = 2500 * N  # *100
    min_replay_size_to_update = 250 * N # *100
    lr = 6e-5
    gamma = 0.99
    minibatch_size = 32
    steps_rollout = 16
    start_eps = 1
    final_eps = 0.1
    final_eps_frame = 1000 * N # *100
    total_steps = 20000 * N # *100
    target_net_update = 625  # 10000 steps
    save_model_steps = 5000 * N  # *100


    # init
    raw_env = gym.make(env_name)
    env = Atari_Wrapper(raw_env, env_name, num_stacked_frames, use_add_done=True)

    in_channels = num_stacked_frames
    num_actions = env.action_space.n

    eps_interval = start_eps - final_eps

    agent = Agent(in_channels, num_actions, start_eps).to(device)  # .to(device)
    target_agent = Agent(in_channels, num_actions, start_eps).to(device)  # .to(device)
    target_agent.load_state_dict(agent.state_dict())

    replay = PER(replay_memory_size)#Experience_Replay(replay_memory_size)
    runner = Env_Runner(env, agent)
    optimizer = optim.Adam(agent.parameters(), lr=lr)  # optim.RMSprop(agent.parameters(), lr=lr)
    huber_loss = torch.nn.SmoothL1Loss()

    num_steps = 0
    num_model_updates = 0
    
    total_episodes = 14
    eval_epsilon = 0.01
    num_stacked_frames = 4

    f = open(dir + env_name + "-Eval" + ".csv", "w")
    f.write('steps,returns\n')


    start_time = time.time()
    pbar = tqdm(total_steps)
    while num_steps < total_steps:
        agent.train()
        # set agent exploration | cap exploration after x timesteps to final epsilon
        new_epsilon = np.maximum(final_eps, start_eps - (eps_interval * num_steps / final_eps_frame))
        agent.set_epsilon(new_epsilon)

        # get data
        obs, actions, rewards, dones = runner.run(steps_rollout)
        replay.insert(agent, obs, actions, rewards, dones)
#         transitions = make_transitions(obs, actions, rewards, dones)
#         replay.insert(transitions)

        # add
        num_steps += steps_rollout

        # check if update
        if num_steps < min_replay_size_to_update:
            continue

        # update
        for update in range(4):
            optimizer.zero_grad()
            minibatch, idxs, is_weight = replay.get(minibatch_size)  # obs: [4 * 84 * 84, 4 * 84 * 84, ...]
            # uint8 to float32 and normalize to 0-1
            obs = ((torch.stack(minibatch[0]).to(dtype)) / 255).to(device)
            actions = np.stack(minibatch[1])
            rewards = torch.tensor(minibatch[2]).to(device)

            # uint8 to float32 and normalize to 0-1
            next_obs = ((torch.stack(minibatch[3]).to(dtype)) / 255).to(device)
            dones = torch.tensor(minibatch[4]).to(device)

            #  *** double dqn ***
            # prediction
            Qs = agent(torch.cat([obs, next_obs]))
            obs_Q, next_obs_Q = torch.split(Qs, minibatch_size, dim=0)
            obs_Q = obs_Q[range(minibatch_size), actions]

            # target
            next_obs_Q_max = torch.max(next_obs_Q, 1)[1].detach()
            target_Q = target_agent(next_obs)[range(minibatch_size), next_obs_Q_max].detach()

            target = rewards + gamma * target_Q * dones
            
            
            errors = torch.abs(target - obs_Q).detach().cpu().numpy()
            # update priority
            for i in range(obs.size()[0]):
                idx = idxs[i]
                replay.update(idx, errors[i])

            # loss
            loss = 1/obs.size()[0] * torch.sum(torch.tensor(is_weight).to(device) * (obs_Q - target) ** 2)
#             loss = huber_loss(obs_Q, target)  # torch.mean(torch.pow(obs_Q - target, 2))
            loss.backward()
            optimizer.step()

        num_model_updates += 1

        # update target network
        if num_model_updates % target_net_update == 0:
            target_agent.load_state_dict(agent.state_dict())

        # print time
        if num_steps % 50000 < steps_rollout:
            end_time = time.time()
            print(f'*** total steps: {num_steps} | time(50K): {end_time - start_time} ***')
            start_time = time.time()
            pbar.update()

        # save the dqn after some time
        if num_steps % (steps_rollout * 100) == 0:
            torch.save(agent, dir+f"{env_name}-{num_steps}.pt")
            
        if num_steps % (steps_rollout * 12) == 0:
            agent.set_epsilon(eval_epsilon)
            agent.eval()

            ob = env.reset()
            num_episode = 0
            returns = []
            while num_episode < total_episodes:

                action = agent.e_greedy(torch.tensor(ob, dtype=dtype).unsqueeze(0).to(device) / 255).detach()
                action = action.cpu().numpy()
                ob, _, done, info, _ = env.step(action)  # set render to false
                if done:
                    ob = env.reset()
                    returns.append(info["return"])
                    num_episode += 1

            f.write(f'{num_steps},{np.mean(returns)}\n')
    f.close()
    env.close()

    # save agent
    torch.save(agent, dir+"agent.pt")
    # load agent
    agent = torch.load(dir+"agent.pt")






    plt.rcParams["figure.figsize"] = (8, 5)
    filename = "training_info.csv"
    eval_name = "MsPacman-v0-Eval.csv"
    n_steps = 15
    f = pd.read_csv(dir + filename)
    g = pd.read_csv(dir + eval_name)
    return_data = f.loc[:][' return']
    mean = return_data.rolling(n_steps).mean()
    deviation = return_data.rolling(n_steps).std()
    under_line = (mean - deviation)
    over_line = (mean + deviation)

    fig = plt.figure()
    plt.plot(f["training_step"], mean, linewidth=1, label="training return")
    plt.fill_between(f["training_step"], under_line, over_line, color='b', alpha=0.1)
    plt.xlabel("Number of steps")
    plt.ylabel("Return")
    # plt.show()
    plt.title("Training return Vs. steps")
    plt.savefig('TrainReturns.png')
    plt.close()

    fig = plt.figure()
    plt.scatter(g["steps"], g["returns"], linewidth=1, color="r", label="evaluation return")
    plt.xlabel("Number of steps")
    plt.ylabel("Return")
    # plt.legend()
    plt.title("Testing return using policy trained by # steps")
    plt.savefig('TestReturns.png')
    plt.close()