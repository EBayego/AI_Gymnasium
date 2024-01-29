# This project is an implementation of an AI for Pacman, a predefined environment from Gymnasium.
# The actions and observations are specified on their website: https://gymnasium.farama.org/environments/atari/kung_fu_master/

# Installing Gymnasium
# !pip install gymnasium
# !pip install "gymnasium[atari, accept-rom-license]"
# !apt-get install -y swig
# !pip install gymnasium[box2d]

import cv2
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributions as distributions
from torch.distributions import Categorical
import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box

# Architecture of the Neural Network
class Network(nn.Module):
  def __init__(self, action_size, seed = 42):
    super(Network, self).__init__()
    self.seed = torch.manual_seed(seed)
    self.conv1 = nn.Conv2d(in_channels = 4, out_channels = 32, kernel_size = (3,3), stride = 2) #convolutional layer, always good to experiment with different values
    self.conv2 = nn.Conv2d(32, 32, (3,3), 2)
    self.conv3 = nn.Conv2d(32, 32, (3,3), 2)
    # normalization layers dont improve the results after some testing

    self.flatten = nn.Flatten() #new flattening layer

    self.fc1 = nn.Linear(in_features = 512, out_features = 128) #in_features = dimensions*output_features from the convolutional layer
    self.fc2a = nn.Linear(128, action_size) #output layer for the action value
    self.fc2s = nn.Linear(128, 1) #output layer for the state value

  def forward(self, state):
    x = F.relu(self.conv1(state))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))

    x = self.flatten(x) #flattening layer

    x = F.relu(self.fc1(x)) #sends the info throw the connected layers having the ReLU activation function applied
    action_values = self.fc2a(x) #no relu here, we don't want to limit these values to being non-negative
    state_value = self.fc2s(x)[0]
    return action_values, state_value

# Setting up the environment
class PreprocessAtari(ObservationWrapper):
  def __init__(self, env, height = 42, width = 42, crop = lambda img: img, dim_order = 'pytorch', color = False, n_frames = 4):
    super(PreprocessAtari, self).__init__(env)
    self.img_size = (height, width)
    self.crop = crop
    self.dim_order = dim_order
    self.color = color
    self.frame_stack = n_frames
    n_channels = 3 * n_frames if color else n_frames
    obs_shape = {'tensorflow': (height, width, n_channels), 'pytorch': (n_channels, height, width)}[dim_order]
    self.observation_space = Box(0.0, 1.0, obs_shape)
    self.frames = np.zeros(obs_shape, dtype = np.float32)

  def reset(self):
    self.frames = np.zeros_like(self.frames)
    obs, info = self.env.reset()
    self.update_buffer(obs)
    return self.frames, info

  def observation(self, img):
    img = self.crop(img)
    img = cv2.resize(img, self.img_size)
    if not self.color:
      if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype('float32') / 255.
    if self.color:
      self.frames = np.roll(self.frames, shift = -3, axis = 0)
    else:
      self.frames = np.roll(self.frames, shift = -1, axis = 0)
    if self.color:
      self.frames[-3:] = img
    else:
      self.frames[-1] = img
    return self.frames

  def update_buffer(self, obs):
    self.frames = self.observation(obs)

def make_env():
  env = gym.make("KungFuMasterDeterministic-v0", render_mode = 'rgb_array')
  env = PreprocessAtari(env, height = 42, width = 42, crop = lambda img: img, dim_order = 'pytorch', color = False, n_frames = 4)
  return env

env = make_env()

state_shape = env.observation_space.shape
num_actions = env.action_space.n
print("Observation shape:", state_shape)
print("Number actions:", num_actions)
print("Action names:", env.env.env.get_action_meanings())

# Initializing the hyperparameters
learning_rate = 1e-4 #inicial weight
discount_factor = 0.99 #number that gives value to the rewards obtained after performing certain actions: a low number ~0 focuses on immediate rewards, a high number ~1 focuses on long-term rewards.
num_environments = 10

# Implementing the A3C class
class Agent():
  def __init__(self, action_size):
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.action_size = action_size
    self.network = Network(action_size).to(self.device)
    self.optimizer = optim.Adam(self.network.parameters(), lr = learning_rate)

  def act(self, state):
    if state.ndim == 3: #if the dimnesion of the state is 3, it is not in a batch and is a single observation, because images have three dimensions (height, width and channel)
      state = [state] #transform the single obs into a list
    state = torch.tensor(state, dtype = torch.float32, device = self.device)
    action_values, _ = self.network.forward(state)
    policy = F.softmax(action_values, dim = -1) #softmax policy, transforming the values into probabilities
    return np.array([np.random.choice(len(p), p = p) for p in policy.detach().cpu().numpy()])

  def step(self, state, action, reward, next_state, done): #batches
    batch_size = state.shape[0]
    state = torch.tensor(state, dtype = torch.float32, device = self.device)
    next_state = torch.tensor(next_state, dtype = torch.float32, device = self.device)
    reward = torch.tensor(reward, dtype = torch.float32, device = self.device)
    done = torch.tensor(done, dtype = torch.bool, device = self.device).to(dtype = torch.float32)

    action_values, state_value = self.network(state) #being an object that inherits from nn.Module, Python automatically calls the method forward
    _, next_state_value = self.network(next_state)
    target_state_value = reward + discount_factor * next_state_value * (1 - done)
    advantage = target_state_value - state_value

    probs = F.softmax(action_values, dim = -1)
    logprobs = F.log_softmax(action_values, dim = -1)
    entropy = -torch.sum(probs * logprobs, axis = -1)
    batch_idx = np.arange(batch_size)
    logp_actions = logprobs[batch_idx, action]

    actor_loss = -(logp_actions * advantage.detach()).mean() - 0.001 * entropy.mean()
    critic_loss = F.mse_loss(target_state_value.detach(), state_value)
    total_loss = actor_loss + critic_loss
    self.optimizer.zero_grad()
    total_loss.backward()
    self.optimizer.step()

# Initializing the A3C agent
agent = Agent(num_actions)

# Evaluating our A3C agent on a single episode
def evaluate(agent, env, num_episodes = 1):
  episodes_rewards = []
  for _ in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    while True:
      action = agent.act(state)
      state, reward, done, info, _ = env.step(action[0])
      total_reward += reward
      if done:
        break
    episodes_rewards.append(total_reward)
  return episodes_rewards

# Testing multiple agents on multiple environments at the same time
class EnvBatch:
  def __init__(self, num_envs = 10):
    self.envs = [make_env() for _ in range(num_envs)]

  def reset(self):
    _states = []
    for env in self.envs:
      _states.append(env.reset()[0])
    return np.array(_states)

  def step(self, actions):
    next_states, rewards, dones, infos, _ = map(np.array, zip(*[env.step(a) for env, a in zip(self.envs, actions)]))
    for i in range(len(self.envs)):
      if dones[i]:
        next_states[i] = self.envs[i].reset()[0]
    return next_states, rewards, dones, infos

# Training the A3C agent
import tqdm

env_batch = EnvBatch(num_environments)
batch_states = env_batch.reset()

with tqdm.trange(0, 3001) as progress_bar:
  for i in progress_bar:
    batch_actions = agent.act(batch_states)
    batch_next_states, batch_rewards, batch_dones, _ = env_batch.step(batch_actions)
    batch_rewards *= 0.01
    agent.step(batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones)
    batch_states = batch_next_states
    if i % 1000 == 0:
      print("Average agent reward: ", np.mean(evaluate(agent, env, num_episodes=10)))

# Visualizing the results
import glob
import io
import base64
import imageio
from IPython.display import HTML, display
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder

def show_video_of_model(agent, env):
  state, _ = env.reset()
  done = False
  frames = []
  while not done:
    frame = env.render()
    frames.append(frame)
    action = agent.act(state)
    state, reward, done, _, _ = env.step(action[0])
  env.close()
  imageio.mimsave('video.mp4', frames, fps=30)

show_video_of_model(agent, env)

def show_video():
    mp4list = glob.glob('*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display(HTML(data='''<video alt="test" autoplay
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")

show_video()