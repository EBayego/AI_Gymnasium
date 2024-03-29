# This project is an implementation of an AI for Pacman, a predefined environment from Gymnasium.
# The actions and observations are specified on their website: https://gymnasium.farama.org/environments/atari/pacman/

# Installing Gymnasium
# !pip install gymnasium
# !pip install "gymnasium[atari, accept-rom-license]"
# !apt-get install -y swig
# !pip install gymnasium[box2d]

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from torch.utils.data import DataLoader, TensorDataset

# Architecture of the Neural Network
class Network(nn.Module):
  def __init__(self, action_size, seed = 42):
    super(Network, self).__init__()
    self.seed = torch.manual_seed(seed)
    self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4) #convolutional layer, 3 initial input channles that equals to RGB, and the other parameters are for experimentation
    self.bn1 = nn.BatchNorm2d(32) # normalization layers
    self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
    self.bn2 = nn.BatchNorm2d(64)
    self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
    self.bn3 = nn.BatchNorm2d(64)
    self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
    self.bn4 = nn.BatchNorm2d(128)

    self.fc1 = nn.Linear(10*10*128, 512) #dimensions*output_features from the convolutional layer
    self.fc2 = nn.Linear(512, 256)
    self.fc3 = nn.Linear(256, action_size)

  def forward(self, state):
    x = F.relu(self.bn1(self.conv1(state)))
    x = F.relu(self.bn2(self.conv2(x)))
    x = F.relu(self.bn3(self.conv3(x)))
    x = F.relu(self.bn4(self.conv4(x)))

    x = x.view(x.size(0), -1) #flattening layer

    x = F.relu(self.fc1(x)) #sends the info throw the connected layers having the ReLU activation function applied
    x = F.relu(self.fc2(x))
    return self.fc3(x)

# Setting up the environment
import gymnasium as gym
env = gym.make('MsPacmanDeterministic-v0', full_action_space = False)
state_shape = env.observation_space.shape #dimensions of the observation space (ex. two-dimensional = (4, ), three-dimensional = (3,3,3))
state_size = env.observation_space.shape[0] #total size of the observation space (ex. two-dimensional = 4, three-dimensional = 27)
num_actions = env.action_space.n #number of possible actions
print('State shape: ', state_shape)
print('State size: ', state_size) #delete?
print('Number of actions: ', num_actions)

# Initializing the hyperparameters
learning_rate = 5e-4 #inicial weight
minibatch_size = 64 #number of observations used in one step of the training to update the model parameters
discount_factor = 0.99 #number that gives value to the rewards obtained after performing certain actions: a low number ~0 focuses on immediate rewards, a high number ~1 focuses on long-term rewards.

# Preprocessing the frames
from PIL import Image
from torchvision import transforms

def preprocess_frame(frame):
  frame = Image.fromarray(frame)
  preprocess = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
  return preprocess(frame).unsqueeze(0)

# Implementing the DCQN class
class Agent():
  def __init__(self, action_size):
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.action_size = action_size
    self.local_qnetwork = Network(action_size).to(self.device) #used to select actions
    self.target_qnetwork = Network(action_size).to(self.device) #used to calculate the Q values of future actions, that will be used in the local network
    self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr = learning_rate)
    self.memory = deque(maxlen = 10000)

  def step(self, state, action, reward, next_state, done):
    state = preprocess_frame(state)
    next_state = preprocess_frame(next_state)
    self.memory.append((state, action, reward, next_state, done))
    if len(self.memory) > minibatch_size: #learn from all the batch of experiences
      experiences = random.sample(self.memory, k = minibatch_size)
      self.learn(experiences, discount_factor)

  def act(self, state, epsilon = 0.): #epsilon is the greedy factor, the probability of performing a random action instead of the optimal one
    state = preprocess_frame(state).to(self.device)
    self.local_qnetwork.eval()
    with torch.no_grad():
      action_values = self.local_qnetwork(state) #Q values of every possible action now
    self.local_qnetwork.train()
    if random.random() > epsilon:
      return np.argmax(action_values.cpu().data.numpy())
    else:
      return random.choice(np.arange(self.action_size))

  def learn(self, experiences, discount_factor):
    states, actions, rewards, next_states, dones = zip(*experiences)
    states = torch.from_numpy(np.vstack(states)).float().to(self.device) #takes the state of all experience tuples, and converts them into tensors
    actions = torch.from_numpy(np.vstack(actions)).long().to(self.device)
    rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
    next_states = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
    dones = torch.from_numpy(np.vstack(dones)).to(dtype=torch.uint8).float().to(self.device)
    next_qtargets = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
    q_targets = rewards + (discount_factor * next_qtargets * (1 - dones))
    q_expected = self.local_qnetwork(states).gather(1, actions)
    loss = F.mse_loss(q_targets, q_expected)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

  # soft update doesn't improve this specific model (tested in experimentation)

# Initializing the DCQN agent
agent = Agent(num_actions)

# Training the DCQN agent
num_episodes = 2000
max_episode_timesteps = 10000
initial_epsilon = 1.0
final_epsilon = 0.01
epsilon_decay = 0.995
epsilon = initial_epsilon
scores = deque(maxlen = 100)

for episode in range(1, num_episodes + 1):
  state, _ = env.reset()
  score = 0
  for t in range(max_episode_timesteps):
    action = agent.act(state, epsilon)
    next_state, reward, done, _, _ = env.step(action)
    agent.step(state, action, reward, next_state, done)
    state = next_state
    score += reward
    if done:
      break
  scores.append(score)
  epsilon = max(final_epsilon, epsilon_decay * epsilon)
  print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores)), end = "")
  if episode % 100 == 0:
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores)))
  if np.mean(scores) >= 500.0:
    print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode - 100, np.mean(scores)), end = "")
    torch.save(agent.local_qnetwork.state_dict(), 'checkpoint.pth')
    break

#Visualizing the results
import glob
import io
import base64
import imageio
from IPython.display import HTML, display
from gym.wrappers.monitoring.video_recorder import VideoRecorder

def show_video_of_model(agent, env_name):
    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    done = False
    frames = []
    while not done:
        frame = env.render()
        frames.append(frame)
        action = agent.act(state)
        state, reward, done, _, _ = env.step(action)
    env.close()
    imageio.mimsave('video.mp4', frames, fps=30)

show_video_of_model(agent, 'MsPacmanDeterministic-v0')

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