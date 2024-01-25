# This project is an implementation of an AI for Lunar Landing, a predefined environment from Gymnasium.
# The actions, observation space and rewards obtained, are specified on their website: https://gymnasium.farama.org/environments/box2d/lunar_lander/

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
import torch.autograd as autograd
from torch.autograd import Variable
from collections import deque, namedtuple

# Architecture of the Neural Network
class Network(nn.Module):
  def __init__(self, state_size, action_size, seed = 42):
      super(Network, self).__init__()
      self.seed = torch.manual_seed(seed)
      self.fc1 = nn.Linear(state_size, 64) #number of neurons is an important parameter to experiment with, in every layer
      self.fc2 = nn.Linear(64, 64) #the number of layers, is also an important parameter with which to experiment
      self.fc3 = nn.Linear(64, action_size)

  def forward(self, state):
      x = self.fc1(state)
      x = F.relu(x) #activation function: activates neurons if positive value, deactivate the ones with negative value
      x = self.fc2(x)
      x = F.relu(x)
      return self.fc3(x)

# Implementing the environment
import gymnasium as gym
env = gym.make('LunarLander-v2')
state_shape = env.observation_space.shape #dimensions of the observation space (ex. two-dimensional = (4, ), three-dimensional = (3,3,3))
state_size = env.observation_space.shape[0] #total size of the observation space (ex. two-dimensional = 4, three-dimensional = 27)
num_actions = env.action_space.n #number of possible actions
print('State shape: ', state_shape)
print('State size: ', state_size)
print('Number of actions: ', num_actions)

# Initializing the hyperparameters
learning_rate = 5e-4 #inicial weight
minibatch_size = 100 #number of observations used in one step of the training to update the model parameters
discount_factor = 0.99 #number that gives value to the rewards obtained after performing certain actions: a low number ~0 focuses on immediate rewards, a high number ~1 focuses on long-term rewards.
replay_buffer_size = 100000 #size of the memory to store the experiences
tau = 0.001 #is a value used to smooth the updates of the weights of a target network with the weights of another target network to avoid abrupt changes.

# Implementing Experience Replay
class ReplayMemory(object):
  def __init__(self, capacity):
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #to be able to execute the code outside Google colab
    self.capacity = capacity
    self.memory = [] #list that store the experiences

  def push(self, event): #save the event on memory
    self.memory.append(event)
    if len(self.memory) > self.capacity:
      del self.memory[0]

  def sample(self, batch_size): #select a random batch of events from the memory
    experiences = random.sample(self.memory, k = batch_size)
    states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(self.device) #takes the state of all experience tuples, and converts them into tensors
    actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(self.device)
    rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(self.device)
    next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(self.device)
    dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None])).to(dtype=torch.uint8).float().to(self.device)
    return states, next_states, actions, rewards, dones

# Implementing the DQN class
class Agent():
  def __init__(self, state_size, action_size):
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.state_size = state_size
    self.action_size = action_size
    self.local_qnetwork = Network(state_size, action_size).to(self.device) #used to select actions
    self.target_qnetwork = Network(state_size, action_size).to(self.device) #used to calculate the Q values of future actions, that will be used in the local network
    self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr = learning_rate)
    self.memory = ReplayMemory(replay_buffer_size)
    self.t_step = 0

  def step(self, state, action, reward, next_state, done):
    self.memory.push((state, action, reward, next_state, done))
    self.t_step = (self.t_step + 1) % 4 #learn every 4 steps
    if self.t_step == 0:
      if len(self.memory.memory) > minibatch_size: #learn from all the batch of experiences
        experiences = self.memory.sample(100)
        self.learn(experiences, discount_factor)

  def act(self, state, epsilon = 0.): #epsilon is the greedy factor, the probability of performing a random action instead of the optimal one
    state = torch.from_numpy(state).float().unsqueeze(0).to(self.device) #new variable to know which batch each state belongs to
    self.local_qnetwork.eval()
    with torch.no_grad():
      action_values = self.local_qnetwork(state) #Q values of every possible action nowQue
    self.local_qnetwork.train()
    if random.random() > epsilon:
      return np.argmax(action_values.cpu().data.numpy())
    else:
      return random.choice(np.arange(self.action_size))

  def learn(self, experiences, discount_factor):
    states, next_states, actions, rewards, dones = experiences
    next_qtargets = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
    q_targets = rewards + (discount_factor * next_qtargets * (1 - dones))
    q_expected = self.local_qnetwork(states).gather(1, actions)
    loss = F.mse_loss(q_targets, q_expected)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    self.soft_update(self.local_qnetwork, self.target_qnetwork, tau)

  def soft_update(self, local_model, target_model, tau):
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
      target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data) #updates the target network parameters with the local network parameters

# Initializing the DQN agent
agent = Agent(state_size, num_actions)

# Training the DQN agent
num_episodes = 2000
max_episode_timesteps = 1000
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
  if np.mean(scores) >= 200.0:
    print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode - 100, np.mean(scores)), end = "")
    torch.save(agent.local_qnetwork.state_dict(), 'checkpoint.pth')
    break

# Visualizing the results
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
        state, reward, done, _, _ = env.step(action.item())
    env.close()
    imageio.mimsave('video.mp4', frames, fps=30)

show_video_of_model(agent, 'LunarLander-v2')

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