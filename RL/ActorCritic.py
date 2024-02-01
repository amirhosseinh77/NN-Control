import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from RL.nn import DeterministicPolicyNetwork, ValueNetwork, ObserverNetwork
from RL.plot import plot_return
from utils.buffer import ReplayBuffer

device = "cuda" if torch.cuda.is_available() else "cpu"

class ActorCriticAgent():
    def __init__(self, state_size, action_size, hidden_dim, action_max, lr=1e-3):
        self.memory = []
        self.actor =  DeterministicPolicyNetwork(state_size, action_size, hidden_dim, action_max).to(device)
        self.critic = ValueNetwork(state_size, hidden_dim).to(device)
        self.observer = ObserverNetwork(state_size, action_size, hidden_dim).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr, weight_decay=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr, weight_decay=1e-4)
        self.observer_optimizer = optim.Adam(self.observer.parameters(), lr=lr, weight_decay=1e-4)


    # def push_memory(self, state, action, reward, next_state, done):
    #     new_experience = [state, action, reward, next_state, done]
    #     self.memory.append(new_experience)

    def learn(self, state, action, reward, next_state, done, gamma=0.99):
        
        # states, actions, rewards, next_states, dones = zip(*self.memory)

        # state_tensor = torch.tensor(state).view(1,-1).to(torch.float32).to(device)
        # next_state_tensor = torch.tensor(next_state).view(1,-1).to(torch.float32).to(device)
        # action = torch.tensor(action).view(1,-1).to(torch.float32).to(device)
        # reward = torch.tensor(reward).view(1,-1).to(torch.float32).to(device)
        # done = torch.tensor(done).view(1,-1).to(torch.float32).to(device)

        state_tensor = torch.tensor(state).to(torch.float32).to(device)
        next_state_tensor = torch.tensor(next_state).to(torch.float32).to(device)
        action = torch.tensor(action).to(torch.float32).to(device).view(-1,1)
        reward = torch.tensor(reward).to(torch.float32).to(device).view(-1,1)
        done = torch.tensor(done).to(torch.float32).to(device).view(-1,1)
        # print(state_tensor.shape, next_state_tensor.shape, action.shape, reward.shape, done.shape)

        
        states_val = self.critic(state_tensor.detach())
        next_states_val = self.critic(next_state_tensor.detach())
        value_targets = reward + gamma * next_states_val * torch.logical_not(done)
        
        # Compute Observer Loss
        observer_loss = F.mse_loss(next_state_tensor, self.observer(state_tensor, action))
        # Update Observer Network
        self.observer_optimizer.zero_grad()
        observer_loss.backward()
        self.observer_optimizer.step()

        # Compute Value Loss
        value_loss = F.mse_loss(states_val, value_targets)
        # Update Value network
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        # Compute Actor Loss
        actor_loss = -(reward + self.critic(self.observer(state_tensor, self.actor(state_tensor)))).mean()
        # print(observer_loss.shape, value_loss.shape, actor_loss.shape)
        # Update Actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # self.memory = []

    def train(self, env, episodes, batch_size = 128):
        self.memory = ReplayBuffer(buffer_size = batch_size)
        returns = []
        for episode in range(episodes):
            score = 0
            done = False
            truncated = False
            state, _ = env.reset()
            while not done and not truncated:
                action = self.actor.select_action(state)
                next_state, reward, done, truncated, info = env.step((action.item(),))
                self.memory.push([state, action, reward, next_state, done])
                if len(self.memory) == batch_size:
                    states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
                    self.learn(states, actions, rewards, next_states, dones)
                # self.push_memory()
                score += reward
                state = next_state
            # print(score)
            returns.append(score)
            plot_return(returns, f'Actor Critic ({device})')
        env.close()
        return returns
