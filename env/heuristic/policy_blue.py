import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import time
torch.autograd.set_detect_anomaly(True)


class ActorNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim * 4 , 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, state_b, state_r):
        if state_b.dim() == 3:
            state_b = state_b.unsqueeze(0)
        if state_r.dim() == 3:
            state_r = state_r.unsqueeze(0)

        state = torch.cat([state_b, state_r], dim=2) 
        state = torch.flatten(state, start_dim=2)
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action_value = self.fc3(x).squeeze(-1)
        return action_value

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, num_blue):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim * 4 + num_blue, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, state_b, state_r, action):
        if state_b.dim() == 3:
            state_b = state_b.unsqueeze(0)
        if state_r.dim() == 3:
            state_r = state_r.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        if action.dim() == 2:
            action = action.unsqueeze(-1)

        state = torch.cat([state_b, state_r], dim=2) 
        state = torch.flatten(state, start_dim=2) 
        x = torch.cat([state, action], dim=-1) 
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        state_value = self.fc3(x).squeeze(-1)
        return state_value


class ReplayMemory:
    def __init__(self, config):
        self.capacity = config.replay_memory_capacity
        self.memory = [None] * self.capacity
        self.pointer = 0
        self.memory_full = False
        self.num_minibatch = config.num_minibatch

    def add_to_buffer(self, state, state_r, action, reward, next_state, next_state_r, done):
        self.memory[self.pointer] = (state, state_r, action, reward, next_state, next_state_r, done)
        self.pointer = (self.pointer + 1) % self.capacity

        if not self.memory_full:
            if self.memory[-1] is not None:
                self.memory_full = True
            else:
                return

        return 
        
    def sample(self):
        length = len(self.memory) if self.memory_full else self.pointer

        indices = np.random.choice(length, self.num_minibatch, replace=False)
        return [self.memory[idx] for idx in indices]
        



class DDPG:
    
    def __init__(self, config):
        self.num_env = config.num_env
        self.num_blue = config.num_blue
        self.num_red = config.num_red
        self.detect_range = config.blue_detect_range
        self.max_turn_rate = config.action_max
        self.gamma = config.gamma 
        self.noise = config.noise
        self.tau = config.tau

        state_dim = config.num_blue + config.num_red
        action_dim = 1 #config.action_dim


        self.actors = []
        self.critics = []
        self.actor_optimizers = []
        self.critic_optimizers = []
        for _ in range(config.num_blue):
            self.actors.append(ActorNetwork(state_dim))
            self.critics.append(CriticNetwork(state_dim, self.num_blue))
            self.actor_optimizers.append(Adam(list(self.actors[-1].parameters()), lr=config.lr_actor)) 
            self.critic_optimizers.append(Adam(list(self.critics[-1].parameters()), lr=config.lr_critic))

        self.replay_memory = ReplayMemory(config)
     
    def get_action(self, state, is_training = True):
        state_blue, state_red = state

        actions = []

        for model in self.actors:
            if is_training:
                noise = torch.normal(mean=0, std= self.noise, size = (1,1))
            else:
                noise = 0 
            actions.append(model(state_blue, state_red) + noise)


        actions_tensor = torch.cat(actions, dim=0)
        actions_tensor = actions_tensor.view(1, -1)

        return actions_tensor
    

    def update(self):
        if not self.replay_memory.memory_full:
            if self.replay_memory.pointer < self.replay_memory.num_minibatch:
                return

        batch = self.replay_memory.sample()
        state, state_r, action, rewards, next_state, next_state_r, done = zip(*batch)


        state = torch.stack(state)
        state_r = torch.stack(state_r)
        #action = torch.tensor(action, dtype=torch.long)
        action = torch.stack(action)

        rewards = torch.stack(rewards) # [40,1,5]
        rewards = rewards.squeeze(1).permute(1, 0).unsqueeze(-1)  # Shape [5, 40, 1]  # Shape [5, 40]

        #reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.stack(next_state)
        next_state_r = torch.stack(next_state_r)
        done = torch.tensor([d[0] for d in done], dtype=torch.float)
        done = done.unsqueeze(1) 

        next_actions = []
        curr_actions = []
        for actor_i in self.actors:
            next_actions.append(actor_i(next_state, next_state_r))
            curr_actions.append(actor_i(state, state_r))


        next_actions_tensor = torch.stack(next_actions, dim=-1)  # Shape will be [40, 1, 5]
        curr_actions_tensor = torch.stack(curr_actions, dim=-1)  # Shape will be [40, 1, 5]

        

        for i, (actor_i, critic_i, act_optimizer_i, cri_optimizer_i, reward) in enumerate(zip(self.actors, self.critics, self.actor_optimizers, self.critic_optimizers, rewards)):

            next_actions_tensor = torch.stack(next_actions, dim=-1)  # Shape will be [40, 1, 5]
            curr_actions_tensor = torch.stack(curr_actions, dim=-1)  # Shape will be [40, 1, 5]

            state_value = critic_i(state, state_r, action)
            next_state_value = critic_i(next_state, next_state_r, next_actions_tensor)

            target_value = reward + (1 - done) * self.gamma * next_state_value

            # Compute critic loss (Mean Squared Error)
            critic_loss = nn.MSELoss()(state_value, target_value)

            # Update critic
            cri_optimizer_i.zero_grad()
            critic_loss.backward(retain_graph=True)
            cri_optimizer_i.step()

            # Compute actor loss (negative of expected return)
            actor_loss = -critic_i(state, state_r, curr_actions_tensor).mean()

            # Update actor
            act_optimizer_i.zero_grad()
            actor_loss.backward(retain_graph=True)
            act_optimizer_i.step()

            next_actions[i] = actor_i(next_state, next_state_r)
            curr_actions[i] = actor_i(state, state_r)



        return        



class SimpleActorCritic:
    
    def __init__(self, config):
        self.num_env = config.num_env
        self.num_blue = config.num_blue
        self.num_red = config.num_red
        self.detect_range = config.blue_detect_range
        self.max_turn_rate = config.action_max
        self.gamma = config.gamma 

        state_dim = 4 #config.state_dim
        action_dim = 1 #config.action_dim

        self.actor = ActorNetwork(state_dim)
        self.critic = CriticNetwork(state_dim)
        self.optimizer_actor = Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=config.lr_actor)
        self.optimizer_critic = Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=config.lr_critic)
        self.replay_memory = ReplayMemory(config)
     

    def get_action(self, state):
        state_blue, state_red = state

        action = self.actor(state_blue)
        return action
    

    def update(self, state, state_r, action, reward, next_state, next_state_r, done):

        state_value = self.critic(state, state_r, action)
        next_state_action = self.actor(next_state, next_state_r)
        next_state_value = self.critic(next_state, next_state_r, next_state_action)

        done = done.float()

        target_value = reward + (1 - done) * self.gamma * next_state_value
        advantage = target_value - state_value

        # Compute critic loss (Mean Squared Error)
        critic_loss = nn.MSELoss()(state_value, target_value)

        # Update critic
        self.optimizer_critic.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.optimizer_critic.step()


        # Compute actor loss (negative of expected return)
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Update actor
        self.optimizer_actor.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.optimizer_actor.step()




        # # Update Actor Network
        # actor_loss = (-torch.log(self.actor(state).gather(1, action)) * advantage.detach()).mean()
        # self.optimizer_actor.zero_grad()
        # actor_loss.backward()
        # self.optimizer_actor.step()

        # # Update Critic Network
        # critic_loss = advantage.pow(2).mean()
        # self.optimizer_critic.zero_grad()
        # critic_loss.backward()
        # self.optimizer_critic.step()

        return        
 

class HeuristicBluePolicy:
    
    def __init__(self, config):
        self.num_env = config.num_env
        self.num_blue = config.num_blue
        self.num_red = config.num_red
        self.detect_range = config.blue_detect_range
        self.max_turn_rate = config.action_max

    def get_action(self, state):
        state_blue, state_red = state

        distances = torch.norm(state_blue[:, :, :2].unsqueeze(2) - state_red[:, :, :2].unsqueeze(1), dim=3)
        distances[distances > self.detect_range] = float('inf')
        red_alive = state_red[:, :, -1]
        distances.transpose(2, 1)[red_alive == 0.] = float('inf')
        closest_red = torch.argmin(distances, dim=2)

        target_directions = state_red[torch.arange(self.num_env).unsqueeze(1), closest_red, :2] - state_blue[:, :, :2]
        target_angles = torch.atan2(target_directions[:, :, 1], target_directions[:, :, 0])

        angle_diff = target_angles - state_blue[:, :, 2]
        angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))

        turn_rate = torch.clamp(angle_diff, -self.max_turn_rate, self.max_turn_rate)

        return turn_rate / self.max_turn_rate
    


