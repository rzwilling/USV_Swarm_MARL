import torch
import torch.nn as nn
from torch.optim import Adam

class ActorNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action_value = self.fc3(x).squeeze(-1)
        return action_value

class CriticNetwork(nn.Module):
    def __init__(self, state_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + 1, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
    
    def forward(self, state, action):
        action = action.unsqueeze(-1)
        x = torch.cat([state, action], dim=-1) 
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        state_value = self.fc3(x).squeeze(-1)
        return state_value

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
     

    def get_action(self, state):
        state_blue, state_red = state

        action = self.actor(state_blue)
        return action
    

    def update(self, state, action, reward, next_state, done):

        state_value = self.critic(state, action)
        next_state_action = self.actor(next_state)
        next_state_value = self.critic(next_state, next_state_action)

        done = done.float()

        target_value = reward + (1 - done) * self.gamma * next_state_value
        advantage = target_value - state_value

        # Compute critic loss (Mean Squared Error)
        critic_loss = nn.MSELoss()(state_value, target_value)

        # Update critic
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()


        # Compute actor loss (negative of expected return)
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Update actor
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
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
    


