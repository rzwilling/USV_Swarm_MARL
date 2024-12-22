import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.data import Data
import time

from env.utils.gnn_utils import build_edge_index
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


class MessageUpdateFunction(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MessageUpdateFunction, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, node_features, edge_features):
        # Kombiniere Knotenmerkmale und Kantenmerkmale
        combined = torch.cat([node_features, edge_features], dim=-1)
        return self.fc(combined)

# Knoten-Update-Funktion für das Zustands-Embedding
class NodeUpdateFunction(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NodeUpdateFunction, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, node_features, aggregated_messages):
        # Kombiniere Knotenmerkmale und aggregierte Nachrichten
        combined = torch.cat([node_features, aggregated_messages], dim=-1)
        return self.fc(combined)

# GNN Layer für die Kantenaktualisierung und Knotenaktualisierung
class GNNLayer(MessagePassing):
    def __init__(self, in_channels_node, in_channels_edge, out_channels_node):
        super(GNNLayer, self).__init__(aggr='mean')  # Mittelwertaggregation für die Nachrichten

        self.message_function = MessageUpdateFunction(in_channels_node + in_channels_edge, out_channels_node)
        self.node_update_function = NodeUpdateFunction(in_channels_node + out_channels_node, out_channels_node)

    def forward(self, x, edge_index, edge_attr):
        # Knotenfeatures und Kantenfeatures weitergeben
        out_edge = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        return out_edge

    def message(self, x_j, edge_attr):
        # Nachrichten von den Nachbarknoten (edge_attr sind die Kantenfeatures)
        return self.message_function(x_j, edge_attr)

    def update(self, aggr_out, x):
        # Aggregierte Nachrichten und Knotenmerkmale kombinieren und aktualisieren
        return self.node_update_function(x, aggr_out)

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_channels, 64)  # Erste Schicht
        self.fc2 = nn.Linear(64, 64)  # Zweite Schicht
        self.fc3 = nn.Linear(64, out_channels)  # Ausgabe-Schicht
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # Keine Aktivierungsfunktion am Ausgang, falls es sich um ein Regressionsproblem handelt
        return x


class GNNWithMLP(nn.Module):
    def __init__(self, num_features, num_edge_features, out_channels_node, mlp_out_channels):
        super(GNNWithMLP, self).__init__()
        # Initialisiere GNN Layer
        self.gnn_layer = GNNLayer(in_channels_node=num_features, in_channels_edge=num_edge_features, out_channels_node=out_channels_node)
        
        # Initialisiere das MLP, das die Ausgabe des GNN verarbeitet
        self.mlp = MLP(in_channels=out_channels_node, out_channels=mlp_out_channels)

    def forward(self, x, edge_index, edge_attr):
        # Berechne die Ausgabe des GNN
        gnn_output = self.gnn_layer(x, edge_index, edge_attr)
        
        # Wende das MLP auf die GNN-Ausgabe an
        output = self.mlp(gnn_output)
        
        return output
    
class GNNWithMLPCritic(nn.Module):
    def __init__(self, num_features, num_edge_features, out_channels_node, mlp_out_channels):
        super(GNNWithMLP, self).__init__()
        # Initialisiere GNN Layer
        self.gnn_layer = GNNLayer(in_channels_node=num_features, in_channels_edge=num_edge_features, out_channels_node=out_channels_node)
        
        # Initialisiere das MLP, das die Ausgabe des GNN verarbeitet
        self.mlp = MLP(in_channels=out_channels_node, out_channels=mlp_out_channels)

    def forward(self, x, edge_index, edge_attr):
        # Berechne die Ausgabe des GNN
        gnn_output = self.gnn_layer(x, edge_index, edge_attr)
        
        # Wende das MLP auf die GNN-Ausgabe an
        output = self.mlp(gnn_output)
        
        return output

class GNNActor(nn.Module):
    def __init__(self, feat_per_node, action_dim, hidden_dim = 64):
        super(GNNActor, self).__init__()
        # GNN 
        self.conv1 = GCNConv(feat_per_node, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # MLP 
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, data):
        # if state_b.dim() == 3:
        #     state_b = state_b.unsqueeze(0)
        # if state_r.dim() == 3:
        #     state_r = state_r.unsqueeze(0)

        # state = torch.cat([state_b, state_r], dim=2) 
        # state = torch.flatten(state, start_dim=2)

        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action_value = self.fc3(x).squeeze(-1)

        return action_value


class GNNCritic(nn.Module):
    def __init__(self, feat_per_node, action_dim, hidden_dim = 64):
        super(GNNActor, self).__init__()
        # GNN 
        self.conv1 = GCNConv(feat_per_node, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # MLP 
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, data):
        # if state_b.dim() == 3:
        #     state_b = state_b.unsqueeze(0)
        # if state_r.dim() == 3:
        #     state_r = state_r.unsqueeze(0)

        # state = torch.cat([state_b, state_r], dim=2) 
        # state = torch.flatten(state, start_dim=2)

        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action_value = self.fc3(x).squeeze(-1)

        return action_value


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
        

class MADDPG:
   
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

        self.communication_range = 15 
        self.observation_range = config.blue_detect_range
        self.attack_range = config.attack_range


        num_features = 4
        num_edge_features = 3
        out_channels_node = 16
        mlp_out_channels = 1

        # self.actor = GNNWithMLP(num_features=num_features, num_edge_features=num_edge_features, 
        #           out_channels_node=out_channels_node, mlp_out_channels=mlp_out_channels)

        #output = model(x, edge_index, edge_attr)


        self.actors = []
        self.critics = []
        self.actor_optimizers = []
        self.critic_optimizers = []
        for _ in range(config.num_blue):
            self.actors.append(GNNWithMLP(num_features=num_features, num_edge_features=num_edge_features, 
                  out_channels_node=out_channels_node, mlp_out_channels=mlp_out_channels))
            self.critics.append(GNNWithMLP(num_features=num_features, num_edge_features=num_edge_features, 
                  out_channels_node=out_channels_node, mlp_out_channels=mlp_out_channels))  
            self.actor_optimizers.append(Adam(list(self.actors[-1].parameters()), lr=config.lr_actor))
            self.critic_optimizers.append(Adam(list(self.critics[-1].parameters()), lr=config.lr_critic))

        self.replay_memory = ReplayMemory(config)
     
    def get_action(self, state, is_training = True):
        state_blue, state_red = state


        # Combine the node features into a single tensor
        # Flatten the node features
        state_b = state_blue.view(-1, state_blue.shape[-1])  # Shape: [2, 4]
        state_r = state_red.view(-1, state_red.shape[-1])  # Shape: [1, 4]

        x = torch.cat([state_b, state_r], dim=0)  # Shape: [3, 4]


        edge_index, edge_attr = build_edge_index(x, self.communication_range, self.observation_range, self.attack_range)

        #self.actor(x, edge_index, edge_attr)
        #breakpoint()


        # x (num_nodes, num_features) oder (batch_size, num_nodes, num_features)
        # edge_index (2, num_edges), wobei num_edges die Anzahl der Kanten im Graphen ist.
        # edge_attr: Dies sind die Merkmale der Kanten im Graphen (Edge Features). (num_edges, num_edge_features), wobei num_edge_features

        actions = []

        for i, model in enumerate(self.actors):
            if is_training:
                noise = torch.normal(mean=0, std= self.noise, size = (1,1))
            else:
                noise = 0 
 
            actions.append(model(x, edge_index, edge_attr)[i] + noise)

        concatenated_tensor = torch.cat(actions, dim=1)

        return concatenated_tensor
    

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
    


