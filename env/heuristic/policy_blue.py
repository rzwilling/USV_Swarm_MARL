import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.nn import GCNConv, MessagePassing, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import time

from env.utils.gnn_utils import build_edge_index
torch.autograd.set_detect_anomaly(True)




class Node:
    def __init__(self, node_id, node_type, features):
        type_dict = {"agent": 0, "adversarial": 1, "land": 2}

        self.node_id = node_id
        self.node_type = type_dict[node_type]  # 'agent', 'adversarial', 'land'
        self.x = features[0]          # Only for agent nodes
        self.y = features[1]          # Only for agent nodes
        self.yaw = features[2]
        self.alive = features[3]

    def to_tensor(self):
        # Return a tensor of node features (x, y, yaw, alive)
        return torch.tensor([self.node_type, self.x, self.y, self.yaw, self.alive], dtype=torch.float)


def initialize_edge_features(nodes):

    edge_index = []
    edge_attr = []
    
    # Loop through all pairs of nodes to create edges
    for i, node_i in enumerate(nodes):
        for j, node_j in enumerate(nodes):
            if i != j:
                # Compute distance, bearing, and relative orientation between node_i and node_j
                distance = np.linalg.norm(np.array([node_i.x, node_i.y]) - np.array([node_j.x, node_j.y]))
                bearing = np.arctan2(node_j.y - node_i.y, node_j.x - node_i.x) - node_i.yaw
                relative_orientation =np.arctan2(node_i.y - node_j.y, node_i.x - node_j.x) - node_j.yaw
                
                edge_feature = torch.tensor([distance, bearing, relative_orientation], dtype=torch.float)

                edge_index.append([node_i.node_id, node_j.node_id])
                edge_attr.append(edge_feature)

      # Convert to PyTorch tensors
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # Transpose to match PyTorch Geometric format
    edge_attr = torch.stack(edge_attr, dim=0)  # Stack the edge features into a tensor
                  
    
    return edge_index, edge_attr


class MessageUpdateFunction(nn.Module):
    def __init__(self, in_channels, in_channels_edge, out_channels):
        super(MessageUpdateFunction, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels * 2 + in_channels_edge, 32),  # Input size is the sum of features from vi, vj, and eji
            torch.nn.ReLU(),
            torch.nn.Linear(32, out_channels)  # Output size is the node feature dimension
        )
    def forward(self, v_i, v_j, e_ij):
        # Kombiniere Knotenmerkmale und Kantenmerkmale
        combined = torch.cat([v_i, v_j, e_ij], dim=-1)
        return self.mlp(combined)
    
class MessageUpdateFunction2(nn.Module):
    def __init__(self, in_channels, in_channels_edge, out_channels):
        super(MessageUpdateFunction2, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels * 2 + in_channels_edge, 32),  # Input size is the sum of features from vi, vj, and eji
            torch.nn.ReLU(),
            torch.nn.Linear(32, out_channels)  # Output size is the node feature dimension
        )
    def forward(self, v_i, v_j, e_ij):
        # Kombiniere Knotenmerkmale und Kantenmerkmale
        combined = torch.cat([v_i, v_j, e_ij], dim=-1)
        return self.mlp(combined)


# Knoten-Update-Funktion für das Zustands-Embedding
class NodeUpdateFunction(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NodeUpdateFunction, self).__init__()
        self.fc = nn.Linear(in_channels + out_channels, out_channels)

    def forward(self, v_i, aggregated_messages):
        # Kombiniere Knotenmerkmale und aggregierte Nachrichten
        combined = torch.cat([v_i, aggregated_messages], dim=-1)
        return self.fc(combined)



class GNNModel(nn.Module):
    def __init__(self, config, in_channels, out_channels):
        super(GNNModel, self).__init__()
        self.conv1 = GNNEmbeddingLayer(config, in_channels, 3, out_channels)  # Erste Convolution-Schicht
        self.comm_layer = config.comm_layer
        self.conv2 = nn.Linear(16*3 + 5, 16)

        if self.comm_layer:
            self.conv3 = GNNCommunicationLayer(config, in_channels + 16, 3, out_channels)  # Erste Convolution-Schicht
            self.conv4 = nn.Linear(16 + 5, 16)
        #self.conv2 = GCNConv(16, out_channels)  # Zweite Convolution-Schicht

    def forward(self, data):

        # Erste Convolution
        x = self.conv1(data)
        x = torch.relu(x)  # Aktivierungsfunktion
        x = self.conv2(x)

        # Zweite Convolution
        if self.comm_layer:
            x = self.conv3(x, data)

            x = torch.relu(x)
            x = self.conv4(x)
        
        return x



# GNN Layer für die Kantenaktualisierung und Knotenaktualisierung
class GNNEmbeddingLayer(MessagePassing):
    def __init__(self, config, in_channels_node, in_channels_edge, out_channels_node):
        super(GNNEmbeddingLayer, self).__init__(aggr='sum') 

        self.message_function = MessageUpdateFunction(in_channels_node, in_channels_edge, out_channels_node)
        #self.node_update_function = NodeUpdateFunction(in_channels_node, out_channels_node)
        self.observation_range = config.blue_detect_range
        self.attack_range = config.attack_range

    def forward(self, data):
        # Knotenfeatures und Kantenfeatures weitergeben
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        #agent_idx = [i for i, row in enumerate(x) if row[0] == 0]
        advent_idx = [i for i, row in enumerate(x) if row[0] == 1]
        island_idx = [i for i, row in enumerate(x) if row[0] == 2]

        #agent_mask = torch.isin(edge_index[1], torch.tensor(agent_idx))
        advent_mask = torch.isin(edge_index[1], torch.tensor(advent_idx))
        island_mask = torch.isin(edge_index[1], torch.tensor(island_idx))


        agent_output = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        advent_output = self.propagate(edge_index[:, advent_mask], x=x, edge_attr=edge_attr[advent_mask, :])
        island_output = self.propagate(edge_index[:, island_mask], x=x, edge_attr=edge_attr[island_mask, :])

        result = torch.cat((x, agent_output, advent_output, island_output), dim=1)

        return result

    def message(self, x_j, x_i, edge_attr):
        # Nachrichten von den Nachbarknoten (edge_attr sind die Kantenfeatures)
        #distance, bearing, relative_orientation = edge_attr

        message = self.message_function(x_i, x_j, edge_attr)

        distance = edge_attr[:, 0]
        node_type = x_j[:, 0]
        min_node_type = torch.min(node_type)

        mask = torch.ones_like(distance, dtype=torch.bool) 

        if min_node_type == 0:
            mask = distance < self.observation_range
        elif min_node_type == 1:
            mask = distance < self.attack_range
        else:
            mask = True

        return mask.view(-1, 1) * message 


   
#GNN Layer für die Kantenaktualisierung und Knotenaktualisierung
class GNNCommunicationLayer(MessagePassing):
    def __init__(self, config, in_channels_node, in_channels_edge, out_channels_node):
        super(GNNCommunicationLayer, self).__init__(aggr='sum') 

        self.message_function = MessageUpdateFunction2(in_channels_node, in_channels_edge, out_channels_node)
        #self.node_update_function = NodeUpdateFunction(in_channels_node, out_channels_node)
        self.observation_range = config.blue_detect_range
        self.attack_range = config.attack_range
        self.communication_range = config.communication_range

    def forward(self, output_x, data):
        # Knotenfeatures und Kantenfeatures weitergeben

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        z = torch.cat((data.x,output_x), dim=1)


        agent_idx = [i for i, row in enumerate(x) if row[0] == 0]
        agent_mask = torch.isin(edge_index[1], torch.tensor(agent_idx))
        agent_output = self.propagate(edge_index[:, agent_mask], x = z, edge_attr=edge_attr[agent_mask, :])

        result = torch.cat((x, agent_output), dim=1)

        return result

    def message(self, x_j, x_i, edge_attr):
        # Nachrichten von den Nachbarknoten (edge_attr sind die Kantenfeatures)
        #distance, bearing, relative_orientation = edge_attr

        message = self.message_function(x_i, x_j, edge_attr)
        distance = edge_attr[:, 0]

        mask = torch.ones_like(distance, dtype=torch.bool) 

        mask = distance < self.communication_range

        return mask.view(-1, 1) * message 

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

class MLP_Q(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLP_Q, self).__init__()
        self.fc1 = nn.Linear(in_channels + 1, 64)  # Erste Schicht
        self.fc2 = nn.Linear(64, 64)  # Zweite Schicht
        self.fc3 = nn.Linear(64, out_channels)  # Ausgabe-Schicht
        self.relu = nn.ReLU()

    def forward(self, x, y):

        x = torch.cat([x, y.T], dim=1) 
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # Keine Aktivierungsfunktion am Ausgang, falls es sich um ein Regressionsproblem handelt
        return x




class ReplayMemory:
    def __init__(self, config):
        self.capacity = config.replay_memory_capacity
        self.memory = [None] * self.capacity
        self.pointer = 0
        self.memory_full = False
        self.num_minibatch = config.num_minibatch

    def add_to_buffer(self, curr_graph, action_b, action_r, reward_b, next_obs_b, next_action_r, episode_done):
        self.memory[self.pointer] = curr_graph, action_b, action_r, reward_b, next_obs_b, next_action_r, episode_done
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
        self.island_position = config.island_position


        num_features = 4
        num_edge_features = 3
        out_channels_node = 16
        mlp_out_channels = 1
        self.team_reward_layer = config.team_reward_layer


        self.actor_gnn = GNNModel(config, num_features + 1, out_channels_node)
        self.actor_nn = MLP(in_channels=out_channels_node, out_channels=mlp_out_channels)
        self.critic_gnn = GNNModel(config, num_features + 1, out_channels_node)
        self.critic_nn = MLP_Q(in_channels=out_channels_node, out_channels=mlp_out_channels)
        self.actor_optimizer = Adam(list(self.actor_nn.parameters()) + list(self.actor_gnn.parameters()), lr=config.lr_actor)
        self.critic_optimizer = Adam(list(self.critic_nn.parameters()) + list(self.critic_gnn.parameters()), lr=config.lr_critic)

        self.replay_memory = ReplayMemory(config)
     
    def get_action(self, observation, is_training = True):


        curr_nodes = [Node(i, "agent", observation[0][0][i]) for i in range(len(observation[0][0]))]
        curr_nodes += [Node(i + self.num_blue, "adversarial", observation[1][0][i]) for i in range(len(observation[1][0]))]
        curr_nodes += [Node(self.num_blue + self.num_red, "land", self.island_position + [0,1])]

        edge_index, edge_attr = initialize_edge_features(curr_nodes)

        node_tensors = torch.stack([node.to_tensor() for node in curr_nodes])

        data = Data(x=node_tensors, edge_index=edge_index, edge_attr=edge_attr)

        # Update Graph NN
        embedding = self.actor_gnn(data)

        if is_training:
            noise = torch.normal(mean=0, std= self.noise, size = (1, self.num_blue)) # To Do
        else:
            noise = 0 

        # Hand over to MLP

        actions = self.actor_nn(embedding[:self.num_blue]).T + noise 

        return actions, data
    
    def compute_next_graph(self, next_obs_b):
        
        # Get next graph
        next_nodes = [Node(i, "agent", next_obs_b[0][0][i]) for i in range(len(next_obs_b[0][0]))]
        next_nodes += [Node(i + self.num_blue, "adversarial", next_obs_b[1][0][i]) for i in range(len(next_obs_b[1][0]))]
        next_nodes += [Node(self.num_blue + self.num_red, "land", self.island_position + [0,1])]

        next_edge_index, next_edge_attr = initialize_edge_features(next_nodes)
        next_node_tensors = torch.stack([node.to_tensor() for node in next_nodes])
        next_data = Data(x=next_node_tensors, edge_index=next_edge_index, edge_attr=next_edge_attr)

        return next_data


    def update(self):
        # Enough data stored to update
        if not self.replay_memory.memory_full:
            if self.replay_memory.pointer < self.replay_memory.num_minibatch:
                return
            
        # Get sample
        data_loader = self.replay_memory.sample()


        critic_loss_list = []
        actor_loss_list = []

        for data in data_loader:
            curr_graph, action_b, action_r, reward_b, next_graph, next_action_r, done = data

            done = done.int()


            target_value = reward_b.T + (1 - done.T) * self.gamma * self.critic_nn(self.critic_gnn(next_graph)[:self.num_blue], (self.actor_nn(self.actor_gnn(next_graph)[:self.num_blue])).T )

            if self.team_reward_layer:
                total_sum = reward_b.sum()
                target_value += 0.2 * total_sum

            critic_loss = nn.MSELoss()(self.critic_nn(self.critic_gnn(curr_graph)[:self.num_blue], action_b), target_value)
            critic_loss_list.append(critic_loss)

        total_critic_loss = torch.stack(critic_loss_list).mean()

        # Update critic network
        self.critic_optimizer.zero_grad()
        total_critic_loss.backward()
        self.critic_optimizer.step()

        for data in data_loader:
            curr_graph, action_b, action_r, reward_b, next_graph, next_action_r, done = data

            done = done.int()


            actor_loss = -self.critic_nn(self.critic_gnn(curr_graph),  self.actor_nn(self.actor_gnn(curr_graph)).T).mean()
            

            actor_loss_list.append(actor_loss)


        # Accumulate losses

        total_actor_loss = torch.stack(actor_loss_list).mean()


        # Update actor network
        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        self.actor_optimizer.step()


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
