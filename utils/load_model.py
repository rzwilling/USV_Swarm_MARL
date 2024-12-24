

import pickle
import torch

import json
import os

from utils.config import Config

def save_model(model, file_path):
    state = {
        'actor_gnn': model.actor_gnn.state_dict(),
        'actor_nn': model.actor_nn.state_dict(),
        'critic_gnn': model.critic_gnn.state_dict(),
        'critic_nn': model.critic_nn.state_dict(),
        'actor_optimizer': model.actor_optimizer.state_dict(),
        'critic_optimizer': model.critic_optimizer.state_dict(),
        'config': model.__dict__, 
    }
    
    torch.save(state, file_path)
    with open(file_path[:-3] + '_pkl.pkl', 'wb') as f:
        pickle.dump(model.replay_memory, f)


def load_model(model, file_path):
    state = torch.load(file_path)

    model.actor_gnn.load_state_dict(state['actor_gnn'])
    model.actor_nn.load_state_dict(state['actor_nn'])
   
    model.critic_gnn.load_state_dict(state['critic_gnn'])
    model.critic_nn.load_state_dict(state['critic_nn'])
   
    model.actor_optimizer.load_state_dict(state['actor_optimizer'])
    model.critic_optimizer.load_state_dict(state['critic_optimizer'])

    model.__dict__.update(state['config'])

    #with open(file_path[:-3] + '_pkl.pkl', 'rb') as f:
    #    model.replay_memory = pickle.load(f)

    #print(model.replay_memory.memory)




def save_config(config, file_path):

    directory = os.path.dirname(file_path)
    
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_path, 'w') as f:
        json.dump(config.__dict__, f, indent=4)

    

def load_config(file_path):
    with open(file_path, 'r') as f:
        config_dict = json.load(f)
    config = Config()
    config.__dict__.update(config_dict)
    return config


