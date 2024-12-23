

import pickle
import torch

from utils.config import Config

def save_model(model, file_path):
    # Speichere die state_dicts der Modelle und Optimizer
    state = {
        'actor_gnn': model.actor_gnn.state_dict(),
        'actor_nn': model.actor_nn.state_dict(),
        'critic_gnn': model.critic_gnn.state_dict(),
        'critic_nn': model.critic_nn.state_dict(),
        'actor_optimizer': model.actor_optimizer.state_dict(),
        'critic_optimizer': model.critic_optimizer.state_dict(),
        'config': model.__dict__,  # Optional: Speichere die Konfiguration
    }
    
    # Speichere die Datei
    torch.save(state, file_path)
    print(f"Modelle und Optimizer gespeichert in {file_path}")

    # Speichere die ReplayMemory
    with open(file_path[:-3] + '_pkl.pkl', 'wb') as f:
        pickle.dump(model.replay_memory, f)


def load_model(model, file_path):
    # Lade den gespeicherten Zustand
    state = torch.load(file_path)
    
    # Lade die state_dicts in die Modelle
    model.actor_gnn.load_state_dict(state['actor_gnn'])
    model.actor_nn.load_state_dict(state['actor_nn'])
    model.critic_gnn.load_state_dict(state['critic_gnn'])
    model.critic_nn.load_state_dict(state['critic_nn'])
    
    # Lade die state_dicts in die Optimizer
    model.actor_optimizer.load_state_dict(state['actor_optimizer'])
    model.critic_optimizer.load_state_dict(state['critic_optimizer'])
    
    # Optional: Konfiguration wiederherstellen
    model.__dict__.update(state['config'])
    
    with open(file_path[:-3] + '_pkl.pkl', 'rb') as f:
        model.replay_memory = pickle.load(f)

    #print(model.replay_memory.memory)

    print(f"Modelle und Optimizer aus {file_path} geladen")


import json
import os

def save_config(config, file_path):
    # Konvertiere die Config in ein Dictionary und speichere es als JSON

    directory = os.path.dirname(file_path)
    
    # Falls das Verzeichnis nicht existiert, erstelle es
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_path, 'w') as f:
        json.dump(config.__dict__, f, indent=4)
    print(f"Config gespeichert in {file_path}")

    

def load_config(file_path):
    # Lade die JSON-Datei und erstelle eine neue Config-Instanz
    with open(file_path, 'r') as f:
        config_dict = json.load(f)
    config = Config()
    config.__dict__.update(config_dict)
    print(f"Config geladen aus {file_path}")
    return config


