import os
from os.path import join as pjoin
import time
import argparse
import random


from env.USVEnv import USVEnv
from utils.logger import Logger
from utils.config import Config
from utils.load_model import load_config, load_model, save_config 
from env.heuristic.policy_red import PigeonRed
from env.heuristic.policy_blue import DDPG, MADDPG, HeuristicBluePolicy, SimpleActorCritic
import torch



    
loaded_config = load_config(r'C:\Users\rvmzw\Documents\GitHub\USV_Swarm_MARL\saved_models\old_w_communication_Test_673_config.json')


step = 100


env = USVEnv(loaded_config)
obs_b, obs_r = env.reset()

agent_red = PigeonRed(loaded_config)
agent_blue = MADDPG(loaded_config)




#load_model(agent_blue, f'maddpg_model_{step}.pth')
load_model(agent_blue, r'C:\Users\rvmzw\Documents\GitHub\USV_Swarm_MARL\results_for_analysis\old_w_comm_team_724_model_49999.pth')

step = 0
while True:
    with torch.no_grad():  # Make a decision
        action_b, curr_graph = agent_blue.get_action(obs_b)
        action_r = agent_red.get_action(obs_r)

    print(action_b)

    step += 1
    # Environment step
    (next_obs_b, next_obs_r), (reward_b, reward_r), done, _, info = env.step(action_b, action_r)
    state_b, state_r = obs_b
    next_state_b, next_state_r = next_obs_b
    

    obs_b = next_obs_b  
    obs_r = next_obs_r

    if step == 10:
        break
