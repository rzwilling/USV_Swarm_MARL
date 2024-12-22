import os
from os.path import join as pjoin
import time
import argparse
import random
import torch

from env.USVEnv import USVEnv
from utils.logger import Logger
from utils.config import Config
from env.heuristic.policy_red import PigeonRed
from env.heuristic.policy_blue import DDPG, HeuristicBluePolicy, SimpleActorCritic


if __name__ == '__main__':
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default=f'Test_{random.randint(0, 1000)}')
    parser.add_argument('--num_episode', type=int, default=20000, help='Number of steps to evaluate')
    parser.add_argument('--red_policy_mode', type=int, default=1, help='0: Aggressive Opponent / 1: Conservative Opponent')
    parser.add_argument('--model_path', type=str, default='results/base_744/model/model_6001.th')
    args = parser.parse_args()

    # Load configuration
    config = Config()
    config.exp_name = args.exp_name
    config.red_policy_mode = args.red_policy_mode
    config.train_model_mode = False
    config.num_env = 1

    # Set up logging
    root = pjoin("results", config.exp_name)
    os.makedirs(root, exist_ok=True)
    logger = Logger(filename=pjoin(root, 'evaluation.log'), config=config, mode='eval')
    config.root_path = root

    # Set up environment and agents
    env = USVEnv(config)
    mc = 3
    if mc == 1:
        agent_blue = SimpleActorCritic(config) #HeuristicBluePolicy(config) # # # #
    elif mc == 2:
        agent_blue = HeuristicBluePolicy(config)
    elif mc == 3:
        agent_blue = DDPG(config)

    agent_red = PigeonRed(config)

    # Evaluation loop
    time_start = time.time()
    obs_b, obs_r = env.reset()
    cnt = 0
    step = 0

    while True:
        with torch.no_grad():  # Make a decision
            action_b = agent_blue.get_action(obs_b)
            action_r = agent_red.get_action(obs_r)

        step += 1
        # Environment step
        (next_obs_b, next_obs_r), (reward_b, reward_r), done, _, info = env.step(action_b, action_r)
        state_b, state_r = obs_b
        next_state_b, next_state_r = next_obs_b


        # Check if episode has ended
        episode_done, _, _, _ = done

        # Store transition and sample minibatch
        agent_blue.replay_memory.add_to_buffer(state_b, state_r, action_b, reward_b, next_state_b, next_state_r, done)

        # Update the agent
        if mc == 1:
            agent_blue.update(state_b, action_b, reward_b, next_state_b, episode_done)
        elif mc == 3:
            if step % 100 == 0:
                agent_blue.update()

        
        # Assign next state
        obs_b = next_obs_b  
        obs_r = next_obs_r


        for info_ in info:
            if info_['seed'] <= args.num_episode:
                logger.log_episode(info_)
                cnt += 1

        if args.num_episode == cnt:
                break
    
    # Calculate and log final evaluation results
    logger.log_evaluation()
    print(f'Evaluation completed. Total time: {time.time() - time_start:.2f} seconds')
    print(f'Results logged in {pjoin(root, "evaluation.log")}')
