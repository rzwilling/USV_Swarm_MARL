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
from env.heuristic.policy_blue import HeuristicBluePolicy


if __name__ == '__main__':
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default=f'Test_{random.randint(0, 1000)}')
    parser.add_argument('--num_episode', type=int, default=10, help='Number of steps to evaluate')
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
    agent_blue = HeuristicBluePolicy(config)
    agent_red = PigeonRed(config)

    # Evaluation loop
    time_start = time.time()
    obs_b, obs_r = env.reset()
    cnt = 0

    while True:
        with torch.no_grad():  # Make a decision
            action_b = agent_blue.get_action(obs_b)
            action_r = agent_red.get_action(obs_r)
        
        # Environment step
        (obs_b, obs_r), _, done, _, info = env.step(action_b, action_r)
        
        # Check if episode has ended
        episode_done = done.any().item() if isinstance(done, torch.Tensor) else done

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
