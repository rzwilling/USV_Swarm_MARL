import os
from os.path import join as pjoin
import time
import argparse
import random


from env.USVEnv import USVEnv
from utils.logger import Logger
from utils.config import Config
from utils.load_model import load_config, load_model, save_config, save_model 
from env.heuristic.policy_red import PigeonRed
from env.heuristic.policy_blue import DDPG, MADDPG, HeuristicBluePolicy, SimpleActorCritic
import torch






if __name__ == '__main__':
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default=f'Test_{random.randint(0, 1000)}')
    parser.add_argument('--num_episode', type=int, default=5000, help='Number of steps to evaluate')
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
    time_start = time.time()
    obs_b, obs_r = env.reset()


    agent_red = PigeonRed(config)

    mc = 4
    if mc == 1:
        agent_blue = SimpleActorCritic(config) #HeuristicBluePolicy(config) # # # #
    elif mc == 2:
        agent_blue = HeuristicBluePolicy(config)
    elif mc == 3:
        agent_blue = DDPG(config)
    elif mc == 4:
        agent_blue = MADDPG(config)




    # Evaluation loop

    #self.blue_x, self.blue_y, self.blue_yaw, self.blue_alive
    cnt = 0
    step = 0

    save_config(config, f'./saved_models/{config.exp_big_name}_{config.exp_name}_config.json') 
 
    while True:
        with torch.no_grad():  # Make a decision
            action_b, curr_graph = agent_blue.get_action(obs_b)
            action_r = agent_red.get_action(obs_r)

        step += 1
        # Environment step
        (next_obs_b, next_obs_r), (reward_b, reward_r), done, _, info = env.step(action_b, action_r)
        state_b, state_r = obs_b
        next_state_b, next_state_r = next_obs_b

        # Check if episode has ended
        episode_done, done_b, _, _ = done

        # Store transition and sample minibatch

        next_graph = agent_blue.compute_next_graph(next_obs_b)
        next_action_r = agent_red.get_action(next_obs_r)

        agent_blue.replay_memory.add_to_buffer(
            curr_graph, action_b, action_r, reward_b, next_graph, next_action_r, done_b)



        # Update the agent
        if mc == 1:
            agent_blue.update(state_b, action_b, reward_b, next_state_b, episode_done)
        elif mc == 3:
            if step % 10 == 0:
                agent_blue.update()
        elif mc == 4:
            if step % 40 == 0:
                agent_blue.update()

        if (step + 1) % 50000 == 0: # 1000
            save_model(agent_blue, f'./saved_models/{config.exp_big_name}_{config.exp_name}_model_{step}.pth')
            #load_model(agent_blue, f'maddpg_model_{step}.pth')
        
        # Assign next state
        obs_b = next_obs_b  
        obs_r = next_obs_r


        for info_ in info:
            if info_['seed'] <= args.num_episode:
                logger.log_episode(info_)
                cnt += 1

        if args.num_episode == cnt:
                break
    


    save_model(agent_blue, f'./saved_models/{config.exp_big_name}_{config.exp_name}_model_final.pth')


    # Calculate and log final evaluation results
    logger.log_evaluation()
    print(f'Evaluation completed. Total time: {time.time() - time_start:.2f} seconds')
    print(f'Results logged in {pjoin(root, "evaluation.log")}')
