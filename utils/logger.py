import sys
import time
import logging
from datetime import date

import wandb

class Logger():
    def __init__(self, filename, config, mode='train'):
        self.FPS_numerator = config.num_timesteps_per_update
        self.mode = mode
        if mode == 'eval':
            self.data_eval = {
                'Blue Win': [],
                'Red Win': [],
                'Draw': [],
                'Blue Island Dead': [],
                'Blue Catch': [],
                'Timestep': []
            }

        stdout_handler = logging.StreamHandler(sys.stdout)
        file_handler = logging.FileHandler(filename, mode='w')
        logging.basicConfig(level=logging.INFO, handlers=[stdout_handler, file_handler],
                            format='%(asctime)s |%(message)s', datefmt="%m-%d %H:%M:%S")
        self.logger = logging.getLogger()

        self.use_wandb = config.use_wandb if hasattr(config, 'use_wandb') else False
        if self.use_wandb:
            wandb.init(project="USV_MARL", 
                       name=f'{date.today().strftime("%Y%m%d")}_{config.exp_name}', 
                       entity='silab',
                       config={
                           'run_seed': config.run_seed,
                       })
            wandb.define_metric("episode")
            wandb.define_metric("episode/*", step_metric="episode")
            wandb.define_metric("model")
            wandb.define_metric("model/*", step_metric="model")
            self.reset_wandb_aggregator()
        
        self.logger.info(f' Blue-{config.blue_agent_type} vs Red-{config.red_agent_type}')
        self.logger.info(f' Directory: {filename}')
        self.iter_episode, self.iter_model = 1, 1
        self.start = 0.0

    def reset_time(self):
        self.start = time.time()
    
    def reset_wandb_aggregator(self):
        self.wandb_aggregator = {
            'episode/blue win': [],
            'episode/red win': [],
            'episode/draw': [],
            'episode/timestep': [],
            'episode/blue reward': [],
            'episode/blue catch': [],
            'episode/blue attack': [],
            'episode/blue island dead': [],
        }
    
    def aggregate_wandb(self):
        for key in self.wandb_aggregator.keys():
            self.wandb_aggregator[key] = sum(self.wandb_aggregator[key]) / len(self.wandb_aggregator[key])

    def log_episode(self, info):
        if self.use_wandb:
            if len(self.wandb_aggregator['episode/blue win']) == 1000:
                self.aggregate_wandb()
                wandb.log({
                    'episode': self.iter_episode,
                    'episode/blue win': self.wandb_aggregator['episode/blue win'],
                    'episode/red win': self.wandb_aggregator['episode/red win'],
                    'episode/draw': self.wandb_aggregator['episode/draw'],
                    'episode/timestep': self.wandb_aggregator['episode/timestep'],
                    'episode/blue reward': self.wandb_aggregator['episode/blue reward'],
                    'episode/blue catch': self.wandb_aggregator['episode/blue catch'],
                    'episode/blue attack': self.wandb_aggregator['episode/blue attack'],
                    'episode/blue island dead': self.wandb_aggregator['episode/blue island dead'],
                })
                self.reset_wandb_aggregator()
                self.iter_episode += 1
            else:
                self.wandb_aggregator['episode/blue win'].append(info['blue win'])
                self.wandb_aggregator['episode/red win'].append(info['red win'])
                self.wandb_aggregator['episode/draw'].append(info['draw'])
                self.wandb_aggregator['episode/timestep'].append(info['timestep'])
                self.wandb_aggregator['episode/blue reward'].append(info['blue reward'])
                self.wandb_aggregator['episode/blue catch'].append(info['blue catch'])
                self.wandb_aggregator['episode/blue attack'].append(info['blue attack'])
                self.wandb_aggregator['episode/blue island dead'].append(info['blue island dead'])


        if self.mode == 'eval':
            self.data_eval['Blue Win'].append(info['blue win'])
            self.data_eval['Red Win'].append(info['red win'])
            self.data_eval['Draw'].append(info['draw'])
            self.data_eval['Blue Island Dead'].append(info['blue island dead'])
            self.data_eval['Blue Catch'].append(info['blue catch'])
            self.data_eval['Timestep'].append(info['timestep'])
            self.iter_episode += 1

        self.logger.info(f' Episode: {self.iter_episode:5d} | ' + ' | '.join([f'{key}: {item:7.3f}' for key, item in info.items()]))
        
    def log_model(self, train_results_blue=dict):
        time_dict = {'FPS': self.FPS_numerator / (time.time() - self.start)}
        train_results = {**train_results_blue, **time_dict}
        logging = f' Model Update: {self.iter_model:5d} | ' + ' | '.join([f'{key}: {item:7.3f}' for key, item in train_results.items()])
        self.logger.info(logging)

        if self.use_wandb:
            wandb.log({**{'model': self.iter_model},
                       **{f'model/{key}': item for key, item in train_results.items()}})
        self.iter_model += 1
        self.start = time.time()

    def log_evaluation(self):
        num_episodes = len(self.data_eval['Blue Win'])
        if num_episodes == 0:
            self.logger.info("No evaluation data available.")
            return

        result = {
            'Blue Win': sum(self.data_eval['Blue Win']) / num_episodes,
            'Red Win': sum(self.data_eval['Red Win']) / num_episodes,
            'Draw': sum(self.data_eval['Draw']) / num_episodes,
            'Blue Island Dead': sum(self.data_eval['Blue Island Dead']) / num_episodes,
            'Blue Catch': sum(self.data_eval['Blue Catch']) / num_episodes,
            'Avg Timestep': sum(self.data_eval['Timestep']) / num_episodes,
        }

        logging = ' ' + ' | '.join([
            f'{key}: {100 * item:.2f} %' if key in ['Blue Win', 'Red Win', 'Draw'] else
            f'{key}: {item:.2f}'
            for key, item in result.items()
        ])
        self.logger.info(" Evaluation Results:")
        self.logger.info(logging)
        self.logger.info(f" Total Episodes: {num_episodes}")