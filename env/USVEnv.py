import os
from os.path import join as pjoin
import shutil

import torch
import numpy as np

from env.utils.utils_env import *
from env.utils.USVEnvVisualizer import USVEnvVisualizer, USVEnvVisualizer_traj

class USVEnv():

    def __init__(self, config):

        self.num_env = config.num_env
        self.num_blue = config.num_blue
        self.num_red = config.num_red
        self.max_step = config.max_step
        self.end_condition_red_touch = config.end_condition_red_touch
        self.action_min = config.action_min
        self.action_max = config.action_max
        self.action_scale = config.action_scale
        self.attack_range = config.attack_range
        self.attack_angle = config.attack_angle
        self.blue_velocity = config.blue_velocity
        self.red_velocity = config.red_velocity
        self.red_hp_max = config.red_hp_max
        self.blue_agent_type = config.blue_agent_type
        self.red_agent_type = config.red_agent_type
        self.random_seed = config.random_seed
        self.episode_num = 0
        self.red_policy_mode = config.red_policy_mode

        ############## Rendering
        self.do_visualize_online = config.do_visualize_online
        self.do_visualize_traj = config.do_visualize_traj
        self.do_visualize_traj_freq = config.do_visualize_traj_freq
        self.island_position = config.island_position
        self.island_radius_plot = config.island_radius
        self.map_size = config.map_size
        if self.do_visualize_online:
            self.visualizer = USVEnvVisualizer(self.num_blue, self.num_red, self.island_position, self.island_radius_plot)
        if self.do_visualize_traj:
            self.visualizer_traj_data = []
            self.visualizer_traj_iter = 0
            self.visualizer_traj_root_path = pjoin(config.root_path, 'images')
            if os.path.exists(self.visualizer_traj_root_path):
                shutil.rmtree(self.visualizer_traj_root_path)
            os.makedirs(self.visualizer_traj_root_path)

        ############## Blue info initailization
        self.blue_x = torch.zeros((self.num_env, self.num_blue))
        self.blue_y = torch.zeros((self.num_env, self.num_blue))
        self.blue_yaw = torch.zeros((self.num_env, self.num_blue))
        self.blue_hp = torch.zeros((self.num_env, self.num_blue))
        self.blue_alive = torch.zeros((self.num_env, self.num_blue), dtype=torch.bool)
        self.blue_dead_moment = torch.zeros((self.num_env, self.num_blue), dtype=torch.bool)
        self.blue_valid = torch.zeros((self.num_env, self.num_blue), dtype=torch.bool)
        
        ############## Red info initailization
        self.red_x = torch.zeros((self.num_env, self.num_red))
        self.red_y = torch.zeros((self.num_env, self.num_red))
        self.red_yaw = torch.zeros((self.num_env, self.num_red))
        self.red_hp = torch.zeros((self.num_env, self.num_red))
        self.red_alive = torch.zeros((self.num_env, self.num_red), dtype=torch.bool)
        self.red_goal = torch.zeros((self.num_env, self.num_red), dtype=torch.bool)
        self.red_policy_index = torch.zeros((self.num_env, self.num_red), dtype=torch.int64)

        ############## Island info initailization
        self.island = torch.tensor(config.island_position)
        self.island_radius = torch.tensor([config.island_radius])

        ############## Logging
        self.log_t = torch.zeros(self.num_env)
        self.log_blue_reward = torch.zeros(self.num_env)
        self.log_blue_catch = torch.zeros(self.num_env)
        self.log_blue_attack = torch.zeros(self.num_env)
        self.log_blue_island_dead = torch.zeros(self.num_env)
        self.log_seed_now = torch.zeros(self.num_env, dtype=torch.int64)
    
    def reset(self):
        for idx in range(self.num_env):
            self.reset_idx(idx)

        if self.blue_agent_type == 'H':
            return (self.get_obs_tensor(), self.get_obs_tensor(red=True))
        
    
    def reset_idx(self, idx):
        
        self.episode_num += 1
        torch.manual_seed(self.random_seed + self.episode_num)
        np.random.seed(self.random_seed + self.episode_num)
        self.log_seed_now[idx] = self.episode_num

        # GAME Reset
        self.blue_hp[idx] = 1.
        self.blue_alive[idx] = True
        self.blue_valid[idx] = True
        self.red_hp[idx] = 1.
        self.red_alive[idx] = True
        self.red_goal[idx] = False
        self.log_t[idx] = 0.
        self.log_blue_reward[idx] = 0.
        self.log_blue_catch[idx] = 0.
        self.log_blue_attack[idx] = 0.
        self.log_blue_island_dead[idx] = 0.
        if self.red_policy_mode == 0:
            self.red_policy_index[idx, :] = torch.zeros(1, dtype=torch.long)
        elif self.red_policy_mode == 1:
            self.red_policy_index[idx, :] = torch.ones(1, dtype=torch.long)

        # Position Reset
        red_x, red_y = spawn_vehicle(np.array([0]).reshape(1,1), np.array([0]).reshape(1, 1), 16, 16.5, self.num_red)
        self.red_x[idx] = torch.from_numpy(red_x[0]).float()
        self.red_y[idx] = torch.from_numpy(red_y[0]).float()
        self.red_yaw[idx] = torch.rand(self.num_red) * 2 * np.pi - np.pi

        # BLUE Position Reset
        blue_x, blue_y = spawn_vehicle(np.array([0]).reshape(1,1), np.array([0]).reshape(1, 1), 2, 4, self.num_blue)
        self.blue_x[idx] = torch.from_numpy(blue_x[0]).float()
        self.blue_y[idx] = torch.from_numpy(blue_y[0]).float()
        self.blue_yaw[idx] = torch.rand(self.num_blue) * 2 * np.pi - np.pi

    def step(self, action_blue, action_red):

        action_blue = torch.clamp(action_blue, -1., 1.)

        # 0. Initialization (data)
        self.red_goal[:] = False
        self.blue_dead_moment[:] = False
        
        # 1. blue position update
        action_blue_norm = np.clip(action_blue * self.action_scale, self.action_min, self.action_max)
        self.blue_yaw += action_blue_norm * self.blue_alive
        self.blue_x += self.blue_velocity * torch.cos(self.blue_yaw) * self.blue_alive
        self.blue_y += self.blue_velocity * torch.sin(self.blue_yaw) * self.blue_alive
        self.blue_yaw[self.blue_yaw > torch.pi] -= 2 * torch.pi
        self.blue_yaw[self.blue_yaw < -torch.pi] += 2 * torch.pi

        # 2. red position update
        action_red_norm = np.clip(action_red * self.action_scale, self.action_min * 2, self.action_max * 2)
        self.red_yaw += action_red_norm * self.red_alive
        self.red_x += self.red_velocity * torch.cos(self.red_yaw) * self.red_alive
        self.red_y += self.red_velocity * torch.sin(self.red_yaw) * self.red_alive
        self.red_yaw[self.red_yaw > torch.pi] -= 2 * torch.pi
        self.red_yaw[self.red_yaw < -torch.pi] += 2 * torch.pi

        # 3. blue collision check
        blue_collision_agent = touch_island_circle(self.blue_x, self.blue_y, self.island, self.island_radius)
        self.blue_hp[blue_collision_agent] = 0.
        self.blue_dead_moment[torch.logical_and(self.blue_alive, blue_collision_agent)] = True
        self.blue_alive[blue_collision_agent] = False

        # 4. red island touch check (only 1 island !!)
        red_touch_agent = touch_island_circle(self.red_x, self.red_y, self.island, self.island_radius)
        self.red_hp[red_touch_agent] = 0.
        self.red_goal[torch.logical_and(self.red_alive, red_touch_agent)] = True
        self.red_alive[red_touch_agent] = False

        # 5. blue-red intercept
        ## 5.0. distance map
        dist_x = torch.cat([self.blue_x, self.red_x], dim=-1).unsqueeze(1) - self.blue_x.unsqueeze(-1)  # [num_env, num_blue, num_blue + num_red]
        dist_y = torch.cat([self.blue_y, self.red_y], dim=-1).unsqueeze(1) - self.blue_y.unsqueeze(-1)  # [num_env, num_blue, num_blue + num_red]
        distances = torch.sqrt(dist_x.pow(2) + dist_y.pow(2))  # [num_env, num_blue, num_blue + num_red]

        ## 5.1. attackable check + attack point calculate
        attackable = attackable_check(self.blue_x, self.blue_y, self.blue_yaw, self.red_x, self.red_y, 
                                      self.attack_range, self.attack_angle, distances[:, :, self.num_blue:])
        hp_decrease = 20 - (distances[:, :, self.num_blue:] / self.attack_range) * 20
        hp_decrease[~attackable] = 0. # hp decrease 0 which is not attackable
        hp_decrease.transpose(2, 1)[~self.red_alive] = 0. # hp decrease 0 which is not alive
        hp_decrease_target, attack_idx = torch.max(hp_decrease, dim=-1)

        ## 5.2. update red hp
        red_hp_before = self.red_hp.clone()
        red_attacked = torch.zeros((self.num_env, self.num_red)).scatter_add_(dim=1, index=attack_idx, src=hp_decrease_target) / self.red_hp_max
        self.red_hp = torch.clamp(self.red_hp - red_attacked, min=0.)
        red_dead_moment = (red_hp_before > 0.) & (self.red_hp == 0.) # [num_env, num_red]
        
        ## 5.3. info for calculating blue reward
        blue_attack_point = (red_hp_before - self.red_hp).sum(-1) # [num_env]
        blue_catch_num = red_dead_moment.sum(-1) # [num_env]

        # 6. red dead check
        self.red_alive[self.red_hp <= 0.] = False

        self.log_t += 1.

        blue_win = (self.red_alive.sum(-1) == 0) & (self.red_goal.sum(-1) == 0)
        red_win = (self.red_goal.sum(-1) > 0) | (self.blue_alive.sum(-1) == 0)

        reward_blue = self.get_reward_blue(blue_attack_point, blue_catch_num, self.blue_dead_moment, blue_win, red_win)
        reward_red = self.get_reward_red()

        self.log_blue_reward += reward_blue.sum(-1)
        self.log_blue_attack += blue_attack_point
        self.log_blue_catch += blue_catch_num
        self.log_blue_island_dead += self.blue_dead_moment.sum(-1)

        done_epi, done_blue, done_red, done_truncation = self.get_done()

        self.render(done_epi[0])

        info = []
        n_reset = 0
        for env_idx, done_epi_idx in enumerate(done_epi):
            if done_epi_idx:
                info.append({
                    'blue win': 1. if blue_win[env_idx].item() else 0.,
                    'red win': 1. if red_win[env_idx].item() else 0.,
                    'draw': 1. if not (blue_win[env_idx].item() or red_win[env_idx].item()) else 0.,
                    'blue reward': self.log_blue_reward[env_idx].item(),
                    'timestep': self.log_t[env_idx].item(),
                    'blue catch': self.log_blue_catch[env_idx].item(),
                    'blue attack': self.log_blue_attack[env_idx].item(),
                    'blue island dead': self.log_blue_island_dead[env_idx].item(),
                    'blue reward feedback': 0.,
                    'seed': self.log_seed_now[env_idx].item(),
                })
                self.reset_idx(env_idx)         
                n_reset += 1

        if n_reset > 0:
            dist_x = torch.cat([self.blue_x, self.red_x], dim=-1).unsqueeze(1) - self.blue_x.unsqueeze(-1)  # [num_env, num_blue, num_blue + num_red]
            dist_y = torch.cat([self.blue_y, self.red_y], dim=-1).unsqueeze(1) - self.blue_y.unsqueeze(-1)  # [num_env, num_blue, num_blue + num_red]
            distances = torch.sqrt(dist_x.pow(2) + dist_y.pow(2))  # [num_env, num_blue, num_blue + num_red]

        if self.blue_agent_type == 'H':
            state_blue = self.get_obs_tensor()
        state_red = self.get_obs_tensor(red=True)

        valid_blue, valid_red = self.get_valid()

        return (state_blue, state_red), (reward_blue, reward_red), \
            (done_epi, done_blue, done_red, done_truncation), (valid_blue, valid_red), info
            
    def render(self, done_epi_0):  # visualize only the first environment.
        if self.do_visualize_online or self.do_visualize_traj:
            blue_states = torch.stack([self.blue_x, self.blue_y, self.blue_yaw, self.blue_alive], dim=-1)[0].clone()
            red_states = torch.stack([self.red_x, self.red_y, self.red_yaw, self.red_alive], dim=-1)[0].clone()

        if self.do_visualize_online:
            self.visualizer.update(blue_states, red_states)
        
        elif self.do_visualize_traj:
            if self.visualizer_traj_iter % self.do_visualize_traj_freq == 0:
                self.visualizer_traj_data.append([blue_states, red_states])
                if done_epi_0:
                    file_name = pjoin(self.visualizer_traj_root_path, f'img_{self.visualizer_traj_iter}.png')
                    USVEnvVisualizer_traj(self.visualizer_traj_data, self.island_position, 
                                            self.island_radius_plot, self.map_size, file_name)
                    self.visualizer_traj_data = []
            
            if done_epi_0:
                self.visualizer_traj_iter += 1
    
    def get_reward_blue(self, blue_attack_point, blue_catch_num, blue_dead_moment, blue_win, red_win):
        
        # dense reward
        r_atk = 5. * blue_attack_point.reshape(-1, 1)
        r_catch = 20. * blue_catch_num.reshape(-1, 1)
        r_dead = -50. * blue_dead_moment

        # sparse reward
        r_win = 50. * blue_win.reshape(-1, 1)
        r_lose = -50. * red_win.reshape(-1, 1)

        return (r_atk + r_catch + r_dead + r_win + r_lose + 1.) / 20.

    def get_reward_red(self):
        return None
    
    def get_done(self):

        timeout_done = self.log_t >= self.max_step  # [num_env]
        red_team_finished = self.red_goal.sum(-1) >= self.end_condition_red_touch  # [num_env]
        red_all_die_done = self.red_alive.sum(-1) == 0  # [num_env]
        blue_all_die_done = self.blue_alive.sum(-1) == 0  # [num_env]
        
        # [num_env]
        episode_done = timeout_done | red_all_die_done | blue_all_die_done | red_team_finished

        # [num_env, num_blue]
        blue_agent_wise_done = red_team_finished.reshape(-1, 1) | red_all_die_done.reshape(-1, 1) | ~self.blue_alive

        # [num_env, num_red]
        red_agent_wise_done = red_team_finished.reshape(-1, 1) | red_all_die_done.reshape(-1, 1) | ~self.red_alive

        return episode_done, blue_agent_wise_done, red_agent_wise_done, timeout_done

    def get_obs_tensor(self, red=False):
        if red:
            return (torch.stack([self.blue_x, self.blue_y, self.blue_yaw, self.blue_alive], dim=-1), 
                    torch.stack([self.red_x, self.red_y, self.red_yaw, self.red_alive], dim=-1),
                    self.red_policy_index)
        else:
            return (torch.stack([self.blue_x, self.blue_y, self.blue_yaw, self.blue_alive], dim=-1), 
                    torch.stack([self.red_x, self.red_y, self.red_yaw, self.red_alive], dim=-1))
    
    def get_valid(self):
        blue_valid_return = self.blue_valid.clone() # Clone the current valid state for blue agents
        self.blue_valid[self.blue_dead_moment] = False  # Update valid state by marking dead agents as invalid for the next step
        red_valid_return = None
        return blue_valid_return, red_valid_return

    def get_info(self):
        blue_win = (self.red_alive.sum(-1) == 0) & (self.red_goal.sum(-1) == 0)
        red_win = (self.red_goal.sum(-1) > 0) | (self.blue_alive.sum(-1) == 0)
        draw = ~ (blue_win | red_win)
        return blue_win, red_win, draw
