import torch
import numpy as np


class Config:

    do_visualize_online = True
    do_visualize_traj = False  # when you visualize online, visualize trajectory will not work.
    do_visualize_traj_freq = 10
    root_path = 'results/temp'

    random_seed = 42
    num_env = 10  # 250
    max_step = 300

    blue_agent_type = 'H'
    red_agent_type = 'H'
    train_model_mode = False

    num_timesteps_total = 5000  #
    num_timesteps_per_update = 500  # 50000
    num_updates = int(num_timesteps_total // num_timesteps_per_update)
    num_timesteps = num_timesteps_per_update // num_env
    num_updates_iter = 1
    save_interval = 1
    device = 'cpu'

    ################# 2. PPO Setting
    lr_actor = 0.0005
    lr_critic = 0.001

    clip_coef = 0.2
    ent_coef = 0.01
    vf_coef = 0.5
    max_grad_norm = 10.
    num_minibatch = 512 #40
    replay_memory_capacity = 100000 # To do 
    noise = 0.7
    tau = 0.001

    gae_lambda = 0.98
    gamma = 0.8 #0.99
    use_gae = True

    out_channels = 16
    hidden = 64

    ################# 3. Environment
    # General setting
    num_blue = 3
    num_red = 1
    island_position = [1, 0]
    island_radius = 1.2
    blue_velocity = 0.5144 / 100 * 5 * 5
    red_velocity = 0.5144 / 100 * 7 * 5
    red_hp_max = 20 * 10
    red_policy_mode = 0
    red_ra_re_pairs = [(0.1, 0.1), (1.0, 2.2)]

    # Action
    action_min = -torch.pi / 16
    action_max = torch.pi / 16
    action_scale = (action_max - action_min) / 2
    attack_range = 1.5
    attack_angle = 3 * 45 / 180 * np.pi
    blue_detect_range = 15.0
    end_condition_red_touch = 1
    red_k1 = 1
    
    # Map
    map_size = 34
