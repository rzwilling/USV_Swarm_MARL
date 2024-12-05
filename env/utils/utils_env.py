import torch
import numpy as np


def touch_island_circle(x_coords, y_coords, island, island_radius):

    x1 = x_coords.reshape(-1)[:, None]  # Shape (N, 1)
    y1 = y_coords.reshape(-1)[:, None]  # Shape (N, 1)
    x2 = island[0].reshape(-1)[None, :]  # Shape (1, M)
    y2 = island[1].reshape(-1)[None, :]  # Shape (1, M)

    distances = torch.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2).reshape(x_coords.shape)

    return distances - island_radius < 0.


def attackable_check(blue_x, blue_y, blue_yaw, red_x, red_y, attack_range, attack_angle, distances):
    
    # Expand blue coordinates and yaw for all red agents
    blue_x_exp = blue_x.unsqueeze(-1)  # Shape becomes [num_env, num_blue, 1]
    blue_y_exp = blue_y.unsqueeze(-1)  # Shape becomes [num_env, num_blue, 1]
    blue_yaw_exp = blue_yaw.unsqueeze(-1)  # Shape becomes [num_env, num_blue, 1]

    # Calculate directional vectors
    red_dir_x = red_x.unsqueeze(1) - blue_x_exp  # Shape becomes [num_env, num_blue, num_red]
    red_dir_y = red_y.unsqueeze(1) - blue_y_exp  # Shape becomes [num_env, num_blue, num_red]

    # Compute the left and right boundary vectors for the attack zone
    left_x = torch.cos(blue_yaw_exp - attack_angle)
    left_y = torch.sin(blue_yaw_exp - attack_angle)
    right_x = torch.cos(blue_yaw_exp + attack_angle)
    right_y = torch.sin(blue_yaw_exp + attack_angle)

    angle_left = -left_x * red_dir_y + left_y * red_dir_x
    angle_right = -right_x * red_dir_y + right_y * red_dir_x

    if attack_angle <= torch.pi / 2:
        angle_test = (angle_left < 0) & (angle_right > 0)
    else:
        angle_test = (angle_left < 0) | (angle_right > 0)

    attackable = (distances < attack_range) & angle_test

    return attackable


def spawn_vehicle(island_x, island_y, r_min, r_max, num_vehicle, theta_min=0, theta_max=2 * np.pi, island_choice=None):
    max_iter = 50
    num_env = island_x.shape[0]
    island_xy = np.stack([island_x, island_y], -1)
    A = 2 / (r_max * r_max - r_min * r_min)
    num_island = island_x.shape[1]
    vehicles_xy = np.zeros((num_env, num_vehicle, 2))

    for env in range(num_env):
        for i in range(num_vehicle):
            find_sol = False
            for _ in range(max_iter):
                selected_island = np.random.choice(num_island,
                                                   p=island_choice[env] if island_choice is not None else None)
                center_x, center_y = island_xy[env, selected_island]
                theta = np.random.uniform(theta_min, theta_max)
                r = np.sqrt(2 * np.random.uniform(0, 1) / A + r_min * r_min)
                vehicle_xy = np.array([center_x + r * np.cos(theta), center_y + r * np.sin(theta)])

                if np.linalg.norm(island_xy[env] - vehicle_xy, axis=1).min() > r_min:
                    find_sol = True
                    vehicles_xy[env, i] = vehicle_xy
                    break

            if not find_sol:
                print(f"Error in environment {env}, vehicle {i}!")

    return vehicles_xy[:, :, 0], vehicles_xy[:, :, 1]
