import os
import math

import torch
import matplotlib
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# if os.name == 'posix': # Check if the operating system is Linux
#     matplotlib.use('Agg') # Use 'Agg' backend for headless Linux environments
# else:
#     matplotlib.use('TkAgg') # Use 'TkAgg' backend for interactive environments


plt.rcParams['figure.figsize'] = [8, 8]

class VisualUSV:
    def __init__(self, color, scale=1):
        self.color = color
        self.scale = scale
        self.body_plot, = plt.plot([], [], color=color, animated=True)
        self.dead_plot = plt.scatter([], [], marker='x', color=color, zorder=-200, linewidth=3, animated=True)

    def update(self, state):
        x, y, yaw = state[0], state[1], state[2]
        head = [x + self.scale * 0.3 * math.cos(yaw), y + self.scale * 0.3 * math.sin(yaw)]
        tail1 = [x + self.scale * 0.2 * math.cos(yaw + 2.35619), y + self.scale * 0.2 * math.sin(yaw + 2.35619)]
        tail2 = [x + self.scale * 0.2 * math.cos(yaw - 2.35619), y + self.scale * 0.2 * math.sin(yaw - 2.35619)]
        self.body_plot.set_data([head[0], tail1[0], tail2[0], head[0]], [head[1], tail1[1], tail2[1], head[1]])

    def deadUSV(self, x, y):
        self.dead_plot.set_offsets(np.array([x, y]))


class USVEnvVisualizer:
    def __init__(self, num_blue, num_red, island_position, island_radius, map_size=34):
        print("=============== LINE 37 ===============")
        print(map_size)
        self.num_blue = num_blue
        self.num_red = num_red
        self.map_size = float(map_size)
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_xlim(-self.map_size / 2, self.map_size / 2)
        self.ax.set_ylim(-self.map_size / 2, self.map_size / 2)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        island = plt.Circle(island_position, island_radius, color='black')
        self.ax.add_artist(island)

        self.blue_usvs = [VisualUSV('blue') for _ in range(num_blue)]
        self.red_usvs = [VisualUSV('red') for _ in range(num_red)]

        for usv in self.blue_usvs + self.red_usvs:
            self.ax.add_line(usv.body_plot)
            self.ax.add_artist(usv.dead_plot)

        #plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        plt.tight_layout()
        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        plt.show(block=False)

    def update(self, blue_states, red_states):
        self.fig.canvas.restore_region(self.background)

        for i, usv in enumerate(self.blue_usvs):
            if i < len(blue_states):
                if blue_states[i][3]:
                    usv.update(blue_states[i])
                    self.ax.draw_artist(usv.body_plot)
                else:
                    usv.deadUSV(blue_states[i][0], blue_states[i][1])
                    self.ax.draw_artist(usv.dead_plot)

        for i, usv in enumerate(self.red_usvs):
            if i < len(red_states):
                if red_states[i][3]:
                    usv.update(red_states[i])
                    self.ax.draw_artist(usv.body_plot)
                else:
                    usv.deadUSV(red_states[i][0], red_states[i][1])
                    self.ax.draw_artist(usv.dead_plot)

        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()

    def close(self):
        plt.close(self.fig)


def USVEnvVisualizer_traj(visualizer_traj_data, island_position, island_radius, map_size, file_name):
    print("=============== LINE 89 ===============")
    print(map_size)
    map_size = float(map_size)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-map_size / 2, map_size / 2)
    ax.set_ylim(-map_size / 2, map_size / 2)
    ax.set_xticks([])
    ax.set_yticks([])
    island = plt.Circle(island_position, island_radius, color='black')
    ax.add_artist(island)
    plt.tight_layout()
    
    data_b = torch.stack([visualizer_traj_data_[0] for visualizer_traj_data_ in visualizer_traj_data])
    data_r = torch.stack([visualizer_traj_data_[1] for visualizer_traj_data_ in visualizer_traj_data])

    data_b = data_b.transpose(1, 0).numpy()  # agent, timestep, info_type
    data_r = data_r.transpose(1, 0).numpy()  # agent, timestep, info_type

    for i, r in enumerate(data_r):
        evaluate_visualize_USV(i, r, 'red')
    
    for i, b in enumerate(data_b):
        evaluate_visualize_USV(i, b, 'blue')
    
    plt.savefig(file_name)
    plt.close('all')


def evaluate_visualize_USV(idx, data, color_type='black', start=3, space=7):

    if color_type == 'blue':
        usv_color = 'blue'
        info_color = 'darkblue'
    elif color_type == 'red':
        usv_color = 'red'
        info_color = 'darkred'
    else:
        usv_color = 'black'
        info_color = 'grey'

    valid = data[data[:, 3] == 1.0].T
    max_len = len(valid[0])
    if max_len < space:
        space_new = - 2
        end = -1
        draw_arrow = False
    else:
        space_new = -space
        end = -3
        draw_arrow = True

    plt.plot(np.array(valid[0][min(start, max_len):space_new]),
             np.array(valid[1][min(start, max_len):space_new]),
             color=info_color, linestyle='dashed', linewidth=1)
    if draw_arrow:
        plt.arrow(np.array(valid[0][space_new]), np.array(valid[1][space_new]),
                    np.array(valid[0][space_new + 1]) - np.array(valid[0][space_new]),
                    np.array(valid[1][space_new + 1]) - np.array(valid[1][space_new]),
                    width=0.07, color=info_color, linewidth=0.25, zorder=1)

    visualize_USV(valid[0][0], valid[1][0], valid[2][0], color=usv_color, attack_range=False, USV_id=idx)


def visualize_USV(x, y, yaw, color, attack_range=False, a_angle=None, a_range=None, 
                  USV_id=None, target=None, scale=None, con_range=None, center_x=0.0, center_y=0.0):
    scale = 0.5
    head = [x + scale * 0.3 * math.cos(yaw), y + scale * 0.3 * math.sin(yaw)]
    tail1 = [x + scale * 0.2 * math.cos(yaw + 2.35619), y + scale * 0.2 * math.sin(yaw + 2.35619)]
    tail2 = [x + scale * 0.2 * math.cos(yaw - 2.35619), y + scale * 0.2 * math.sin(yaw - 2.35619)]

    plt.plot([head[0], tail1[0]], [head[1], tail1[1]], c=color)
    plt.plot([tail1[0], tail2[0]], [tail1[1], tail2[1]], c=color)
    plt.plot([tail2[0], head[0]], [tail2[1], head[1]], c=color)

    if USV_id is not None:
        plt.text(x, y, str(USV_id + 1), color='black', fontsize=10, ha='left', va='bottom')

    if attack_range:
        t = np.linspace(-1, 1, 100)
        out_x = a_range * np.cos(yaw + t * a_angle) + x
        out_y = a_range * np.sin(yaw + t * a_angle) + y

        plt.plot(out_x, out_y, c=color, linestyle='--')
        plt.plot([x, out_x[0]], [y, out_y[0]], c=color, linestyle='--')
        plt.plot([x, out_x[-1]], [y, out_y[-1]], c=color, linestyle='--')

    if con_range:
        t = np.linspace(-1, 1, 100)
        out_x = con_range[0][0] * np.cos(yaw + t * 2 * np.pi) + x
        out_y = con_range[0][0] * np.sin(yaw + t * 2 * np.pi) + y
        plt.plot(out_x, out_y, c='grey', linestyle='--', linewidth=0.25)

    if target is not None:
        for target_item in target:
            plt.annotate('', xy=(target_item[0], target_item[1]), xytext=(head[0], head[1]), 
                         arrowprops={'arrowstyle': '->'}, va='center')