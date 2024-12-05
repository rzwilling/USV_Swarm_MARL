import torch


class HeuristicBluePolicy:
    
    def __init__(self, config):
        self.num_env = config.num_env
        self.num_blue = config.num_blue
        self.num_red = config.num_red
        self.detect_range = config.blue_detect_range
        self.max_turn_rate = config.action_max

    def get_action(self, state):
        state_blue, state_red = state
        distances = torch.norm(state_blue[:, :, :2].unsqueeze(2) - state_red[:, :, :2].unsqueeze(1), dim=3)
        distances[distances > self.detect_range] = float('inf')
        red_alive = state_red[:, :, -1]
        distances.transpose(2, 1)[red_alive == 0.] = float('inf')
        closest_red = torch.argmin(distances, dim=2)

        target_directions = state_red[torch.arange(self.num_env).unsqueeze(1), closest_red, :2] - state_blue[:, :, :2]
        target_angles = torch.atan2(target_directions[:, :, 1], target_directions[:, :, 0])

        angle_diff = target_angles - state_blue[:, :, 2]
        angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))

        turn_rate = torch.clamp(angle_diff, -self.max_turn_rate, self.max_turn_rate)

        return turn_rate / self.max_turn_rate