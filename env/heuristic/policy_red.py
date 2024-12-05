import torch


class PigeonRed:

    def __init__(self, config):
        self.config = config
        self.num_env = config.num_env
        self.num_red = config.num_red
        self.num_blue = config.num_blue

        self.ra_re_pairs = torch.tensor(config.red_ra_re_pairs)
        self.island = torch.tensor(config.island_position, dtype=torch.float32).unsqueeze(0).repeat(self.num_env, 1)
        self.island_radius = torch.full((self.num_env,), config.island_radius, dtype=torch.float32)
        self.k1 = config.red_k1

    def get_action(self, state):
        state_blue, state_red, policy_index = state
        
        new_ra_re = self.ra_re_pairs[policy_index]
        self.ra = new_ra_re[:, :, 0]
        self.re_indiv = new_ra_re[:, :, 1]
        
        xy_red = state_red[:, :, :2]
        xy_blue = state_blue[:, :, :2]
        red_yaw = state_red[:, :, 2]

        u_att_mtx = self.u_attack(xy_red)
        u_esc_mtx = self.u_escape(xy_red, xy_blue)
        u_avd_mtx = self.u_avoid(xy_red)

        u_pigeon = u_att_mtx + u_esc_mtx + 5 * u_avd_mtx

        yaw_direction = torch.stack([torch.cos(red_yaw), torch.sin(red_yaw)], dim=-1)

        sin = yaw_direction[:, :, 0] * u_pigeon[:, :, 1] - u_pigeon[:, :, 0] * yaw_direction[:, :, 1]
        cos_angle = torch.sum(yaw_direction * u_pigeon, dim=-1) / (
                    torch.norm(yaw_direction, dim=-1) * torch.norm(u_pigeon, dim=-1))
        angles = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))
        action = torch.ones((self.num_env, self.num_red), device=state_red.device)
        action[sin < 0.0] = -1.0
        action = action * angles / (torch.pi / 8)

        return action

    def u_attack(self, xy_red):
        xy_target_island = self.island.unsqueeze(1)
        dist_R_to_island = torch.norm(xy_target_island - xy_red, dim=2)
        attack_vector = xy_target_island - xy_red
        U_att_matrix = self.k1 * attack_vector / dist_R_to_island.unsqueeze(2)
        return U_att_matrix

    def u_escape(self, xy_red, xy_blue):
        re_indiv_k2 = self.re_indiv.unsqueeze(-1)
        dist_RB = torch.cdist(xy_red, xy_blue)
        dist_RB_valid = (0 < dist_RB) & (dist_RB < re_indiv_k2)
        escape_vector = xy_red.unsqueeze(2) - xy_blue.unsqueeze(1)

        dist_RB = torch.where(dist_RB_valid, dist_RB, torch.ones_like(dist_RB))
        K_2 = torch.exp(1 + (torch.pow((dist_RB - re_indiv_k2), 2) / torch.pow(re_indiv_k2, 2)))
        U_esc_matrix = K_2.unsqueeze(-1) * dist_RB_valid.unsqueeze(-1) / dist_RB.unsqueeze(-1) * escape_vector
        return U_esc_matrix.sum(dim=2)

    def u_avoid(self, xy_red):
        dist_RR = torch.cdist(xy_red, xy_red)
        ra = self.ra.unsqueeze(-1)
        dist_RR_valid = (0 < dist_RR) & (dist_RR < ra)
        avoid_vector = xy_red.unsqueeze(2) - xy_red.unsqueeze(1)
        K_3 = (ra - dist_RR) / ra
        dist_RR = torch.where(dist_RR == 0, torch.ones_like(dist_RR) * 1e-5, dist_RR)
        U_avd_matrix = K_3.unsqueeze(-1) * dist_RR_valid.unsqueeze(-1) / dist_RR.unsqueeze(-1) * avoid_vector
        return U_avd_matrix.sum(dim=2)
