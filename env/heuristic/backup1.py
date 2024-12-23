



        next_actions = []
        curr_actions = []



        for actor_i in self.actors:
            next_actions.append(actor_i(next_state, next_state_r))
            curr_actions.append(actor_i(state, state_r))
        next_actions_tensor = torch.stack(next_actions, dim=-1)  # Shape will be [40, 1, 5]
        curr_actions_tensor = torch.stack(curr_actions, dim=-1)  # Shape will be [40, 1, 5]
        
        for i, (actor_i, critic_i, act_optimizer_i, cri_optimizer_i, reward) in enumerate(zip(self.actors, self.critics, self.actor_optimizers, self.critic_optimizers, rewards)):
            next_actions_tensor = torch.stack(next_actions, dim=-1)  # Shape will be [40, 1, 5]
            curr_actions_tensor = torch.stack(curr_actions, dim=-1)  # Shape will be [40, 1, 5]
            state_value = critic_i(state, state_r, action)
            next_state_value = critic_i(next_state, next_state_r, next_actions_tensor)
            target_value = reward + (1 - done) * self.gamma * next_state_value
            # Compute critic loss (Mean Squared Error)
            critic_loss = nn.MSELoss()(state_value, target_value)
            # Update critic
            cri_optimizer_i.zero_grad()
            critic_loss.backward(retain_graph=True)
            cri_optimizer_i.step()
            # Compute actor loss (negative of expected return)
            actor_loss = -critic_i(state, state_r, curr_actions_tensor).mean()
            # Update actor
            act_optimizer_i.zero_grad()
            actor_loss.backward(retain_graph=True)
            act_optimizer_i.step()
            next_actions[i] = actor_i(next_state, next_state_r)
            curr_actions[i] = actor_i(state, state_r)
        return  