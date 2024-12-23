    def update(self):
        if not self.replay_memory.memory_full:
            if self.replay_memory.pointer < self.replay_memory.num_minibatch:
                return

        batch = self.replay_memory.sample()
        state, state_r, action, action_r,  rewards, next_state, next_state_r, next_action_r, done = zip(*batch)


        state = torch.stack(state)
        state_r = torch.stack(state_r)
        action = torch.stack(action)
        action_r = torch.stack(action_r)
        next_action_r = torch.stack(next_action_r)

        rewards = torch.stack(rewards) # [40,1,5]
        rewards = rewards.squeeze(1).permute(1, 0).unsqueeze(-1)  # Shape [5, 40, 1]  # Shape [5, 40]

        next_state = torch.stack(next_state)
        next_state_r = torch.stack(next_state_r)
        done = torch.tensor([d[0] for d in done], dtype=torch.float)
        done = done.unsqueeze(1) 


        state_blue_graph = state.squeeze(1) #state.view(-1, state.shape[-1])  # Shape: [2, 4]
        state_red_graph = state_r.squeeze(1) # state_r.view(-1, state_r.shape[-1])  # Shape: [1, 4]

        x = torch.cat([state_blue_graph, state_red_graph], dim=1)  # Shape: [3, 4]

        data_list = []


        curr_batch_actions = []
        # Process each item and unpack the returned values into separate lists
        for item in torch.unbind(x, dim=0):
            edge_index, edge_attr = build_edge_index(item, self.communication_range, self.observation_range, self.attack_range)

            temp_act = []

            for i, actor_i in enumerate(self.actors):
                temp_act.append(actor_i(Data(x = item, edge_index = edge_index, edge_attr = edge_attr))[i])
            
        
            curr_batch_actions.append(torch.stack(temp_act))




        #for entry in data_list:


        next_state_blue_graph = next_state.squeeze(1) #state.view(-1, state.shape[-1])  # Shape: [2, 4]
        next_state_red_graph = next_state_r.squeeze(1) # state_r.view(-1, state_r.shape[-1])  # Shape: [1, 4]

        y = torch.cat([next_state_blue_graph, next_state_red_graph], dim=1)  # Shape: [3, 4]
        next_data_list = []


        for item in torch.unbind(y, dim=0):
            edge_index, edge_attr = build_edge_index(item, self.communication_range, self.observation_range, self.attack_range)

            temp_act = []

            for i, actor_i in enumerate(self.actors):
                temp_act.append(actor_i(Data(x = item, edge_index = edge_index, edge_attr = edge_attr))[i])
            
        
            next_data_list.append(torch.stack(temp_act))


        batch_next = next_data_list

        action_graph = action.squeeze(1) 
        action_r_graph = action_r.squeeze(1)


        print(action_graph.shape)
        print(action_r_graph.shape)

        actions_concat = torch.cat([action_graph, action_r_graph], dim=1).unsqueeze(-1)  # Shape: [3, 4]

        x_2 = torch.cat([x, actions_concat], dim=-1) 
        data_list_critic = []

        for item in torch.unbind(x_2, dim=0):
            edge_index, edge_attr = build_edge_index(item, self.communication_range, self.observation_range, self.attack_range)

            temp_act = []

            for i, actor_i in enumerate(self.critics):
                print(item)
                print(edge_index)
                print(edge_attr)
                temp_act.append(actor_i(Data(x = item, edge_index = edge_index, edge_attr = edge_attr))[i])
            
        
            data_list_critic.append(torch.stack(temp_act))

        


       next_action_r_graph = next_action_r.view(2, 1)


        print("Line 424")
        print(next_data_list[0].shape)
        print(next_action_r.shape)
        print(torch.cat(next_data_list, dim =1).shape)

        next_actions_concat =  torch.cat([torch.cat(next_data_list, dim =1), next_action_r_graph], dim=0) #.unsqueeze(-1) 

        y_2 = torch.cat([y, next_actions_concat], dim=-1) 


        next_data_list_critic = []
        
        for item in torch.unbind(y_2, dim=0):
            edge_index, edge_attr = build_edge_index(item, self.communication_range, self.observation_range, self.attack_range)
            
            temp_act = []
            for i, actor_i in enumerate(self.critics):
                temp_act.append(actor_i(Data(x = item, edge_index = edge_index, edge_attr = edge_attr))[i])
            
            next_data_list_critic.append(torch.stack(temp_act))
                        
    
        #batch_next_critic = Batch.from_data_list(next_data_list_critic)


        

        for i, (actor_i, critic_i, act_optimizer_i, cri_optimizer_i, reward) in enumerate(zip(self.actors, self.critics, self.actor_optimizers, self.critic_optimizers, rewards)):
            state_value = []

            data_list_critic = []

            for item in torch.unbind(x_2, dim=0):
                edge_index, edge_attr = build_edge_index(item, self.communication_range, self.observation_range, self.attack_range)

                temp_act = []

                for i, actor_i in enumerate(self.critics):
                    temp_act.append(critic_i(Data(x = item, edge_index = edge_index, edge_attr = edge_attr))[i])
            state_value.append(torch.stack(temp_act))


            for data in data_list_critic:
                state_value.append(critic_i(data))

            next_state_value = []
            for data in next_data_list_critic:
                next_state_value = critic_i(data)

            state_value = torch.stack(state_value)
            next_state_value = torch.stack(next_state_value)

            #state_value = critic_i(batch_curr_critic)
            #next_state_value = critic_i(batch_next_critic)

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

            next_actions[i] = actor_i(batch_next)
            curr_actions[i] = actor_i(batch_curr)



        return        
