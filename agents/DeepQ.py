
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 13:17:09 2020

@author: mateusz
"""

import torch.nn as nn
import torch.nn.functional as F
import random
import torch

import copy
from collections import namedtuple, deque
import numpy as np
from itertools import product

from utils import dictionary_of_actions, dict_of_actions_revert_q


class DQN(object):

    def __init__(self, conf, action_size, state_size, device):
        self.num_qubits = conf['env']['num_qubits']
        self.num_layers = conf['env']['num_layers']
        memory_size = conf['agent']['memory_size']
        
        self.final_gamma = conf['agent']['final_gamma']
        self.epsilon_min = conf['agent']['epsilon_min']
        self.epsilon_decay = conf['agent']['epsilon_decay']
        learning_rate = conf['agent']['learning_rate']
        self.update_target_net = conf['agent']['update_target_net']
        neuron_list = conf['agent']['neurons']
        drop_prob = conf['agent']['dropout']
        self.with_angles = conf['agent']['angles']
        
        if "memory_reset_switch" in conf['agent'].keys():
            self.memory_reset_switch =  conf['agent']["memory_reset_switch"]
            self.memory_reset_threshold = conf['agent']["memory_reset_threshold"]
            self.memory_reset_counter = 0
        else:
            self.memory_reset_switch =  False
            self.memory_reset_threshold = False
            self.memory_reset_counter = False

        self.action_size = action_size
        
        self.state_size = state_size if self.with_angles else state_size - self.num_layers
        self.state_size = self.state_size + 1 if conf['agent']['en_state'] else self.state_size
    
          
        self.translate = dictionary_of_actions(self.num_qubits)
        self.rev_translate = dict_of_actions_revert_q(self.num_qubits)
        
        self.policy_net = self.unpack_network(neuron_list, drop_prob).to(device)
        self.target_net = copy.deepcopy(self.policy_net)
        self.target_net.eval()
        

        self.gamma = torch.Tensor([np.round(np.power(self.final_gamma,1/self.num_layers),2)]).to(device)   # discount rate
        self.memory = ReplayMemory(memory_size)
        self.epsilon = 1.0  # exploration rate

        self.optim = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss = torch.nn.SmoothL1Loss()
        self.device = device
        self.step_counter = 0

   
        self.Transition = namedtuple('Transition',
                            ('state', 'action', 'reward',
                            'next_state','done'))

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def act(self, state):

        state = state.unsqueeze(0)
        epsilon = False
        if torch.rand(1).item() <= self.epsilon:
            epsilon = True
            return (torch.randint(self.action_size, (1,)).item(), epsilon)
        act_values = self.policy_net.forward(state)
       
        return torch.argmax(act_values[0]).item(), epsilon

    def replay(self, batch_size):
        if self.step_counter %self.update_target_net ==0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.step_counter += 1
        
        transitions = self.memory.sample(batch_size)
        batch = self.Transition(*zip(*transitions))
        
        next_state_batch = torch.stack(batch.next_state)
        state_batch = torch.stack(batch.state)
        action_batch = torch.stack(batch.action)#, device=self.device)
        reward_batch = torch.stack(batch.reward)#.to(device=self.device)
        done_batch = torch.stack(batch.done)#.to(device=self.device)
        

        state_action_values = self.policy_net.forward(state_batch).gather(1, action_batch.unsqueeze(1))
        """ Double DQN """        
        next_state_values = self.target_net.forward(next_state_batch)
        next_state_actions = self.policy_net.forward(next_state_batch).max(1)[1].detach()
        next_state_values = next_state_values.gather(1, next_state_actions.unsqueeze(1)).squeeze(1)
        
       
    
        """ Compute the expected Q values """
        expected_state_action_values = (next_state_values * self.gamma) * (1-done_batch) + reward_batch
        expected_state_action_values = expected_state_action_values.view(-1, 1)

        assert state_action_values.shape == expected_state_action_values.shape, "Wrong shapes in loss"
        cost = self.fit(state_action_values, expected_state_action_values)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon,self.epsilon_min)
        assert self.epsilon >= self.epsilon_min, "Problem with epsilons"
        return cost

    def fit(self, output, target_f):
        self.optim.zero_grad()
        loss = self.loss(output, target_f)
        loss.backward()
        self.optim.step()
        return loss.item()

    def unpack_network(self, neuron_list, p):
        layer_list = []
        neuron_list = [self.state_size] + neuron_list 
        for input_n, output_n in zip(neuron_list[:-1], neuron_list[1:]):
            layer_list.append(nn.Linear(input_n, output_n))
            layer_list.append(nn.LeakyReLU())
            layer_list.append(nn.Dropout(p=p))
        layer_list.append(nn.Linear(neuron_list[-1], self.action_size))
        return nn.Sequential(*layer_list)


class ReplayMemory(object):

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.Transition = namedtuple('Transition',
                                    ('state', 'action', 'reward',
                                    'next_state','done'))

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    def clean_memory(self):
        self.memory = []
        self.position = 0

if __name__ == '__main__':
    pass

