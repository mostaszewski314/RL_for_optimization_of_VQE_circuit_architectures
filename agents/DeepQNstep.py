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
from collections import namedtuple, deque
import numpy as np
from itertools import product
import copy
from utils import dictionary_of_actions
from .DeepQ import DQN
        

class DQN_Nstep(DQN):

    def __init__(self, conf, action_size, state_size, device):
        super(DQN_Nstep, self).__init__(conf, action_size, state_size, device)
        memory_size = conf['agent']['memory_size']

        self.memory = N_step_ReplayMemory(memory_size, conf['agent']['n_step'], self.gamma)



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
        done_batch = torch.stack(batch.done)
        

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


    
class N_step_ReplayMemory(object):

    def __init__(self, capacity, n_step, gamma):
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.memory = deque(maxlen=capacity)
        self.n_step_memory = deque(maxlen=n_step)
        self.Transition = namedtuple('Transition',
                                    ('state', 'action', 'reward',
                                    'next_state','done'))
        
    def _n_step(self):
        """ Constructs n step reward"""
        reward, n_state, done = self.n_step_memory[-1][-3:]
        for _, _, rwd, next_st, do in list(self.n_step_memory)[::-1][1:]:
            reward = self.gamma * reward * (1 - do) + rwd
            n_state, done = (next_st, do) if do else (n_state, done)
        return reward, n_state, done
    
    
    def push(self, *args):
        """Saves a transition."""
        
        self.n_step_memory.append(self.Transition(*args))
        if len(self.n_step_memory) < self.n_step:
            return
        reward, n_state, done = self._n_step()
        state, action = self.n_step_memory[0][: 2]
        self.memory.append(self.Transition(*[state, action, reward[0], n_state, done]))
   
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    def clean_memory(self):
        self.memory = deque(maxlen=self.capacity)
        self.n_step_memory = deque(maxlen=self.n_step)

if __name__ == '__main__':
    pass
