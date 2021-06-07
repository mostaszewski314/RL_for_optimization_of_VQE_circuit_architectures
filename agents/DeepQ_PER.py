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
from collections import namedtuple
import numpy as np
from itertools import product

from utils import dictionary_of_actions
from .DeepQ import DQN, ReplayMemory

        
class DQN_PER(DQN):

    def __init__(self, conf, action_size, state_size, device):
        super(DQN_PER, self).__init__(conf, action_size, state_size, device)
        memory_size = conf['agent']['memory_size']

        alpha = conf['agent']['alpha']
        beta = conf['agent']['beta']
        beta_incr = conf['agent']['beta_incr']
        self.memory = PER_ReplayMemory(memory_size, alpha=alpha, beta = beta, beta_incr = beta_incr)


    def replay(self, batch_size: int):
        if self.step_counter %self.update_target_net ==0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.step_counter += 1  

            
        transitions, indices, weights = self.memory.sample(batch_size)
        batch = self.Transition(*zip(*transitions))

        next_state_batch = torch.stack(batch.next_state)
        state_batch = torch.stack(batch.state)
        action_batch = torch.tensor(batch.action, device=self.device)
        reward_batch = torch.stack(batch.reward)
        done_batch = torch.stack(batch.done)

        weights = torch.tensor(weights, device=self.device)

        # if not self.with_angles:
        #     state_batch = state_batch[:, :-self.num_layers]
        #     next_state_batch = next_state_batch[:, :-self.num_layers]

        state_action_values = self.policy_net.forward(state_batch).gather(1, action_batch.unsqueeze(1))
        """ Double DQN """        
        next_state_values = self.target_net.forward(next_state_batch)
        next_state_actions = self.policy_net.forward(next_state_batch).max(1)[1].detach()
        next_state_values = next_state_values.gather(1, next_state_actions.unsqueeze(1)).squeeze(1)
            
        """ Compute the expected Q values """
        expected_state_action_values = (next_state_values * self.gamma) * (1-done_batch) + reward_batch
        expected_state_action_values = expected_state_action_values.view(-1, 1)

        assert state_action_values.shape == expected_state_action_values.shape, "Wrong shapes in loss"
        cost = self.fit(state_action_values, expected_state_action_values, weights, indices)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return cost
    
    def fit(self, output, target_f, weights, indices):
        self.optim.zero_grad()
        loss = self.loss(output, target_f) * weights.view(-1, 1)
        priorities = loss + 1e+5
        loss = loss.mean()
        loss.backward()
        self.memory.update_priorities(indices, priorities.data.cpu().numpy())
        self.optim.step()
        return loss.item()



class PER_ReplayMemory(ReplayMemory):
    
    def __init__(self, capacity, alpha=0.6, beta = 0.4, beta_incr = 0.001):
        super(PER_ReplayMemory, self).__init__(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_pp = beta_incr
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self._args = (capacity, alpha, np.copy(beta), beta_incr)
     

    def push(self, *args):
        """Saves a transition."""
        max_p = self.priorities.max() if self.memory else 1.0
        if len(self.memory) < self.capacity:
            self.memory.append(None)        
        self.memory[self.position] = self.Transition(*args)
        self.priorities[self.position] = max_p
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.memory) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.position]
        
        self.beta = np.min([1., self.beta + self.beta_pp]) 

        probs  = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]
        
        total    = len(self.memory)
        weights  = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights  = np.array(weights, dtype=np.float32)

        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def clean_memory(self):
        self.memory = []
        self.position = 0
        self.alpha =  self._args[1]
        self.beta =  self._args[2]
        self.beta_pp =  self._argsp[3]
        self.priorities = np.zeros(( self._args[0],), dtype=np.float32)
  
if __name__ == '__main__':
    pass
