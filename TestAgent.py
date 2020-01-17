import os
import glob
import numpy as np
import time

import tensorflow as tf
from tensorflow import keras, GradientTape
from tensorflow.keras import layers, optimizers, losses, models

from Environment import *

class TestAgent:
    def __init__(self, env, delay=0.0, show_debug_info=False):
        self.env = env
        self.action_delay = delay
        self.last_action_time = time.time()
        self.debug = show_debug_info

    def act(self, observation):
        if self.debug: print("Observation: "+ str(observation))

        return self.policy(observation), [1,2,3]

    def policy(self, observation):
        push_left = 0
        push_right = 2
        none = 1
        
        current_time = time.time()
        if (current_time - self.last_action_time) >= self.action_delay:
            self.last_action_time = time.time()
            action = push_right
        else:
            action = none
            
        return action


if __name__ == '__main__':
    env = Environment("MsPacmanPreprocessed-v0",use_custom_env_register=True,show_debug_info=True, show_preprocessed=True)
    print(env.rendering_delay)
    agent = TestAgent(env, 0.005)
    #rollouts, _ = env.collect_rollouts(agent, 1, 2000)
    obs = env.reset()
    action, _ = agent.act(obs)
    env.render_agent(agent)
    env.close()
    
