import numpy as np

import gym
from gym.spaces import Box
from gym.wrappers import TimeLimit
from cv2 import cv2
import matplotlib.pyplot as plt

#Ho dovuto riscrivere i wrapper per dotarli di un metodo render, in maniera tale da verificare il funzionamento del preprocessing
class GenericWrapper(gym.Wrapper):

    def __init__(self, environment_name):
        super().__init__(gym.make(environment_name))
        self.environment_name = environment_name

    def render(self):
        self.env.render()

    def clone(self):
        return GenericWrapper(self.environment_name)

    def get_observation_space(self):
        return self.env.observation_space    

    def get_action_space(self):
        return self.env.action_space    

    
