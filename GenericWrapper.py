import numpy as np

import gym
from gym.spaces import Box
from gym.wrappers import TimeLimit
from cv2 import cv2
import matplotlib.pyplot as plt

#Ho dovuto riscrivere i wrapper per dotarli di un metodo render, in maniera tale da verificare il funzionamento del preprocessing
class GenericWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)

    def render(self):
        self.env.render()


    
