import numpy as np

import gym
from gym.spaces import Box
from gym.wrappers import TimeLimit
from cv2 import cv2
import matplotlib.pyplot as plt
        
class AtariRamWrapper(gym.Wrapper):
    r"""Atari 2600 preprocessings. 
    This class follows the guidelines in 
    Machado et al. (2018), "Revisiting the Arcade Learning Environment: 
    Evaluation Protocols and Open Problems for General Agents".
    Specifically:
    * NoopReset: obtain initial state by taking random number of no-ops on reset. 
    * Frame skipping: 4 by default
    * Max-pooling: most recent two observations
    * Termination signal when a life is lost: turned off by default. Not recommended by Machado et al. (2018).
    * Resize to a square image: 84x84 by default
    * Grayscale observation: optional
    * Scale observation: optional
    Args:
        env (Env): environment
        noop_max (int): max number of no-ops
        frame_skip (int): the frequency at which the agent experiences the game. 
        screen_size (int): resize Atari frame
        terminal_on_life_loss (bool): if True, then step() returns done=True whenever a
            life is lost. 
        grayscale_obs (bool): if True, then gray scale observation is returned, otherwise, RGB observation
            is returned.
        scale_obs (bool): if True, then observation normalized in range [0,1] is returned. It also limits memory
            optimization benefits of FrameStack Wrapper.
    """

    def __init__(self, environment_name, noop_max=30, frame_skip=4, screen_width=88, screen_height=80, terminal_on_life_loss=False, render_scale_factor=1):
        super().__init__(gym.make(environment_name))
        assert frame_skip > 0
        assert screen_width > 0
        assert screen_height > 0
        assert noop_max >= 0
        if frame_skip > 1:
            assert 'NoFrameskip' in self.env.spec.id, 'disable frame-skipping in the original env. for more than one' \
                                                 ' frame-skip as it will be done by the wrapper'
        self.noop_max = noop_max
        assert self.env.unwrapped.get_action_meanings()[0] == 'NOOP'

        self.environment_name=environment_name
        self.frame_skip = frame_skip
        self.screen_width = screen_width
        self.screen_height = screen_height

        self.render_scale_factor = render_scale_factor

        self.terminal_on_life_loss = terminal_on_life_loss

        self.ale = self.env.unwrapped.ale
        self.lives = 0
        self.game_over = False

        last_observation = None
        

    def step(self, action):
        R = 0.0

        for t in range(self.frame_skip):
            last_observation, reward, done, info = self.env.step(action)
            R += reward
            self.game_over = done

            if self.terminal_on_life_loss:
                new_lives = self.ale.lives()
                done = done or new_lives < self.lives
                self.lives = new_lives

            if done:
                break
        return last_observation, R, done, info

    def reset(self, **kwargs):
        # NoopReset
        last_observation = self.env.reset(**kwargs)
        
        noops = self.env.unwrapped.np_random.randint(1, self.noop_max + 1) if self.noop_max > 0 else 0
        
        last_observation = None
        for _ in range(noops):
            last_observation, _, done, _ = self.env.step(0)
            if done:
                last_observation = self.env.reset(**kwargs)

        self.lives = self.ale.lives()
        return last_observation

    def render(self, show_scaled=True):
        if show_scaled:
            render = self.env.render(mode="rgb_array")
            render = cv2.cvtColor(render, cv2.COLOR_RGB2BGR)
            obs = cv2.resize(render, (self.screen_height*self.render_scale_factor,self.screen_width*self.render_scale_factor), interpolation=cv2.INTER_AREA)
            cv2.imshow(self.environment_name,obs)
            key=cv2.waitKey(1)
        else:
            self.env.render()
    
    def clone(self):
        return AtariRamWrapper(self.environment_name,self.noop_max,self.frame_skip,self.screen_width,self.screen_height,self.terminal_on_life_loss)

    def get_observation_space(self):
        return self.env.observation_space.shape 

    def get_action_space(self):
        return self.env.action_space