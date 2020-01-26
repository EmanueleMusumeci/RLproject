import numpy as np

import gym
from gym.spaces import Box
from gym.wrappers import TimeLimit
from cv2 import cv2
import matplotlib.pyplot as plt
        
class AtariWrapper(gym.Wrapper):
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

    def __init__(self, env, noop_max=30, frame_skip=4, screen_width=88, screen_height=80, terminal_on_life_loss=False, grayscale_obs=True,scale_obs=False,render_scale_factor=3,
    crop_height_factor=0,crop_width_factor=0):
        super().__init__(env)
        assert frame_skip > 0
        assert screen_width > 0
        assert screen_height > 0
        assert noop_max >= 0
        if frame_skip > 1:
            assert 'NoFrameskip' in env.spec.id, 'disable frame-skipping in the original env. for more than one' \
                                                 ' frame-skip as it will be done by the wrapper'
        self.noop_max = noop_max
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

        self.frame_skip = frame_skip
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.terminal_on_life_loss = terminal_on_life_loss
        self.grayscale_obs = grayscale_obs
        self.scale_obs = scale_obs
        self.render_scale_factor = render_scale_factor
        self.crop_height_factor = crop_height_factor
        self.crop_width_factor = crop_width_factor

        # buffer of most recent two observations for max pooling
        if grayscale_obs:
            self.obs_buffer = [np.empty(env.observation_space.shape[:2], dtype=np.uint8),
                               np.empty(env.observation_space.shape[:2], dtype=np.uint8)]
        else:
            self.obs_buffer = [np.empty(env.observation_space.shape, dtype=np.uint8),
                               np.empty(env.observation_space.shape, dtype=np.uint8)]

        print("B1")

        self.ale = env.unwrapped.ale
        print("B2")
        self.lives = 0
        self.game_over = False

        _low, _high, _obs_dtype = (0, 255, np.uint8) if not scale_obs else (0, 1, np.float32)
        if grayscale_obs:
            self.observation_space = Box(low=_low, high=_high, shape=(self.screen_width, self.screen_height), dtype=_obs_dtype)
        else:
            self.observation_space = Box(low=_low, high=_high, shape=(self.screen_width, self.screen_height, 3), dtype=_obs_dtype)
        print("B3")
        
    def __del__(self):
        self.env.close()

    def step(self, action):
        R = 0.0

        for t in range(self.frame_skip):
            _, reward, done, info = self.env.step(action)
            R += reward
            self.game_over = done

            if self.terminal_on_life_loss:
                new_lives = self.ale.lives()
                done = done or new_lives < self.lives
                self.lives = new_lives

            if done:
                break
            if t == self.frame_skip - 2:
                if self.grayscale_obs:
                    self.ale.getScreenGrayscale(self.obs_buffer[0])
                else:
                    self.ale.getScreenRGB2(self.obs_buffer[0])
            elif t == self.frame_skip - 1:
                if self.grayscale_obs:
                    self.ale.getScreenGrayscale(self.obs_buffer[1])
                else:
                    self.ale.getScreenRGB2(self.obs_buffer[1])
        return self._get_obs(), R, done, info

    def reset(self, **kwargs):
        # NoopReset
        print("reset1")
        self.env.reset(**kwargs)
        
        print("reset2")
        noops = self.env.unwrapped.np_random.randint(1, self.noop_max + 1) if self.noop_max > 0 else 0
        
        print("reset3")
        for _ in range(noops):
            _, _, done, _ = self.env.step(0)
            if done:
                self.env.reset(**kwargs)

        print("reset4")
        #CAUSES SEGMENTATION FAULT (CORE DUMPED)
        #self.lives = self.ale.lives()
        self.lives = 0
        print("reset5")
        if self.grayscale_obs:
            print("reset5.1")
            self.ale.getScreenGrayscale(self.obs_buffer[0])
        else:
            print("reset5.1")
            self.ale.getScreenRGB2(self.obs_buffer[0])
        print("reset7")
        self.obs_buffer[1].fill(0)
        print("reset8")
        return self._get_obs()

    def _get_obs(self):
        if self.frame_skip > 1:  # more efficient in-place pooling
            np.maximum(self.obs_buffer[0], self.obs_buffer[1], out=self.obs_buffer[0])

        obs = cv2.resize(self.obs_buffer[0][:self.obs_buffer[0].shape[0]-round(self.obs_buffer[0].shape[0]*self.crop_height_factor)+1,:self.obs_buffer[0].shape[1]-round(self.obs_buffer[0].shape[1]*self.crop_width_factor)+1], (self.screen_height, self.screen_width), interpolation=cv2.INTER_AREA)
        if self.scale_obs:
            obs = np.asarray(obs, dtype=np.float32) / 255.0
        else:
            obs = np.asarray(obs, dtype=np.uint8)
        return obs

    def render(self, show_preprocessed=False):
        if show_preprocessed:
            obs = cv2.resize(self._get_obs(), (self.screen_height*self.render_scale_factor,self.screen_width*self.render_scale_factor), interpolation=cv2.INTER_AREA)
            cv2.imshow("Preprocessed",obs)
            key=cv2.waitKey(1)
        else:
            self.env.render()

    def preprocessed_shape(self):
        if self.grayscale_obs:
            channels = 1
        else:
            channels = 3
        resized_height = self.screen_height
        resized_width = self.screen_width

        return resized_height,resized_width,channels

    
