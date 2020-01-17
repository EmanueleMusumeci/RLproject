import gym
from gym.envs.registration import registry, register, make, spec
from gym.wrappers import AtariPreprocessing
from collections import namedtuple, defaultdict
import time
from AtariWrapper import AtariWrapper
from GenericWrapper import GenericWrapper
import matplotlib.pyplot as plt

##TODO: add a way to store the NN shape

#Il CustomEnvironmentRegister serve a registrare environment con caratteristiche custom
#(come il numero massimo di episodi)

#Named tuple used to store useful info about the custom environment being registered
GenericEnvironmentInfo = namedtuple("EnvironmentInfo","custom_id type render_delay")
AtariEnvironmentInfo = namedtuple("EnvironmentInfo","custom_id type render_delay frame_skip screen_width screen_height crop_height_factor crop_width_factor preprocess")
class CustomEnvironmentRegister:
    def __init__(self):
        self.registeredEnvironments = {}
        
        register(
            id='MountainCarCustom-v0',
            entry_point='gym.envs.classic_control:MountainCarEnv',
            max_episode_steps=1000,      # MountainCar-v0 uses 200
            #reward_threshold=-110.0,
        )
        self.registeredEnvironments["MountainCar-v0"] = GenericEnvironmentInfo("MountainCarCustom-v0","classic",0.0)

        #
        self.registeredEnvironments["MsPacman-v0"] = AtariEnvironmentInfo("MsPacman-v0","atari",
        0.05,
        1,
        210,
        160,
        0,
        0,
        False
        )

        #
        self.registeredEnvironments["MsPacmanPreprocessed-v0"] = AtariEnvironmentInfo("MsPacman-v0","atari",
        0.05,
        1,
        88,
        88,
        0.15,
        0,
        True
        )
    
    def register_environment(self, config_path):
        #Implement uploading environment from path
        return

    def get_environment(self, env_name):
        #Check if we have registered a custom version
        if env_name in self.registeredEnvironments:
            env = gym.make(self.registeredEnvironments[env_name].custom_id)
            env_info = self.registeredEnvironments[env_name]
        
            return env, env_info
        else:
            return gym.make(env_name), None



##NOTICE: MODIFY THIS TUPLE IF MORE INFO ARE NEEDED FROM ENVIRONMENT
RolloutTuple = namedtuple("RolloutTuple", "observation reward action action_probabilities")

class Environment:
    def __init__(self, environment_name, use_custom_env_register=True, show_debug_info=False, show_preprocessed=False):
        if use_custom_env_register:
            self.env_register = CustomEnvironmentRegister()
            #Create wrapper
            env, env_info = self.env_register.get_environment(environment_name)
            self.type = env_info.type
            self.action_space = env.action_space
            self.observation_space = env.observation_space
            self.show_preprocessed = show_preprocessed
            
            if(self.type == "atari"):
                self.rendering_delay = env_info.render_delay
                self.frame_skip = env_info.frame_skip
                self.screen_width = env_info.screen_width
                self.screen_height = env_info.screen_height
                self.preprocess = env_info.preprocess
                if self.preprocess:
                    self.gym_wrapper = AtariWrapper(env, frame_skip=env_info.frame_skip,screen_width=self.screen_width,screen_height=self.screen_height,scale_obs=True,crop_height_factor=env_info.crop_height_factor)
                else:
                    self.gym_wrapper = AtariWrapper(env, frame_skip=env_info.frame_skip,screen_width=self.screen_width,screen_height=self.screen_height,scale_obs=False, grayscale_obs=False)
            else:
                self.env_register = None
                self.type = "default"
                self.rendering_delay = 0.0
                self.preprocess = False

                env = gym.make(environment_name)

                self.gym_wrapper = GenericWrapper(env)

        self.debug = show_debug_info

    def step(self,action,render=False):
        observation, reward, done, info = self.gym_wrapper.step(action)
        if render: 
            self.render()
        return observation, reward, done, info

    def close(self):
        self.gym_wrapper.close()    

    def reset(self):
        return self.gym_wrapper.reset()

    def render(self):
        if self.preprocess:
            self.gym_wrapper.render(self.show_preprocessed)
        else:
            self.gym_wrapper.render()

    def rollout(self, agent, nSteps, render=False, delay=0.0):
        rollout_info = []

        #get first observation
        observation = self.reset()
        done = False
        for i in range(nSteps):
            #Compute next agentStep
            action, action_probabilities = agent.act(observation)

            #Perform environment step
            observation, reward, done, info = self.gym_wrapper.step(action)

            #Render environment after action
            if render:
                self.render()
            
            #Save informations to info list
            rollout_info.append(RolloutTuple(observation,reward,action,action_probabilities))

            #Terminate prematurely if environment is DONE
            if done:
                break
        
        return rollout_info, done

    ## NOTICE PARAMETERS THAT MAY BECOME GLOBAL ##
    ##delay
    def collect_rollouts(self, agent, nRollouts, nSteps, render=False):
        '''
        Collects nRollouts rollouts (different execution of an environment), each one of
        nSteps using a certain agent
        Parameters:
            environment: the gym environment to simulate
            agent: the policy that generates the actions to perform in the simulation
            nRollouts: number of trajectories to collect
            nSteps: number of steps in each trajectory
        Return:
            A dictionary containing:
            -rollouts: a 
        '''
        rollouts = []
        for i in range(nRollouts):
            if self.debug: print("Rollout #"+str(i))
            rollout, done = self.rollout(agent, nSteps, render=render, delay=self.rendering_delay)
            rollouts.append(rollout)
            
            #Terminate prematurely if environment is DONE
            if done:
                continue
        
        return rollouts, done

    def render_agent(self, agent):
        observation = self.reset()
        done = False
        while not done:
            action, _ = agent.act(observation)
            observation, reward, done, info = self.step(action)
            time.sleep(self.rendering_delay)
            if done: break
            self.render()
    
    def get_observation_shape(self):
        #return self.space_converter(self.observation_space)
        return self.observation_space.shape

    def get_action_shape(self):
        #return self.space_converter.(self.action_space)
        return self.action_space.n