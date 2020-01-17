import gym
from gym.envs.registration import registry, register, make, spec
from collections import namedtuple

#Named tuple used to store useful info about the custom environment being registered
EnvironmentInfo = namedtuple("EnvironmentInfo","custom_id Type")
class CustomEnvironmentRegister:
    def __init__(self):
        self.registeredEnvironments = {}
        
        register(
            id='MountainCarCustom-v0',
            entry_point='gym.envs.classic_control:MountainCarEnv',
            max_episode_steps=1000,      # MountainCar-v0 uses 200
            #reward_threshold=-110.0,
        )
        self.registeredEnvironments["MountainCar-v0"] = EnvironmentInfo("MountainCarCustom-v0","classic")
    
    def register_environment(self, config_path):
        #Implement uploading environment from path
        return

    def get_environment(self, env_name):
        #Check if we have registered a custom version
        if env_name in self.registeredEnvironments:
            return gym.make(self.registeredEnvironments[env_name].custom_id), self.registeredEnvironments[env_name].type
        else:
            return gym.make(env_name)