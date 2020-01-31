if __name__ == '__main__':
    import numpy
    import sys
    #numpy.set_printoptions(threshold=sys.maxsize)
    
    #Disable tensorflow debugging info
    import tensorflow as tf
    import datetime

    from Logger import *
    from TRPOAgent import *
    from Environment import *
    
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
    env_name = "MountainCar-v0"
    
    channels = []

    logger = Logger(name=env_name,log_directory="TRPO_project/Models/"+env_name,debugChannels=channels)

    env = Environment(env_name,logger,use_custom_env_register=True, debug=True, show_preprocessed=False)

    starting_episode = 180
    
    agent = TRPOAgent(env,None,steps_per_rollout=512, steps_between_rollouts=1, rollouts_per_sampling=8, multithreaded_rollout=True, batch_size=512, DELTA=0.01, epsilon=0.0, epsilon_greedy=False)
        
    agent.load_weights(starting_episode)

    env.render_agent(agent)