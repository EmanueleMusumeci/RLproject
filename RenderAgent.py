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
    
    channels = [
        #"rollouts",
        #"advantages",
        #"rollouts_dump",
        #"act",
        #"training",
        "batch_info",
        "linesearch",
        "learning",
        "thread_rollouts",
        #"model",
        #"utils",
        "utils_kl",
        "cg",
        #"surrogate",
        #"EnvironmentRegister",
        #"environment"
        ]

    logger = Logger(name=env_name,log_directory="TRPO_project/Models/"+env_name,debugChannels=channels)

    env = Environment(env_name,logger,use_custom_env_register=True, debug=True, show_preprocessed=False)

    agent = TRPOAgent(env,logger,steps_per_rollout=512,steps_between_rollouts=1, rollouts_per_sampling=8, multithreaded_rollout=True, batch_size=512, DELTA=0.01,
    debug_rollouts=True, debug_act=True, debug_training=True, debug_model=True, debug_learning=True)
    
    #initial_time = time.time()
    #env.collect_rollouts(agent,10,1000)
    #first_time = time.time()
    #env.collect_rollouts_multithreaded(agent,10,1000,15)
    #second_time = time.time()

    #print("Non-multithreaded: ",first_time-initial_time,", multithreaded: ",second_time-first_time)
    
    #agent.training_step(0)
    #history = agent.learn(500,episodesBetweenModelBackups=5,start_from_episode=0)
    #plt.plot(history)
    #plt.show()
        
    agent.load_weights(5)

    env.render_agent(agent)