import os
import glob
import gym
import numpy as np
import time
import math
from collections import namedtuple
import errno



from matplotlib import pyplot as plt 
from scipy.signal import lfilter

from utils import *
from EnvironmentNew import Environment
from Models import Policy, Value
from Logger import Logger
import tensorflow as tf
from tensorflow import GradientTape
from tensorflow import keras

#sets to float64 to avoid compatibility issues between numpy 
# (whose default is float64) and keras(whose default is float32)
keras.backend.set_floatx("float64")

#RolloutStatistics = namedtuple("RolloutStatistics","actions action_probabilities rewards discounted_rewards mean_discounted_rewards observations advantages size")
RolloutStatistics = namedtuple("RolloutStatistics","actions action_probabilities rewards discounted_rewards observations size")
TrainingInfo = namedtuple("TrainingInfo","mean_value_loss linesearch_successful mean_kl_divergence")

THREADS_NUMBER=2

class TRPOAgent:
    def __init__(self, 
    env,
    logger,
    
    #Rollouts
    steps_per_rollout=1000, 
    steps_between_rollouts=5, 
    rollouts_per_sampling=16, 
    multithreaded_rollout=False,
    full_rollout_episodes=0,
    rollout_until_success_episode=None,

    single_batch_training=False,

    #Training
    batch_size = 1000,

    #Coefficients
    #conjugate_gradients_damping=0.001, 
    conjugate_gradients_damping = 0.001,
    conjugate_gradients_iterations = 10,
    DELTA=0.01, 
    #DELTA=0.02,
    backtrack_coeff=0.6,
    #backtrack_coeff=1e-3, 
    backtrack_iters=10, 
    #backtrack_iters=5,
    #GAMMA = 0.99,
    GAMMA= 0.99,
    LAMBDA = 0.95,
    value_learning_rate=1e-2,
    value_function_fitting_epochs=5,
    epsilon_greedy=True,
    epsilon=0.2,
    epsilon_decrease_factor=0.005,

    #Debug
    debug_rollouts=False, 
    debug_act=False, 
    debug_training=False, 
    debug_model=False, 
    debug_learning=False,

    render=False,

    #Others
    model_backup_dir="TRPO_project/Models"):

        assert(env!=None), "You need to create an environment first"
        self.env = env
        self.model_backup_dir = model_backup_dir+"/"+self.env.environment_name

        #*******
        #LOGGING
        #*******
        #self.debug = show_debug_info
        #if debug_rollouts:
        #    self.logger.add_debug_channel("rollouts")
        #if debug_act:ir+"/"+self.env.environment_name

        #assert(logger!=None), "You need to create a logger first"
        self.logger = logger

        #*******
        #LOGGING
        #*******
        #self.debug = show_debug_info
        #if debug_rollouts:
        #    self.logger.add_debug_channel("rollouts")
        #if debug_act:
        #    self.logger.add_debug_channel("act")
        #if debug_training:
        #    self.logger.add_debug_channel("training")
        #if debug_learning:
        #    self.logger.add_debug_channel("learning")

        #************
        #COEFFICIENTS
        #************
        #Rewards discount factor
        self.GAMMA = GAMMA
        self.LAMBDA = LAMBDA

        #Conjugate gradients damping
        self.conjugate_gradients_damping = conjugate_gradients_damping
        self.conjugate_gradients_iterations = conjugate_gradients_iterations

        #Line search
        self.backtrack_coeff = backtrack_coeff
        self.backtrack_iters = backtrack_iters
        #self.EPS = 1e-8
        self.EPS = 1e-8
        self.DELTA = DELTA
        
        
        self.epsilon_greedy=epsilon_greedy
        self.epsilon=epsilon
        self.epsilon_decrease_factor = epsilon_decrease_factor
        
        #********
        #ROLLOUTS
        #********
        self.rollouts_per_sampling = rollouts_per_sampling
        self.steps_per_rollout = steps_per_rollout
        self.steps_between_rollouts = steps_between_rollouts
        self.steps_to_new_rollout = 0
        self.steps_since_last_rollout = self.steps_between_rollouts
        self.multithreaded_rollout=multithreaded_rollout
        self.full_rollout_episodes=full_rollout_episodes
        self.full_rollout_size = self.env.gym_wrapper.env._max_episode_steps
        self.rollout_statistics = []

        self.single_batch_training = single_batch_training

        #********
        #TRAINING
        #********
        self.batch_size = batch_size

        #This will get updated every training_step
        self.policy = Policy(self.env, self.logger, debug=debug_model)
        #This will be updated for every batch cycle inside the training step, in order to compute the ratio
        #between the new and the old policy in the surrogate function
        self.new_model = self.policy.clone()

        self.state_value = Value(self.env, self.logger, value_learning_rate, debug=debug_model)
        self.value_function_fitting_epochs = value_function_fitting_epochs

        self.theta_old = self.policy.get_flat_params()
        if steps_per_rollout==0: self.steps_per_rollout = self.env.max_episode_steps

        self.last_action=None

        self.render=render

        self.actions, self.action_probabilities, self.observations, self.discounted_rewards, self.total_rollout_elements = None, None, None, None, None


    def collect_rollout_statistics(self, steps_per_rollout, rollouts_per_sampling, multithreaded=False, only_successful=False, rollout_until_success=False):
        if rollout_until_success:
            actions, action_probabilities, observations, rewards = env.collect_rollouts_multithreaded_only_successful(self,rollouts_per_sampling,steps_per_rollout,THREADS_NUMBER)
        else: 
            actions, action_probabilities, observations, rewards = env.collect_rollouts_multithreaded(self,rollouts_per_sampling,steps_per_rollout,THREADS_NUMBER)

        def discount_rewards(rewards, GAMMA):
            discounted_rewards=[]
            for i, rollout_rewards in enumerate(rewards):
                discounted_reward=0
                discounted_rewards.append([])
                gamma = 1
                for reward in reversed(rollout_rewards):
                    discounted_reward = discounted_reward + gamma*reward
                    gamma = gamma * self.GAMMA
                    discounted_rewards[i].insert(0, discounted_reward)
                        
            return discounted_rewards

        discounted_rewards = discount_rewards(rewards,self.GAMMA)
        assert(len(actions)==len(action_probabilities)==len(rewards)==len(discounted_rewards)==len(observations))
        total_elements = 0
        for i in range(len(actions)):
            total_elements += len(actions[i])

        return actions, action_probabilities, observations, rewards, discounted_rewards, total_elements

    def training_step(self, episode, old_loss):
        def get_mean_kl_divergence(params=None):
            model = self.policy
            if params is not None:
                model = self.new_model
            output = model(all_observations)
            new_policy_action_probabilities = tf.nn.softmax(output).numpy()
            old_output = self.policy(all_observations)
            old_policy_action_probabilities = tf.nn.softmax(old_output)
            #kl = tf.reduce_mean(tf.reduce_sum(tf.stop_gradient(old_policy_action_probabilities) * tf.stop_gradient(tf.math.log(old_policy_action_probabilities / new_policy_action_probabilities)), axis=1))
            kl = tf.reduce_mean(tf.reduce_sum(old_policy_action_probabilities * tf.math.log(old_policy_action_probabilities / new_policy_action_probabilities), axis=1))
            return kl

        def fisher_vector_product(step_direction_vector):
            def kl_gradients(params=None):
                model = self.policy
                if params is not None:
                    model = self.new_model
                
                kl_divergence_gradients = model.get_flat_gradients(get_mean_kl_divergence)
                grad_vector_product = tf.reduce_sum(kl_divergence_gradients * step_direction_vector)
                return grad_vector_product

            kl_flat_grad_grads = self.policy.get_flat_gradients(kl_gradients)


            return (kl_flat_grad_grads + (self.conjugate_gradients_damping * step_direction_vector)).numpy()

        def conjugate_gradients(fvp,pg, residual_tol=1e-5):
            result = np.zeros_like(pg)
            r = pg.copy()
            p = pg.copy()

            rdotr = np.dot(r, r)

            for i in range(self.conjugate_gradients_iterations):
                if logger!=None:
                    self.log("Conjugate gradients iteration: ",i, writeToFile=True,debug_channel="cg")

                temp_fisher_vector_product = fvp(p)
                
                print(temp_fisher_vector_product)
                alpha = rdotr / (np.dot(p, temp_fisher_vector_product))
                if logger!=None:
                    self.log("CG Hessian: ",alpha,debug_channel="cg")
                result += alpha * p
                r -= alpha * temp_fisher_vector_product
                new_rdotr = np.dot(r, r)
                #beta = new_rdotr / rdotr + EPS
                beta = new_rdotr / (rdotr +self.EPS)
                
                p = r + beta * p
                rdotr = new_rdotr
                if rdotr < residual_tol:
                    break
            return result

        def surrogate_function(params=None):
            model = self.policy
            if params is not None:
                model = self.new_model
            #Calcola la nuova policy usando la rete neurale aggiornata ad ogni batch cycle del training step
            output = model(all_observations)
            print(len(output))
            new_policy_action_probabilities = tf.nn.softmax(output)
            new_policy_action_probabilities = tf.reduce_sum(all_actions_one_hot * new_policy_action_probabilities, axis=1)

            #Calcola la vecchia policy usando la vecchia rete neurale salvata
            old_output = self.policy(all_observations)
            old_policy_action_probabilities = tf.nn.softmax(old_output)
            #pi(a|s) ovvero ci interessa solo l'azione che abbiamo compiuto, da cui il prodotto per actions_one_hot
            old_policy_action_probabilities = tf.reduce_sum(all_actions_one_hot * old_policy_action_probabilities, axis=1).numpy() +self.EPS


            policy_ratio = new_policy_action_probabilities / old_policy_action_probabilities # (Schulman et al., 2017) Equation(14)

            #self.log("policy ratio: ", policy_ratio,", advantage: ", advantage, writeToFile=True, debug_channel="surrogate")
            loss = tf.reduce_mean(policy_ratio * all_advantages)
            #self.log("loss value: ", loss, writeToFile=True, debug_channel="surrogate")

            return loss 

        def sample_loss(pg,oldtheta,newtheta):
            return pg.T.dot(newtheta-oldtheta)

        def logpi(observations, actions_one_hot, advantages):            
            output = self.policy(observations)
            pi = tf.nn.softmax(output)
            res = tf.reduce_sum(tf.math.log(tf.reduce_sum(actions_one_hot * pi, axis=1)) * advantages)
            return res

        def policy_gradient(observations,actions_one_hot,advantages):
            def logpi(params=None):
                model = self.policy
                if params is not None:
                    model = self.new_model            
                output = model(observations)
                pi = tf.nn.softmax(output)
                res = tf.reduce_sum(tf.math.log(tf.reduce_sum(actions_one_hot * pi, axis=1)) * advantages)
                return res
            #logpi_grad = self.policy.get_flat_gradients(logpi)
            logpi_grad = get_flat_gradients(logpi, self.policy.model.trainable_variables)

            return logpi_grad

        self.log("Rollout statistics size: ", self.total_rollout_elements, ", Batch size: ", self.batch_size,writeToFile=True,debug_channel="batch_info")
        self.log("\n\n***************\nBEGINNING TRAINING\n***************\n\n",writeToFile=True,debug_channel="learning")

        max_reward=-math.inf
        mean_reward=-math.inf

        policy_gradients = []
        state_values = []
        mean_state_values = []
        actions_one_hot = []
        advantages = []
        mean_rewards = []
        total_rewards = []
        for i in range(self.rollouts_per_sampling):
            mean_rewards.append(sum(self.rewards[i])/len(self.rewards))
            total_rewards.append(sum(self.rewards[i]))
            state_values.append(self.state_value(self.observations[i]))
            advantages.append(self.discounted_rewards[i] - state_values[i])
            mean_state_values.append(np.mean(state_values[i]))
            #Normalize advantages
            advantages[i] = (advantages[i] - advantages[i].mean())/(advantages[i].std() + self.EPS)

            #4.3) Current batch actions one hot
            actions_one_hot.append(tf.one_hot(self.actions[i], self.env.get_action_shape(), dtype="float64"))
            policy_gradients.append(policy_gradient(self.observations[i],actions_one_hot[i],advantages[i]).numpy())
        
            history = self.state_value.fit(self.observations[i], self.rewards[i], epochs=self.value_function_fitting_epochs, verbose=0)
            value_loss = history.history["loss"][-1]

        all_observations = np.concatenate(self.observations)
        all_actions_one_hot = np.concatenate(actions_one_hot)
        all_advantages = np.concatenate(advantages)
        
        print(surrogate_function())

        g = get_flat_gradients(surrogate_function,self.policy.model.trainable_variables).numpy().flatten()
        #g = tf.reduce_mean(policy_gradients, axis=0).numpy()
        stepdir = conjugate_gradients(fisher_vector_product, g)
        print(stepdir)
        shs = .5 * stepdir.dot(fisher_vector_product(stepdir))
        if np.isnan(shs).any(): return None, None, None, None
        #shs = g.T.dot(stepdir)
        print(shs)
        lm = np.sqrt(shs / self.DELTA)
        fullstep = stepdir / lm
        #neggdotstepdir = -g.dot(stepdir)
        #thprev = self.policy.get_flat_params()           
        fullstep = np.sqrt(2*self.DELTA/shs + self.EPS) * stepdir

        if np.isnan(fullstep).any():
            return


        def linesearch():
            theta = self.policy.get_flat_params()
            self.log("Old loss: ", old_loss, debug_channel="linesearch")
            for (_n_backtracks, step) in enumerate(self.backtrack_coeff**np.arange(self.backtrack_iters)):

                theta_new = theta + step * fullstep
                
                self.new_model.set_flat_params(theta_new)

                #new_loss = sample_loss(g,theta,theta_new)
                new_loss = surrogate_function(theta_new)
                #new_loss = logpi(all_observations, all_actions_one_hot, all_advantages).numpy()

                mean_kl_div = get_mean_kl_divergence()
                if mean_kl_div <= self.DELTA and new_loss>=0:
                #if mean_kl_div <= self.DELTA and new_loss>old_loss:
                #if mean_kl_div <= self.DELTA and new_loss<old_loss:
                    self.log("Linesearch worked at ", _n_backtracks, ", New mean kl div: ", mean_kl_div, ", New policy loss value: ", new_loss, writeToFile=True, debug_channel="linesearch")
                    return True, theta_new, new_loss
                if _n_backtracks == self.backtrack_iters - 1:
                    self.log("Linesearch failed. Mean kl divergence: ", mean_kl_div, ", Discarded policy loss value: ", new_loss, writeToFile=True, debug_channel="linesearch")
                    return False, theta, old_loss
                
        linesearch_success, theta, policy_loss = linesearch()        
        self.log("Current batch value loss: ",value_loss,writeToFile=True,debug_channel="batch_info")

        self.policy.set_flat_params(theta)
        mean_reward = np.mean(mean_rewards)

        max_reward = max(total_rewards)


        mean_state_val = np.mean(mean_state_values).flatten()

        self.log("END OF TRAINING STEP #", episode)
        self.log("TRAINING STEP STATS: Max reward: ",max_reward," Mean reward: ", mean_reward, ", Policy loss: ", policy_loss, ", Mean state val: ", mean_state_val, debug_channel="batch_info")

        self.steps_since_last_rollout += 1
        self.epsilon = self.epsilon - self.epsilon_decrease_factor

        return max_reward, mean_reward, value_loss, policy_loss

    #the ACT function: produces an action from the current policy
    def act(self,observation,last_action=None,epsilon_greedy=True):

        #Trasforma la observation in un vettore riga [[array]]
        observation = observation[np.newaxis, :]
        #self.log("observation: ",observation,", len: ",len(observation), writeToFile=True, debug_channel="act")

        #Estrai la prossima azione dalla NN della policy
        policy_output = self.policy(observation)
        
        action_probabilities = tf.nn.softmax(policy_output).numpy().flatten()
        #self.log("action_probabilities: ",action_probabilities,", len: ",len(action_probabilities), writeToFile=True, debug_channel="act")

        #SBAGLIATO!!!
        action = np.random.choice(range(action_probabilities.shape[0]), p=action_probabilities)
        
        #action = np.argmax(policy_output)

        exploitation_chance = np.random.uniform(0,1)
        last_action_repeat_chance = 0.8
        if self.epsilon_greedy and exploitation_chance < self.epsilon:
            if self.epsilon_greedy and np.random.uniform(0,1) < last_action_repeat_chance and last_action!=None:
                action = last_action
            else:
            #perform a random action
                action = np.random.randint(0,self.env.get_action_shape())
        
        #self.log("chosen_action: ",action, writeToFile=True, debug_channel="act")

        return action, action_probabilities

    def learn(self, nEpisodes, episodesBetweenModelBackups=-1, start_from_episode=0):
        initial_time = time.time()
        history_values = {"max_reward":[],"value_loss":[],"sample_loss":[]}
        loss_values = []

        plot, subplots = plt.subplots(3, 1, constrained_layout=True)

        #old_loss = math.inf
        old_loss = -math.inf
        for episode in range(start_from_episode,nEpisodes):
            self.log("Episode #", episode, writeToFile=True, debug_channel="learning")

            if start_from_episode>0:
                self.load_weights(start_from_episode)

            rollout_until_success = False
            current_episode_steps_per_rollout = self.steps_per_rollout

            #DECIDE WETHER ONLY SELECTING SUCCESSFUL ROLLOUTS OR NOT
            full_rollout_episode=episode<self.full_rollout_episodes
            
            if full_rollout_episode:
                rollout_until_success = True

            if full_rollout_episode:
                current_episode_steps_per_rollout = self.full_rollout_size-1000
                current_rollouts_per_sampling = math.ceil(self.rollouts_per_sampling / 2)
            else:
                current_episode_steps_per_rollout = self.steps_per_rollout
                current_rollouts_per_sampling = self.rollouts_per_sampling

            #UPDATE LATEST ROLLOUT INFO
            if self.steps_since_last_rollout == self.steps_between_rollouts:
                self.log("Performing rollouts: rollout length: ",self.steps_per_rollout,writeToFile=True,debug_channel="learning")
                self.steps_since_last_rollout = 0

                self.actions, self.action_probabilities, self.observations, self.rewards, self.discounted_rewards, self.total_rollout_elements = self.collect_rollout_statistics(current_episode_steps_per_rollout, current_rollouts_per_sampling, multithreaded=self.multithreaded_rollout,rollout_until_success=rollout_until_success)

                self.log("Rollouts performed",writeToFile=True,debug_channel="learning")

            max_reward, mean_reward, value_loss, policy_loss = self.training_step(episode, old_loss)

            if max_reward==None or mean_reward==None or value_loss==None or policy_loss==None: continue            

            #history_entry = {"max_reward":max_reward,"value_loss":value_loss,"policy_loss":policy_loss}

            history_values["max_reward"].append(max_reward)
            history_values["value_loss"].append(value_loss)
            history_values["sample_loss"].append(policy_loss)

            self.log_history(history_values)
            self.plot_history(plot, subplots, history_values)

            old_loss = policy_loss

            if self.render: 
                env.render_agent(agent, nSteps = self.steps_per_rollout)
            
            import os
            #print(os.getcwd())
            if episodesBetweenModelBackups!=-1 and episode!=0 and episode%episodesBetweenModelBackups == 0:
                self.save_policy_weights(episode)
                self.save_value_weights(episode)
        return history
            
    def save_policy_weights(self,episode):
        filename = self.model_backup_dir+"/Policy."+self.env.get_environment_description()+"."+str(episode)
        #Create directory if it doesn't exist
        try:
            os.makedirs(self.model_backup_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        self.policy.save_model_weights(filename)

    def save_value_weights(self,episode):
        filename = self.model_backup_dir+"/Value."+self.env.get_environment_description()+"."+str(episode)
        #Create directory if it doesn't exist
        try:
            os.makedirs(self.model_backup_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        self.state_value.save_model_weights(filename)

    def log(self, *strings, writeToFile=False, debug_channel="Generic"):
        strings = [str(s) for s in strings]
        if self.logger!=None:
            self.logger.print(strings, writeToFile=writeToFile, debug_channel=debug_channel)

    def load_weights(self, episode, dir=""):
        if dir=="": dir = self.model_backup_dir
        policy_filename = "/Policy."+self.env.get_environment_description()+"."+str(episode)
        self.policy.load_model_weights(dir+policy_filename)

        value_filename = "/Value."+self.env.get_environment_description()+"."+str(episode)
        self.state_value.load_model_weights(dir+value_filename)
    
    def load_history(self):
        self.logger.load_history()

    def log_history(self,live_plots=False,**history):
        self.logger.log_history(**history,live_plots=live_plots)

    def plot_history(self, plot, subplots, history):
        current_plot=0
        for k,v in history.items():
            subplots[current_plot].plot(v)
            subplots[current_plot].set_title(k)
            subplots[current_plot].set_xlabel('Episode')
            subplots[current_plot].set_ylabel(k)
            plot.suptitle(k, fontsize=16)
            current_plot+=1

            plt.draw()
            plt.pause(0.001)

if __name__ == '__main__':
    import numpy
    import sys
    #numpy.set_printoptions(threshold=sys.maxsize)
    
    #Disable tensorflow debugging info
    import tensorflow as tf
    import datetime
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
    
    channels = [
        #"rollouts",
        #"advantages",
        #"rollouts_dump",
        "act",
        #"training",
        "batch_info",
        "linesearch",
        "learning",
        "thread_rollouts",
        #"model",
        #"utils",
        #"utils_kl",
        #"cg",
        #"GAE",
        #"surrogate",
        #"EnvironmentRegister",
        #"environment"
        ]

    #tf.set_random_seed(0)
    np.random.seed(0)


    #CartPole-v1
    #env_name = "CartPole-v0"
    #logger = Logger(name=env_name,log_directory="TRPO_project/Models/"+env_name,debugChannels=channels)
    #desired_rollouts = 30
    #env = Environment(env_name,logger,desired_rollouts,use_custom_env_register=True, debug=True, show_preprocessed=False, same_seed=True)
    #agent = TRPOAgent(env,logger,steps_per_rollout=1000,steps_between_rollouts=1, rollouts_per_sampling=desired_rollouts,multithreaded_rollout=True, full_rollout_episodes=0,single_batch_training = True, batch_size=100,DELTA=0.01, epsilon=0.15, epsilon_greedy=True,value_function_fitting_epochs=1, value_learning_rate=1e-3,backtrack_coeff=0.8, backtrack_iters=5,render=True)

    #Acrobot-v0
    #env_name = "Acrobot-v1"
    #logger = Logger(name=env_name,log_directory="TRPO_project/Models/"+env_name,debugChannels=channels)
    #desired_rollouts = 30
    #env = Environment(env_name,logger,desired_rollouts,use_custom_env_register=True, debug=True, show_preprocessed=False, same_seed=True)
    #agent = TRPOAgent(env,logger,steps_per_rollout=1000,steps_between_rollouts=1, rollouts_per_sampling=desired_rollouts,multithreaded_rollout=True, full_rollout_episodes=0,single_batch_training = True, batch_size=100,DELTA=0.01, epsilon=0.15, epsilon_greedy=True,value_function_fitting_epochs=1, value_learning_rate=1e-3,backtrack_coeff=0.8, backtrack_iters=5,render=True)
    
    #MountainCar-v0
    env_name = "MountainCar-v0"
    logger = Logger(name=env_name,log_directory="TRPO_project/Models/"+env_name,debugChannels=channels)
    desired_rollouts = 12
    env = Environment(env_name,logger,desired_rollouts,use_custom_env_register=True, debug=True, show_preprocessed=False, same_seed=True)
    agent = TRPOAgent(env,logger,steps_per_rollout=1600,steps_between_rollouts=1, rollouts_per_sampling=desired_rollouts,multithreaded_rollout=True, full_rollout_episodes=0,single_batch_training = True, batch_size=1000,DELTA=0.01, epsilon=0.4, epsilon_greedy=True, epsilon_decrease_factor=0.005, value_function_fitting_epochs=5, value_learning_rate=1e-2,backtrack_coeff=0.6, backtrack_iters=10,render=True)
    
    #MsPacmanPreprocessed-v0
    #env_name = "MsPacmanPreprocessed-v0"
    #logger = Logger(name=env_name,log_directory="TRPO_project/Models/"+env_name,debugChannels=channels)
    #desired_rollouts = 10
    #env = Environment(env_name,logger,desired_rollouts,use_custom_env_register=True, debug=True, show_preprocessed=False, same_seed=True)
    #agent = TRPOAgent(env,logger,steps_per_rollout=1500,steps_between_rollouts=1, rollouts_per_sampling=15, multithreaded_rollout=True, batch_size=4500, DELTA=0.01)

    #CartPole-v0

    #MountainCar-v0

    history = agent.learn(500,episodesBetweenModelBackups=1,start_from_episode=0)

        
#    #TODO:PROBLEMI:
#1) La divergenza diventa troppo piccola, dunque lo stesso per l'hessiano, dunque l'improvement diventa nullo e il miglioramento casuale
#Soluzione tentata: sommare  + EPS alla divergenza
#2) Se faccio questa cosa del EPS, iniziano a spuntare fuori i nan
#Soluzione tentata: eseguire un controllo sui nan nei nuovi parametri da assegnare alla rete neurale (trovato su internet anche se non so se sia efficace)
#Da testare:
#Altrimenti risalire alla fonte: Nuova soluzione: https://github.com/tensorforce/tensorforce/issues/26 Controllare quando shs<0 e skippare l'update in quel caso
#(infatti viene mostrato l'errore di runtime di sqrt impossibile)
#CAUSA: In CONJUGATE GRADIENTS KL divergence gradients negativi

#ERRORI:
#L'azione deve essere campionata con argmax!!!
#Normalizzazione delle osservazioni e TILE CLIPPING