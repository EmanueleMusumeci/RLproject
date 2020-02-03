import os
import glob
import gym
import numpy as np
import time
import math
from collections import namedtuple
import errno
import argparse


from matplotlib import pyplot as plt 
from scipy.signal import lfilter

from utils import *
from Environment import Environment
from Models import Policy, Value
from Logger import Logger
import tensorflow as tf
from tensorflow import GradientTape
from tensorflow import keras

#sets to float64 to avoid compatibility issues between numpy 
# (whose default is float64) and keras(whose default is float32)
keras.backend.set_floatx("float64")

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

    single_batch_training=False,

    #Training
    batch_size = 1000,

    #Coefficients
    conjugate_gradients_damping = 0.001,
    conjugate_gradients_iterations = 10,
    DELTA=0.01, 
    backtrack_coeff=0.6, 
    backtrack_iters=10,
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

    #Others
    model_backup_dir="Models/"):

         
        assert(env!=None), "You need to create an environment first"
        self.env = env

        if model_backup_dir=="Models/":
            dirname = os.path.dirname(__file__)
            self.model_backup_dir = os.path.join(dirname, 'Models')+"/"+self.env.environment_name
        else:
            self.model_backup_dir = model_backup_dir
        #*******
        #LOGGING
        #*******
        #self.debug = show_debug_info
        #if debug_rollouts:
        #    self.logger.add_debug_channel("rollouts")
        #if debug_act:ir+"/"+self.env.environment_name

        #assert(logger!=None), "You need to create a logger first"
        self.logger = logger

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

        self.actions, self.action_probabilities, self.observations, self.discounted_rewards, self.total_rollout_elements = None, None, None, None, None


    def collect_rollout_statistics(self, steps_per_rollout, rollouts_per_sampling, multithreaded=False):
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
                self.new_model.set_flat_params(params)

            output = model(all_observations)
            new_policy_action_probabilities = tf.nn.softmax(output).numpy()
            old_output = self.policy(all_observations)
            old_policy_action_probabilities = tf.nn.softmax(old_output)
            
            kl = tf.reduce_mean(tf.reduce_sum(old_policy_action_probabilities * tf.math.log(old_policy_action_probabilities / new_policy_action_probabilities), axis=1))
            return kl

        def fisher_vector_product(step_direction_vector):
            def kl_gradients(params=None):
                model = self.policy
                if params is not None:
                    model = self.new_model
                    self.new_model.set_flat_params(params)

                kl_divergence_gradients = get_flat_gradients(get_mean_kl_divergence,model.model.trainable_variables)
                grad_vector_product = tf.reduce_sum(kl_divergence_gradients * step_direction_vector)
                return grad_vector_product

            kl_flat_grad_grads = get_flat_gradients(kl_gradients, self.policy.model.trainable_variables)

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

                alpha = rdotr / (np.dot(p, temp_fisher_vector_product))
                if logger!=None:
                    self.log("CG Hessian: ",alpha,debug_channel="cg")
                result += alpha * p
                r -= alpha * temp_fisher_vector_product
                new_rdotr = np.dot(r, r)

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
                self.new_model.set_flat_params(params)

            #Compute the new policy using the neural net update at each batch cycle of the training step
            output = model(current_batch_observations)
            
            new_policy_action_probabilities = tf.nn.softmax(output)
            new_policy_action_probabilities = tf.reduce_sum(current_batch_actions_one_hot * new_policy_action_probabilities, axis=1)

            #Compute the old policy using the old neural network 
            old_output = self.policy(current_batch_observations)
            old_policy_action_probabilities = tf.nn.softmax(old_output)
            #pi(a|s) has to be an array made by the probability of the action the agent made in state s. To enable Tensorflow to build the session graph
            #we only need to use Tensorflow primitives, therefore the need to build the Policy probability vector in this way
            old_policy_action_probabilities = tf.reduce_sum(current_batch_actions_one_hot * old_policy_action_probabilities, axis=1).numpy() +self.EPS


            policy_ratio = new_policy_action_probabilities / old_policy_action_probabilities # (Schulagrange_multiplieran et al., 2017) Equation(14)

            loss = tf.reduce_mean(policy_ratio * current_batch_advantages) # (Schulagrange_multiplieran et al., 2017) Equation(14)

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
                    self.new_model.set_flat_params(params)
          
                output = model(observations)
                pi = tf.nn.softmax(output)
                res = tf.reduce_sum(tf.math.log(tf.reduce_sum(actions_one_hot * pi, axis=1)) * advantages)
                return res

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
            mean_rewards.append(np.mean(sum(self.rewards[i])))
            total_rewards.append(sum(self.rewards[i]))
            state_values.append(self.state_value(self.observations[i]))
            advantages.append(self.discounted_rewards[i] - state_values[i])
            mean_state_values.append(np.mean(state_values[i]))

            #Normalize advantages: this proved to made the convergence less brittle
            advantages[i] = (advantages[i] - advantages[i].mean())/(advantages[i].std() + self.EPS)

            #Current batch actions one hot
            actions_one_hot.append(tf.one_hot(self.actions[i], self.env.get_action_shape(), dtype="float64"))
            
            #Other way of computing the policy gradients, sample inefficient (see report Section 2)
            #policy_gradients.append(policy_gradient(self.observations[i],actions_one_hot[i],advantages[i]).numpy())

        all_observations = np.concatenate(self.observations)
        all_actions_one_hot = np.concatenate(actions_one_hot)
        all_advantages = np.concatenate(advantages)
        all_rewards = np.concatenate(self.rewards)
        all_discounted_rewards = np.concatenate(self.discounted_rewards)

        #history = self.state_value.fit(all_observations, all_rewards, epochs=self.value_function_fitting_epochs, verbose=0)
        history = self.state_value.fit(all_observations, all_discounted_rewards, epochs=self.value_function_fitting_epochs, verbose=0)
        value_loss = history.history["loss"][-1]

        total_elements = len(all_observations)

        #Compute the number of batches
        number_of_batches = math.ceil(total_elements / self.batch_size)

        #Collect useful statistics to plot
        policy_losses = []

        #Split all rollout data among batches
        current_batch_actions_one_hot = []
        current_batch_observations = []
        current_batch_advantages = []
        for batch in range(number_of_batches):

            policy_loss = 0
            #Data for current batch 
            current_batch_actions_one_hot = all_actions_one_hot[batch*self.batch_size:(batch+1)*self.batch_size]
            current_batch_observations = all_observations[batch*self.batch_size:(batch+1)*self.batch_size]
            current_batch_advantages = all_advantages[batch*self.batch_size:(batch+1)*self.batch_size]

            self.log("Batch #", batch, ", batch length: ", len(current_batch_observations), writeToFile=True, debug_channel="batch_info")

            #print(surrogate_function())

            g = get_flat_gradients(surrogate_function,self.policy.model.trainable_variables).numpy().flatten()

            gradient_step_direction = conjugate_gradients(fisher_vector_product, g)
            
            sT_H_s = .5 * gradient_step_direction.dot(fisher_vector_product(gradient_step_direction))
            
            #If something went wrong leading to nan values, skip this training step
            if np.isnan(sT_H_s).any(): return None, None, None, None, None
            
            lagrange_multiplier = np.sqrt(sT_H_s / self.DELTA)
            
            max_gradient_step = gradient_step_direction / lagrange_multiplier

            #Alternative way of evaluating the termination condition for the backtracking line search, unused
            #neggdotgradient_step_direction = -g.dot(gradient_step_direction)
            
            max_gradient_step = np.sqrt(2*self.DELTA/sT_H_s + self.EPS) * gradient_step_direction
            
            if np.isnan(max_gradient_step).any():
                return

            def linesearch():
                theta = self.policy.get_flat_params()
                self.log("Old loss: ", old_loss, debug_channel="linesearch")
                for (_n_backtracks, step) in enumerate(self.backtrack_coeff**np.arange(self.backtrack_iters)):

                    theta_new = theta + step * max_gradient_step
                    
                    #new_loss = sample_loss(g,theta,theta_new)
                    new_loss = surrogate_function(theta_new)
                    #new_loss = logpi(all_observations, all_actions_one_hot, all_advantages).numpy()

                    mean_kl_div = get_mean_kl_divergence(theta_new)

                    if mean_kl_div <= self.DELTA and new_loss>=0:
                    #if mean_kl_div <= self.DELTA and new_loss>old_loss:
                        self.log("Linesearch worked at ", _n_backtracks, ", New mean kl div: ", mean_kl_div, ", New policy loss value: ", new_loss, writeToFile=True, debug_channel="linesearch")
                        return True, theta_new, new_loss.numpy()
                    if _n_backtracks == self.backtrack_iters - 1:
                        self.log("Linesearch failed. Mean kl divergence: ", mean_kl_div, ", Discarded policy loss value: ", new_loss, writeToFile=True, debug_channel="linesearch")
                        return False, theta, old_loss
                    
            linesearch_success, theta, policy_loss = linesearch()
                       
            self.log("Current batch value loss: ",value_loss,writeToFile=True,debug_channel="batch_info")
            self.policy.set_flat_params(theta)
            old_loss = policy_loss

            policy_losses.append(policy_loss)
        
        mean_reward = np.mean(mean_rewards)
        max_reward = max(total_rewards)
        mean_state_val = np.mean(mean_state_values).flatten()

        self.log("END OF TRAINING STEP #", episode)
        self.log("TRAINING STEP STATS: Max reward: ",max_reward," Mean reward: ", mean_reward, ", Policy loss: ", policy_loss, ", Mean state val: ", mean_state_val, debug_channel="batch_info")

        self.steps_since_last_rollout += 1
        self.epsilon = self.epsilon - self.epsilon_decrease_factor

        return max_reward, mean_reward, value_loss, policy_losses, policy_loss

    #the ACT function: produces an action from the current policy
    def act(self,observation,last_action=None,epsilon_greedy=True):

        #Trasforma la observation in un vettore riga [[array]]
        observation = observation[np.newaxis, :]
        
        #Estrai la prossima azione dalla NN della policy
        policy_output = self.policy(observation)
        
        action_probabilities = tf.nn.softmax(policy_output).numpy().flatten()
        
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
        
        return action, action_probabilities

    def learn(self, nEpisodes, episodesBetweenModelBackups=-1, episodesBetweenAgentRenders=-1, start_from_episode=0):
        initial_time = time.time()
        loss_values = []

        old_loss = -math.inf

        if start_from_episode>0:
            self.load_weights(start_from_episode)
            self.logger.load_history(to_episode=start_from_episode)

        for episode in range(start_from_episode,nEpisodes):
            self.log("Episode #", episode, writeToFile=True, debug_channel="learning")

            current_episode_steps_per_rollout = self.steps_per_rollout
            current_rollouts_per_sampling = self.rollouts_per_sampling

            #UPDATE LATEST ROLLOUT INFO
            if self.steps_since_last_rollout == self.steps_between_rollouts:
                self.log("Performing rollouts: rollout length: ",current_episode_steps_per_rollout,writeToFile=True,debug_channel="learning")
                self.steps_since_last_rollout = 0

                self.actions, self.action_probabilities, self.observations, self.rewards, self.discounted_rewards, self.total_rollout_elements = self.collect_rollout_statistics(current_episode_steps_per_rollout, current_rollouts_per_sampling, multithreaded=self.multithreaded_rollout)
                self.log("Rollouts performed",writeToFile=True,debug_channel="learning")

            max_reward, mean_reward, value_losses, policy_losses, last_policy_loss = self.training_step(episode, old_loss)

            if max_reward==None or mean_reward==None or value_losses==None or policy_losses==None or last_policy_loss==None: continue            

            mean_value_loss = np.mean(value_losses)
            mean_policy_loss = np.mean(policy_losses)

            history_values = {
                "Max reward" : max_reward,
                "Mean reward" : mean_reward,
                "Sample loss" : last_policy_loss,
                "Value loss" : mean_value_loss
            }
            self.log_history(**history_values)

            #save latest policy loss as "old_loss"
            old_loss = last_policy_loss

            if episodesBetweenAgentRenders>0 and episode%episodesBetweenAgentRenders == 0:
                env.render_agent(agent, nSteps = self.steps_per_rollout, epsilon_greedy=False)
            
            if episodesBetweenModelBackups>0 and episode%episodesBetweenModelBackups == 0:
                self.save_policy_weights(episode)
                self.save_value_weights(episode)
            
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

    def log_history(self,**history):
        self.logger.log_history(**history)


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
        "training",
        "batch_info",
        "linesearch",
        "learning",
        "thread_rollouts",
        #"model",
        "model_summary",
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

    parser = argparse.ArgumentParser(description="Run TRPO Agent. Specify at least one running mode between 'train' and 'run'.")
    parser.add_argument('env', help="Environment where the agent is trained/run")
    subparsers = parser.add_subparsers(description="Use '<environment_name> train -h' or '<environment_name> run -h' to receive more information on usage", dest='command')
    train_parser = subparsers.add_parser('train', description="Train the agent on the specified environment")
    train_parser.add_argument("-episodes", help="Number of episodes of training (default: 1000)", default=1000, type=int)
    train_parser.add_argument("-from_episode", help="Episode from which training is resumed (default: 0, there has to be a backup from that episode)", default=0, type=int)
    train_parser.add_argument("-render", help="Number of training steps between two agent renders (default: -1, it means no renders)", default = -1, type=int)
    train_parser.add_argument("-backup_every", help="Number of training steps between two agent model backups (default: 1, a backup is saved every step)", default=1, type=int)
    train_parser.add_argument("-save_dir", help="If specified, represents the ABSOLUTE path to the directory where the agent models, training logs and plots will be saved (two files are produced for each model: Value and Policy weights)(default: <current-directory>/Models/<environment-name>)", default="Models/", type=str)
    run_parser = subparsers.add_parser("run", description="Run the agent on the specified environment")
    run_parser.add_argument("episode",help="REQUIRED: agent episode to run", type=int)
    run_parser.add_argument("-to_episode", help="If specified, represents the last episode to render, forming the range [episode,to-episode]",default=-1, type=int)
    run_parser.add_argument("-skip", help="If specified, represents the number of episodes to skip between two subsequent agent runs (default: 0)", default=0, type=int)
    run_parser.add_argument("-load_dir", help="If specified, represents the ABSOLUTE path to the dir containing the models to load (default: <current-directory>/Models/<environment-name>). Please notice: the weights have to be generated by the training process of this agent", default="", type=str)
    parser.add_argument("-no_plot", help="Disable live plot (default: no, plot is enabled by default)", default=False, action="store_true")
    args = parser.parse_args()

    #print(args)

    logger = Logger(name=args.env,log_directory="TRPO_project/Models/"+args.env,debugChannels=channels, live_plot=(not args.no_plot))
    env = Environment(args.env,logger,use_custom_env_register=True, debug=True, show_preprocessed=False, same_seed=True)

    if args.command == "train":
        print("HELLO")
        if args.env=="CartPole-v1":
            #CartPole-v1
            desired_rollouts = 30
            agent = TRPOAgent(env,logger,steps_per_rollout=1000,steps_between_rollouts=1, rollouts_per_sampling=desired_rollouts,multithreaded_rollout=True, single_batch_training = True, batch_size=1000,DELTA=0.01, epsilon=0.15, epsilon_greedy=True,value_function_fitting_epochs=1, value_learning_rate=1e-3,backtrack_coeff=0.8, backtrack_iters=5, model_backup_dir=args.save_dir)
        elif args.env=="LunarLander-v2":
            #LunarLander-v2
            desired_rollouts = 30
            agent = TRPOAgent(env,logger,steps_per_rollout=1000,steps_between_rollouts=1, rollouts_per_sampling=desired_rollouts,multithreaded_rollout=True, single_batch_training = True, batch_size=1000,DELTA=0.01, epsilon=0.2, epsilon_decrease_factor=0.005, epsilon_greedy=True,value_function_fitting_epochs=5, value_learning_rate=1e-2,backtrack_coeff=0.6, backtrack_iters=10, model_backup_dir=args.save_dir)
        elif args.env=="Acrobot-v0":
            #Acrobot-v0
            desired_rollouts = 30
            agent = TRPOAgent(env,logger,steps_per_rollout=1000,steps_between_rollouts=1, rollouts_per_sampling=desired_rollouts,multithreaded_rollout=True, single_batch_training = True, batch_size=500,DELTA=0.01, epsilon=0.15, epsilon_greedy=True,value_function_fitting_epochs=1, value_learning_rate=1e-3,backtrack_coeff=0.8, backtrack_iters=5, model_backup_dir=args.save_dir)
        elif args.env=="MountainCar-v0":
            #MountainCar-v0
            desired_rollouts = 10
            print("HELLO2")
            agent = TRPOAgent(env,logger,steps_per_rollout=2000,steps_between_rollouts=1, rollouts_per_sampling=desired_rollouts,multithreaded_rollout=True, single_batch_training = True, batch_size=4000,DELTA=0.01, epsilon=0.4, epsilon_greedy=True, epsilon_decrease_factor=0.005, value_function_fitting_epochs=5, value_learning_rate=1e-2,backtrack_coeff=0.6, backtrack_iters=10, model_backup_dir=args.save_dir)
        elif args.env=="MsPacmanPreprocessed-v0":
            #MsPacmanPreprocessed-v0
            desired_rollouts = 4
            agent = TRPOAgent(env,logger,steps_per_rollout=5000,steps_between_rollouts=4, rollouts_per_sampling=desired_rollouts,multithreaded_rollout=True, single_batch_training = False, batch_size=250,DELTA=0.01, epsilon=0.3, epsilon_greedy=True, epsilon_decrease_factor=0.000, value_function_fitting_epochs=5, value_learning_rate=1e-2,backtrack_coeff=0.6, backtrack_iters=10, model_backup_dir=args.save_dir)
        elif args.env=="MsPacman-ram-v0":
            #MsPacman-ram-v0
            desired_rollouts = 30
            agent = TRPOAgent(env,logger,steps_per_rollout=5000,steps_between_rollouts=1, rollouts_per_sampling=desired_rollouts,multithreaded_rollout=True,single_batch_training = False, batch_size=5000,DELTA=0.01, epsilon=0.3, epsilon_greedy=True, epsilon_decrease_factor=0.0005, value_function_fitting_epochs=5, value_learning_rate=1e-2,backtrack_coeff=0.6, backtrack_iters=10, model_backup_dir=args.save_dir)
        else:
            desired_rollouts = 15
            agent = TRPOAgent(env,logger,steps_per_rollout=2000,steps_between_rollouts=1, rollouts_per_sampling=desired_rollouts,multithreaded_rollout=True, single_batch_training = True, batch_size=4000,DELTA=0.01, epsilon=0.4, epsilon_greedy=True, epsilon_decrease_factor=0.005, value_function_fitting_epochs=5, value_learning_rate=1e-2,backtrack_coeff=0.6, backtrack_iters=10, model_backup_dir=args.save_dir)
        
        if args.from_episode>0:
            agent.load_weights(args.from_episode)
        agent.learn(args.episodes,episodesBetweenModelBackups=args.backup_every,episodesBetweenAgentRenders=args.render,start_from_episode=0)
    elif args.command == "run":
        agent = TRPOAgent(env,logger)
        from_episode = args.episode
        to_episode = args.to_episode
        load_dir = args.load_dir
        if to_episode==-1 or to_episode<from_episode:
            agent.load_weights(args.episode, dir=load_dir)
            env.render_agent(agent)
        else:
            if args.skip<=0: skip=1
            else:
                skip = args.skip
            i = from_episode
            while i<=to_episode:
                agent.load_weights(i,dir=load_dir)
                print("Rendering episode: ",i)
                env.render_agent(agent)
                i+=skip

