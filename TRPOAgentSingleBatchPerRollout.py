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
        self.EPS = 0
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


    def training_step(self, episode, steps_per_rollout, rollouts_per_sampling, old_loss, rollout_until_success=False):
        
        def get_mean_kl_divergence(new_theta=None):
            if new_theta is None:
                model = self.policy
            else:
                model = self.new_model
                self.new_model.set_flat_params(new_theta)

            output = model(current_batch_observations)
            new_policy_action_probabilities = tf.nn.softmax(output).numpy()
            old_output = self.policy(current_batch_observations)
            old_policy_action_probabilities = tf.nn.softmax(old_output)
            kl = tf.stop_gradient(tf.reduce_mean(tf.reduce_sum(old_policy_action_probabilities * tf.math.log(old_policy_action_probabilities / new_policy_action_probabilities), axis=1)))
            return kl

        def fisher_vector_product(step_direction_vector):
            def kl_gradients():
                
                kl_divergence_gradients = self.policy.get_flat_gradients(get_mean_kl_divergence)
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

                temp_fisher_vectors_product = fvp(p)
                
                alpha = rdotr / (np.dot(p, temp_fisher_vectors_product) +self.EPS)
                if logger!=None:
                    self.log("CG Hessian: ",alpha,debug_channel="cg")
                #alpha = rdotr / (np.dot(p, temp_fisher_vectors_product))

                print(alpha.shape)
                print(p)
                print(p.shape)

                result += alpha * p
                r -= alpha * temp_fisher_vectors_product
                new_rdotr = np.dot(r, r)
                #beta = new_rdotr / rdotr + EPS
                beta = new_rdotr / (rdotr +self.EPS)
                
                p = r + beta * p
                rdotr = new_rdotr
                if rdotr < residual_tol:
                    break
            return result

        def surrogate_function(new_theta=None):
            if new_theta is None:
                model = self.policy
            else:
                model = self.new_model
                self.new_model.set_flat_params(new_theta)
            
            model = self.new_model
            #Calcola la nuova policy usando la rete neurale aggiornata ad ogni batch cycle del training step
            output = model(current_batch_observations)
            new_policy_action_probabilities = tf.nn.softmax(output)

            #Calcola la vecchia policy usando la vecchia rete neurale salvata
            output = self.policy(current_batch_observations)
            old_policy_action_probabilities = tf.nn.softmax(output)

            #pi(a|s) ovvero ci interessa solo l'azione che abbiamo compiuto, da cui il prodotto per actions_one_hot
            new_policy_action_probabilities = tf.reduce_sum(current_batch_actions_one_hot * new_policy_action_probabilities, axis=1)
            old_policy_action_probabilities = tf.reduce_sum(current_batch_actions_one_hot * old_policy_action_probabilities, axis=1)

            policy_ratio = new_policy_action_probabilities / old_policy_action_probabilities # (Schulman et al., 2017) Equation(14)

            #self.log("policy ratio: ", policy_ratio,", advantage: ", advantage, writeToFile=True, debug_channel="surrogate")
            loss = tf.reduce_mean(policy_ratio * current_batch_advantages)
            #self.log("loss value: ", loss, writeToFile=True, debug_channel="surrogate")

            return loss 

        def surrogate_function(new_theta=None):
            if new_theta is None:
                model = self.policy
            else:
                model = self.new_model
                self.new_model.set_flat_params(new_theta)
            
            model = self.new_model
            #Calcola la nuova policy usando la rete neurale aggiornata ad ogni batch cycle del training step
            output = model(current_batch_observations)
            new_policy_action_probabilities = tf.nn.softmax(output)

            #Calcola la vecchia policy usando la vecchia rete neurale salvata
            output = self.policy(current_batch_observations)
            old_policy_action_probabilities = tf.nn.softmax(output)

            #pi(a|s) ovvero ci interessa solo l'azione che abbiamo compiuto, da cui il prodotto per actions_one_hot
            new_policy_action_probabilities = tf.reduce_sum(current_batch_actions_one_hot * new_policy_action_probabilities, axis=1)
            old_policy_action_probabilities = tf.reduce_sum(current_batch_actions_one_hot * old_policy_action_probabilities, axis=1)

            policy_ratio = new_policy_action_probabilities / old_policy_action_probabilities # (Schulman et al., 2017) Equation(14)

            #self.log("policy ratio: ", policy_ratio,", advantage: ", advantage, writeToFile=True, debug_channel="surrogate")
            loss = tf.reduce_mean(policy_ratio * current_batch_advantages)
            #self.log("loss value: ", loss, writeToFile=True, debug_channel="surrogate")

            return loss 

            #1) Count steps since last rollout: if it is equal to the number of steps between two consecutive rollouts,
            #a) Collect new rollouts
            #b) theta_old is updated to the parameters of the current policy

        def policy_gradient():
            def logpi():            
                output = self.policy(current_batch_observations)
                pi = tf.nn.softmax(output)
                res = tf.reduce_sum(tf.math.log(tf.reduce_sum(current_batch_actions_one_hot * pi, axis=1)) * current_batch_advantages)
                print(res)
                return res

            print(logpi())
            logpi_grad = self.policy.get_flat_gradients(logpi)

            return logpi_grad

        def collect_rollout_statistics(steps_per_rollout, rollouts_per_sampling, multithreaded=False, only_successful=False, rollout_until_success=False):

            if rollout_until_success:
                actions, action_probabilities, observations, rewards = env.collect_rollouts_multithreaded_only_successful(self,rollouts_per_sampling,steps_per_rollout,THREADS_NUMBER)
            else: 
                actions, action_probabilities, observations, rewards = env.collect_rollouts_multithreaded(self,rollouts_per_sampling,steps_per_rollout,THREADS_NUMBER)

            def discount_rewards(rewards, GAMMA):

                discounted_rewards=[]
                for i, rollout_rewards in enumerate(rewards):
                    discounted_reward=0
                    discounted_rewards.append([])
                    for reward in reversed(rollout_rewards):
                        discounted_reward = reward + self.GAMMA*discounted_reward
                        discounted_rewards[i].insert(0, discounted_reward)
                            
                return discounted_rewards

            discounted_rewards = discount_rewards(rewards,self.GAMMA)

            assert(len(actions)==len(action_probabilities)==len(rewards)==len(discounted_rewards)==len(observations))

            #print(len(actions))

            total_elements = 0
            for i in range(len(actions)):
                total_elements += len(actions[i])

            return actions, action_probabilities, observations, rewards, discounted_rewards, total_elements

        if self.steps_since_last_rollout == self.steps_between_rollouts:
            self.log("Performing rollouts: rollout length: ",self.steps_per_rollout,writeToFile=True,debug_channel="learning")
            self.steps_since_last_rollout = 0

            self.actions, self.action_probabilities, self.observations, self.rewards, self.discounted_rewards, self.total_rollout_elements = collect_rollout_statistics(steps_per_rollout, rollouts_per_sampling, multithreaded=self.multithreaded_rollout,rollout_until_success=rollout_until_success)
            

            self.log("Rollouts performed",writeToFile=True,debug_channel="learning")

            #Change theta_old old policy params once every steps_between_rollouts rollouts
            theta = self.policy.get_flat_params()
            self.theta_old = theta
        
        #2) Compute the number of batches
        #number_of_batches = self.rollouts_per_sampling
        if self.single_batch_training: 
            number_of_batches = 1
            actions = np.concatenate(self.actions)
            action_probabilities = np.concatenate(self.action_probabilities)
            observations = np.concatenate(self.observations)
            rewards = np.concatenate(self.rewards)
            discounted_rewards = np.concatenate(self.discounted_rewards)
        else:
            #number_of_batches = math.floor(self.total_rollout_elements / self.batch_size)
            number_of_batches = self.rollouts_per_sampling

        self.log("Rollout statistics size: ", self.total_rollout_elements, ", Batch size: ", self.batch_size,", Number of batches: ",number_of_batches,writeToFile=True,debug_channel="batch_info")
        self.log("\n\n***************\nBEGINNING TRAINING\n***************\n\n",writeToFile=True,debug_channel="learning")

        #4) For each batch
        mean_kl_divs = []
        value_losses = []
        policy_losses = []

        max_reward=-math.inf
        mean_reward=-math.inf
        for batch in range(number_of_batches):

            policy_loss = 0
            value_loss = 0

            #4.1) Data for current batch 
            
            if self.single_batch_training:
                current_batch_actions = actions
                current_batch_observations = observations
                current_batch_discounted_rewards = discounted_rewards
                current_batch_rewards = rewards
            else:
                current_batch_actions = self.actions[batch]
                current_batch_observations = self.observations[batch]
                current_batch_discounted_rewards = self.discounted_rewards[batch]
                current_batch_rewards = self.rewards[batch]

            current_batch_mean_reward = sum(current_batch_rewards)/self.rollouts_per_sampling
            current_batch_total_reward = sum(current_batch_rewards)

            self.log("Batch #", batch, ", batch length: ", len(current_batch_actions), writeToFile=True, debug_channel="batch_info")

            current_batch_predicted_state_values = self.state_value(current_batch_observations)

            #current_batch_advantages = compute_advantages_vanilla(current_batch_discounted_rewards, current_batch_observations, self.state_value)
            #current_batch_advantages = np.array(GAE(self.state_value, current_batch_observations, current_batch_rewards, self.GAMMA, self.LAMBDA))
            current_batch_advantages = current_batch_discounted_rewards - current_batch_predicted_state_values
            
            #Normalize advantages
            current_batch_advantages = (current_batch_advantages - current_batch_advantages.mean())/(current_batch_advantages.std() +self.EPS)
            

            #4.3) Current batch actions one hot
            current_batch_actions_one_hot = tf.one_hot(current_batch_actions, self.env.get_action_shape(), dtype="float64")


            #g vector
            #policy_gradient = self.policy.get_flat_gradients(surrogate_function).numpy()
            #print(policy_gradient)
            #self.log("policy_gradient: ", policy_gradient, writeToFile=True, debug_channel="training")


            #4.7) Use conjugate gradients to efficiently compute the gradient direction
            #self.log("Initiating conjugate gradients", writeToFile=True, debug_channel="training")
            #H^-1 * g
            #gradient_step_direction = conjugate_gradients(policy_gradient)
            #self.log("gradient_step_direction: ", gradient_step_direction, writeToFile=True, debug_channel="training")

            #4.8) Fisher vectors product to solve the constrained optimization problem
            #fvp = fisher_vectors_product(gradient_step_direction)
            #self.log("fisher_vectors_product: ", fvp, writeToFile=True, debug_channel="training")

            #4.9) Compute the maximum gradient step
            #From spinningup OpenAi implementation

            #max_gradient_step = (np.sqrt(2*self.DELTA/np.dot(gradient_step_direction,fisher_vectors_product(gradient_step_direction)))) * gradient_step_direction
            #self.log("max_gradient_step: ", max_gradient_step, debug_channel="training")

            #g = self.policy.get_flat_gradients(surrogate_function).numpy()
            g = policy_gradient().numpy()
            print(g)
            stepdir = conjugate_gradients(fisher_vector_product, g)
            shs = .5 * stepdir.dot(fisher_vector_product(stepdir))
            lm = np.sqrt(shs / self.DELTA)
            fullstep = stepdir / lm
            #neggdotstepdir = -g.dot(stepdir)
            thprev = self.policy.get_flat_params()
            #loss = surrogate_function            
            
            if np.isnan(fullstep).any():
                continue
            def linesearch():
                theta = self.policy.get_flat_params()                
                #old_loss_value = surrogate_function().numpy()
                #self.log("loss_value before: ", old_loss_value)

                for (_n_backtracks, step) in enumerate(self.backtrack_coeff**np.arange(self.backtrack_iters)):

                    #From spinningup OpenAi implementation
                    #theta_new = theta - step * max_gradient_step
                    theta_new = theta + step * fullstep
                    
                    #NaN protection
                    #if np.isnan(theta_new).any():
                    #    self.log("NaN detected in new parameters. Linesearch failed. Not updating parameters to avoid NaN catastrophe!!!",debug_channel="linesearch")
                    #    return False, theta, 0.0, old_loss_value
                    #else:
                    
                    new_loss_value = surrogate_function(theta_new).numpy()
                                                            
                    mean_kl_div = get_mean_kl_divergence(theta_new).numpy()

                    #improvement = (old_loss_value - new_loss_value)

                    #self.log("improvement: ", improvement, writeToFile=True, debug_channel="linesearch")
                    #self.log("New policy loss: ", new_loss_value, debug_channel="linesearch")
                    #if mean_kl_div <= self.DELTA and improvement >= 0:
                    #https://medium.com/@jonathan_hui/rl-trust-region-policy-optimization-trpo-part-2-f51e3b2e373a verso la fine
                    #if mean_kl_div <= self.DELTA:
                    if mean_kl_div <= self.DELTA and new_loss_value>old_loss:
                    #if mean_kl_div <= self.DELTA and new_loss_value>=0 and (old_loss_value<0 or (old_loss_value>=0 and new_loss_value <= old_loss_value)):
                    #if mean_kl_div <= self.DELTA and ((old_loss_value<0 and new_loss_value>=old_loss_value) or (old_loss_value>0 and new_loss_value <= old_loss_value)):
                        self.log("Linesearch worked at ", _n_backtracks, ", New mean kl div: ", mean_kl_div, ", New policy loss value: ", new_loss_value, writeToFile=True, debug_channel="linesearch")
                        return True, theta_new, new_loss_value
                    if _n_backtracks == self.backtrack_iters - 1:
                        self.log("Linesearch failed. Mean kl divergence: ", mean_kl_div, ", Discarded policy loss value: ", new_loss_value, writeToFile=True, debug_channel="linesearch")
                        return False, theta, old_loss
                    

            def linesearch2(f, x, fullstep, expected_improve_rate):
                accept_ratio = .1
                max_backtracks = 10
                fval = f(x)
                for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
                    xnew = x + stepfrac * fullstep
                    newfval = f(xnew)
                    actual_improve = fval - newfval
                    expected_improve = expected_improve_rate * stepfrac
                    ratio = actual_improve / expected_improve
                    if ratio > accept_ratio and actual_improve > 0:
                        return True, xnew, newfval
                return False, x, fval                    

            #linesearch_success, new_theta, mean_kl_div, policy_loss = linesearch()
            #linesearch_success, theta, policy_loss = linesearch2(loss, thprev, fullstep, neggdotstepdir / lm)
            linesearch_success, theta, policy_loss = linesearch()
       
            history = self.state_value.fit(current_batch_observations, current_batch_discounted_rewards, epochs=self.value_function_fitting_epochs, verbose=0)
            value_loss = history.history["loss"][-1]

            self.log("Current batch value loss: ",value_loss,writeToFile=True,debug_channel="batch_info")

            self.policy.set_flat_params(theta)
            if get_mean_kl_divergence(theta) > 2.0 * self.DELTA:
                self.policy.set_flat_params(thprev)
            #if linesearch_success:
                self.log("Linesearch successful, updating policy parameters", writeToFile=True, debug_channel="linesearch")

                #NaN protection
            #    if np.isnan(theta).any():
            #        self.log("NaN detected in new parameters. Not updating parameters to avoid NaN catastrophe!!!",debug_channel="learning")
            #    else: 
            #        self.policy.set_flat_params(new_theta)

            #self.log("END OF TRAINING BATCH #", batch, debug_channel="batch_info")
            #self.log("BATCH STATS: Reward: ",current_batch_reward,", Mean KL: ",mean_kl_div," Policy loss: ", policy_loss, ", Value loss: ", state_val, ", linesearch_success: ", linesearch_success, ", epsilon: ", self.epsilon, debug_channel="batch_info")
        
            if mean_reward==-math.inf:
                mean_reward = current_batch_mean_reward
            else:
                mean_reward = (current_batch_mean_reward + mean_reward)/2

            if current_batch_total_reward > max_reward:
                max_reward = current_batch_total_reward


        state_val = tf.reduce_mean(self.state_value(current_batch_observations)).numpy().flatten()

        self.log("END OF TRAINING STEP #", episode)
        self.log("TRAINING STEP STATS: Max reward: ",max_reward," Mean reward: ", mean_reward, ", Policy loss: ", policy_loss, ", Mean state val: ", state_val, debug_channel="batch_info")

        self.steps_since_last_rollout += 1
        self.epsilon = self.epsilon - self.epsilon_decrease_factor

        return current_batch_total_reward, current_batch_mean_reward, value_loss, policy_loss

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
        #action = np.random.choice(range(action_probabilities.shape[0]), p=action_probabilities)
        
        action = np.argmax(policy_output)

        exploitation_chance = np.random.uniform(0,1)
        last_action_repeat_chance = 0.8
        if self.epsilon_greedy and exploitation_chance < self.epsilon:
        #    if self.epsilon_greedy and np.random.uniform(0,1) < last_action_repeat_chance and last_action!=None:
        #        action = last_action
        #    else:
            #perform a random action
            action = np.random.randint(0,self.env.get_action_shape())
        
        #self.log("chosen_action: ",action, writeToFile=True, debug_channel="act")

        return action, action_probabilities

    def learn(self, nEpisodes, episodesBetweenModelBackups=-1, start_from_episode=0):
        initial_time = time.time()
        history_values = {"mean_reward":[],"value_loss":[],"policy_loss":[]}
        loss_values = []

        plot, subplots = plt.subplots(3, 1, constrained_layout=True)

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

            total_reward, mean_reward, value_loss, policy_loss = self.training_step(episode,current_episode_steps_per_rollout,current_rollouts_per_sampling, old_loss, rollout_until_success=rollout_until_success)            

            #history_entry = {"max_reward":max_reward,"value_loss":value_loss,"policy_loss":policy_loss}

            history_values["mean_reward"].append(mean_reward)
            history_values["value_loss"].append(value_loss)
            history_values["policy_loss"].append(policy_loss)

            self.log_history(history_values)
            self.plot_history(plot, subplots, history_values)

            old_loss = policy_loss

            if self.render: 
                env.render_agent(agent, nSteps = self.steps_per_rollout)
            
            import os
            print(os.getcwd())
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
    
    env_name = "MountainCar-v0"
    
    channels = [
        #"rollouts",
        #"advantages",
        #"rollouts_dump",
        "act",
        #"training",
        "batch_info",
        "linesearch",
        "learning",
        #"thread_rollouts",
        #"model",
        #"utils",
        #"utils_kl",
        #"cg",
        #"GAE",
        #"surrogate",
        #"EnvironmentRegister",
        #"environment"
        ]

    logger = Logger(name=env_name,log_directory="TRPO_project/Models/"+env_name,debugChannels=channels)

    #tf.set_random_seed(0)
    np.random.seed(0)
    env = Environment(env_name,logger,use_custom_env_register=True, debug=True, show_preprocessed=False, same_seed=True)

    #CartPole-v1
    #agent = TRPOAgent(env,logger,steps_per_rollout=1500,steps_between_rollouts=1, rollouts_per_sampling=10,multithreaded_rollout=True, full_rollout_episodes=0,single_batch_training = True, batch_size=100,DELTA=0.01, epsilon=0.15, epsilon_greedy=True,value_function_fitting_epochs=5, value_learning_rate=1e-1,backtrack_coeff=1e-4, backtrack_iters=10,render=True)
    agent = TRPOAgent(env,logger,steps_per_rollout=5000,steps_between_rollouts=1, rollouts_per_sampling=10,multithreaded_rollout=True, full_rollout_episodes=0,single_batch_training = True, batch_size=1000,DELTA=0.01, epsilon=0.3, epsilon_greedy=True,value_function_fitting_epochs=5, value_learning_rate=1e-2,backtrack_coeff=1e-1, backtrack_iters=10,render=True)
    #agent = TRPOAgent(env,logger,steps_per_rollout=1500,steps_between_rollouts=1, rollouts_per_sampling=15, multithreaded_rollout=True, batch_size=4500, DELTA=0.01)
      
    #initial_time = time.time()
    #env.collect_rollouts(agent,10,1000)
    #first_time = time.time()
    #env.collect_rollouts_multithreaded(agent,10,1000,15)
    #second_time = time.time()

    #print("Non-multithreaded: ",first_time-initial_time,", multithreaded: ",second_time-first_time)
    #agent.load_weights(0)

    #agent.training_step(0)
    history = agent.learn(500,episodesBetweenModelBackups=1,start_from_episode=0)
    #plt.plot(history)
    #plt.show()
        
    
    #env.render_agent(agent)

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