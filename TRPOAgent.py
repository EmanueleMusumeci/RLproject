import os
import glob
import gym
import numpy as np
import time
import math
from collections import namedtuple
import errno

import tensorflow as tf
from tensorflow import GradientTape
from tensorflow import keras
#sets to float64 to avoid compatibility issues between numpy 
# (whose default is float64) and keras(whose default is float32)
keras.backend.set_floatx("float64")

from matplotlib import pyplot as plt 

from utils import *
from Environment import Environment, RolloutTuple
from Models import Policy, Value
from Logger import Logger

#RolloutStatistics = namedtuple("RolloutStatistics","actions action_probabilities rewards discounted_rewards mean_discounted_rewards observations advantages size")
RolloutStatistics = namedtuple("RolloutStatistics","actions action_probabilities rewards discounted_rewards observations size")
TrainingInfo = namedtuple("TrainingInfo","mean_value_loss linesearch_successful mean_kl_divergence")

THREADS_NUMBER=8

class TRPOAgent:
    def __init__(self, 
    env,
    logger,
    
    #Rollouts
    steps_per_rollout=0, 
    steps_between_rollouts=0, 
    rollouts_per_sampling=10, 
    multithreaded_rollout=False,

    #Training
    batch_size = 4096,

    #Coefficients
    #conjugate_gradients_damping=0.001, 
    conjugate_gradients_damping = 0.001,
    conjugate_gradients_iterations = 20,
    #DELTA=0.01, 
    DELTA=0.01,
    #backtrack_coeff=.6,
    backtrack_coeff=3e-1, 
    #backtrack_iters=10, 
    backtrack_iters=5,
    #gamma = 0.99,
    gamma= 0.995,
    value_learning_rate=1e-1,
    value_function_fitting_epochs=10,
    epsilon_greedy=True,
    epsilon=0.4,
    epsilon_decrease_factor=0.005,

    #Debug
    debug_rollouts=False, 
    debug_act=False, 
    debug_training=False, 
    debug_model=False, 
    debug_learning=False,

    #Others
    model_backup_dir="TRPO_project/Models"):

        assert(env!=None), "You need to create an environment first"
        self.env = env
        self.model_backup_dir = model_backup_dir+"/"+self.env.environment_name

        #assert(logger!=None), "You need to create a logger first"
        self.logger = logger

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
        self.gamma = gamma

        #Conjugate gradients damping
        self.conjugate_gradients_damping = conjugate_gradients_damping
        self.conjugate_gradients_iterations = conjugate_gradients_iterations

        #Line search
        self.backtrack_coeff = backtrack_coeff
        self.backtrack_iters = backtrack_iters
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

        self.rollout_statistics = []

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

    def collect_rollout_statistics(self, multithreaded=False):
        print(multithreaded)
        if multithreaded: 
            rollouts = env.collect_rollouts_multithreaded(self,self.rollouts_per_sampling,self.steps_per_rollout,THREADS_NUMBER)
        else: 
            rollouts, _ = env.collect_rollouts(self,self.rollouts_per_sampling,self.steps_per_rollout)

        def collect_actions(rollouts):
            actions = []
            action_probabilities = []
            #entropies = []
            #rollouts is a list of lists (each one of these lists is a single rollout)
            for rollout in rollouts:
                #rollout is a list of RolloutTuple namedtuples
                entropy=0
                for tupl in rollout:
                    actions.append(tupl.action)
                    action_probabilities.append(tupl.action_probabilities)
                    #entropy += -tf.reduce_sum(action_probabilities*tf.math.log(action_probabilities))
            return actions, action_probabilities

        def collect_rewards(rollouts, gamma):
            rewards = []
            result = []
            discounted_rewards = []
            for i in range(len(rollouts)):
                rewards.append([])
                for tupl in rollouts[i]:
                    rewards[i].append(tupl.reward)
                    result.append(tupl.reward)
                discounted_reward=0
                print(len(rewards[i]))
                for reward in rewards[i][::-1]:
                    discounted_reward = reward + self.gamma*discounted_reward
                    discounted_rewards.insert(0, discounted_reward)
                    #contain the total discounted reward (Q-value) for each class
                    #discounted_rewards.insert(0,discounted_reward)
            
            print(len(discounted_rewards))
            print(len(rewards))
            print(len(result))

            return result, discounted_rewards

        def collect_observations(rollouts):
            observations = []
            for rollout in rollouts:
                for tupl in rollout:
                    observations.append(tupl.observation)
            return observations

        self.log("Unpacking actions",writeToFile=True,debug_channel="rollouts")
        actions, action_probabilities = collect_actions(rollouts)
        self.log("len(actions): ",len(actions),writeToFile=True,debug_channel="rollouts_dump")
        self.log("Unpacking rewards",writeToFile=True,debug_channel="rollouts")
        rewards, discounted_rewards = collect_rewards(rollouts, self.gamma)
        self.log("len(rewards): ",len(rewards),writeToFile=True,debug_channel="rollouts_dump")
        self.log("Unpacking observations",writeToFile=True,debug_channel="rollouts")
        observations = collect_observations(rollouts)
        self.log("len(observations): ",len(observations),writeToFile=True,debug_channel="rollouts_dump")
        #self.log("Computing advantages",writeToFile=True,debug_channel="rollouts")
        #advantages = compute_advantages(discounted_rewards, observations, self.state_value)
        #assert(len(actions)==len(action_probabilities)==len(rewards)==len(discounted_rewards)==len(observations)==len(advantages))
        assert(len(actions)==len(action_probabilities)==len(rewards)==len(discounted_rewards)==len(observations))
        size = len(actions)

        statistics = RolloutStatistics(
            actions,
            action_probabilities,
            rewards,
            discounted_rewards,
            observations,
            #advantages,
            size
            )

        return statistics

    def surrogate_function(self, args, logger=None):
        current_batch_observations = args["observation"]
        actions_one_hot = args["actions_one_hot"]
        advantage = args["advantage"]
        
        #Calcola la nuova policy usando la rete neurale aggiornata ad ogni batch cycle del training step
        new_policy_action_probabilities = self.new_model(current_batch_observations, tensor=True)
        new_policy_action_probabilities = tf.nn.softmax(new_policy_action_probabilities)
        #Calcola la vecchia policy usando la vecchia rete neurale salvata
        old_policy_action_probabilities = self.policy(current_batch_observations, tensor=True)
        old_policy_action_probabilities = tf.nn.softmax(old_policy_action_probabilities)
        
        self.log("actions_one_hot: ",actions_one_hot, writeToFile=True, debug_channel="training_dump")
        self.log("new_policy_action_probabilities: ",new_policy_action_probabilities, writeToFile=True, debug_channel="training_dump")
        self.log("old_policy_action_probabilities: ",old_policy_action_probabilities, writeToFile=True, debug_channel="training_dump")

        #pi(a|s) ovvero ci interessa solo l'azione che abbiamo compiuto, da cui il prodotto per actions_one_hot
        new_policy_action_probabilities = tf.reduce_sum(actions_one_hot * new_policy_action_probabilities, axis=1)
        old_policy_action_probabilities = tf.reduce_sum(actions_one_hot * old_policy_action_probabilities, axis=1)

        self.log("reduced new_policy_action_probabilities: ",new_policy_action_probabilities, writeToFile=True, debug_channel="training_dump")
        self.log("reduced old_policy_action_probabilities: ",old_policy_action_probabilities, writeToFile=True, debug_channel="training_dump")


        policy_ratio = new_policy_action_probabilities / old_policy_action_probabilities # (Schulman et al., 2017) Equation(14)
        #logger.print("policy_ratio: ", policy_ratio)
        #Calcola la loss function: ha anche aggiunto un termine di entropia
        self.log("policy ratio: ", policy_ratio,", advantage: ", advantage, writeToFile=True, debug_channel="surrogate")
        loss = tf.reduce_mean(policy_ratio * advantage)
        self.log("loss value: ", loss, writeToFile=True, debug_channel="surrogate")
        #return -loss
        return loss 

    def fisher_vectors_product(self, model, step_direction_vector, damping_factor, args, logger=None):
        #this method is supposed to compute the Fishers's vector product as the Hessian of the Kullback-Leibler divergence

        def get_mean_kl_divergence(args, logger=None):
            current_batch_observations = args["observation"]
            #current_batch_actions_one_hot = args["actions_one_hot"]            

            #4.5) Compute new policy using latest policy (update with latest batch) and old policy (the one that dates back to the latest rollout) 
            #Calcola la nuova policy usando la rete neurale aggiornata ad ogni batch cycle del training step
            new_policy_action_probabilities = tf.nn.softmax(self.new_model(current_batch_observations, tensor=True))
            
            #Calcola la vecchia policy usando la vecchia rete neurale salvata
            old_policy_action_probabilities = tf.nn.softmax(self.policy(current_batch_observations, tensor=True))

            kl = tf.reduce_mean(tf.reduce_sum(old_policy_action_probabilities * tf.math.log(old_policy_action_probabilities / new_policy_action_probabilities), axis=1))
            if math.isinf(kl):
                if logger!=None:
                    log("Inf divergence: old_policy_probabilities: ",old_policy_action_probabilities,", new_policy_probabilities: ", new_policy_action_probabilities,writeToFile=True,debug_channel="utils_kl_dump", skip_stack_levels=2, logger=logger)
            elif math.isnan(kl):
                if logger!=None:
                    log("nan divergence: old_policy_probabilities: ",old_policy_action_probabilities,", new_policy_probabilities: ", new_policy_action_probabilities,writeToFile=True,debug_channel="utils_kl_dump", skip_stack_levels=2, logger=logger)
            else:
                if logger!=None:
                    log("Divergence: ",kl,", old_policy_probabilities: ",old_policy_action_probabilities,", new_policy_probabilities: ", new_policy_action_probabilities,writeToFile=True,debug_channel="utils_kl_dump", skip_stack_levels=2, logger=logger)
            return kl


        def kl_value(logger=None):
            kl_divergence_gradients = model.get_flat_gradients(get_mean_kl_divergence, args)
            if logger!=None:
                log("KL divergence gradients: ",kl_divergence_gradients,writeToFile=True,debug_channel="utils_kl", skip_stack_levels=2, logger=logger)

            kl = tf.reduce_sum(kl_divergence_gradients * step_direction_vector)
            return kl

        kl_flat_grad_grads = model.get_flat_gradients(kl_value)

        if logger!=None:
            log("KL divergence Hessian: ",kl_flat_grad_grads,writeToFile=True,debug_channel="utils_kl", skip_stack_levels=2, logger=logger)

        return kl_flat_grad_grads + step_direction_vector * damping_factor
    
    #    #TODO: Testare se funziona correttamente
    def conjugate_gradients(self, model, policy_gradient, damping_factor, cg_iters, args, residual_tol=1e-10, logger=None):
        result = np.zeros_like(policy_gradient)
        r = tf.identity(policy_gradient)
        p = tf.identity(policy_gradient)

        rdotr = np.dot(r, r)

        for i in range(cg_iters):
            if logger!=None:
                log("Conjugate gradients iteration: ",i, writeToFile=True,debug_channel="cg", skip_stack_levels=3, logger=logger)

            #Verificare se il risultato di fisher_vectors_product è A o A * p
            temp_fisher_vectors_product = self.fisher_vectors_product(model, p, damping_factor, args, logger)
            
            alpha = rdotr / (np.dot(p, temp_fisher_vectors_product) + EPS)
            self.log("CG alpha: ", alpha,debug_channel="cg")
            #alpha = rdotr / (np.dot(p, temp_fisher_vectors_product))

            result += alpha * p
            r -= alpha * temp_fisher_vectors_product
            new_rdotr = np.dot(r, r)
            #beta = new_rdotr / rdotr + EPS
            beta = new_rdotr / rdotr
            
            p = r + beta * p
            rdotr = new_rdotr
            if rdotr < residual_tol:
                break
        return result

    def training_step(self, episode):

        #1) Count steps since last rollout: if it is equal to the number of steps between two consecutive rollouts,
            #a) Collect new rollouts
            #b) theta_old is updated to the parameters of the current policy
        if self.steps_since_last_rollout == self.steps_between_rollouts:
            self.log("Performing rollouts: rollout length: ",self.steps_per_rollout,writeToFile=True,debug_channel="learning")
            self.steps_since_last_rollout = 0
            self.rollout_statistics = self.collect_rollout_statistics(multithreaded=self.multithreaded_rollout)
            self.log("Rollouts performed",writeToFile=True,debug_channel="learning")

            #Change theta_old old policy params once every steps_between_rollouts rollouts
            theta = self.policy.get_flat_params()
            self.theta_old = theta
        
        #2) Compute the number of batches
        number_of_batches = math.ceil(self.rollout_statistics.size / self.batch_size)
        self.log("Rollout statistics size: ", self.rollout_statistics.size, ", Batch size: ", self.batch_size,", Number of batches: ",number_of_batches,writeToFile=True,debug_channel="batch_info")

        self.log("\n\n***************\nBEGINNING TRAINING\n***************\n\n",writeToFile=True,debug_channel="learning")

        #4) For each batch
        mean_kl_divs = []
        value_losses = []
        policy_losses = []
        rewards = []
        for batch in range(number_of_batches):

            policy_loss = 0
            value_loss = 0
            #4.1) Data for current batch 
            current_batch_actions = np.array(self.rollout_statistics.actions[batch*self.batch_size:(batch+1)*self.batch_size])
            current_batch_observations = np.array(self.rollout_statistics.observations[batch*self.batch_size:(batch+1)*self.batch_size])
            #current_Q_values = Q_values[batch]
            current_batch_discounted_rewards = np.array(self.rollout_statistics.discounted_rewards[batch*self.batch_size:(batch+1)*self.batch_size])

            current_batch_reward = tf.reduce_sum(self.rollout_statistics.rewards[batch*self.batch_size:(batch+1)*self.batch_size]).numpy()

            self.log("Batch #", batch, ", batch length: ", len(current_batch_actions), writeToFile=True, debug_channel="batch_info")

            #4.2) Compute advantages with latest value network
            def compute_advantages(current_batch_discounted_rewards, current_batch_observations, value_model):
                self.log("observations: ",current_batch_observations,", len: ", len(current_batch_observations), writeToFile=True, debug_channel="rollouts_dump")
                self.log("current_batch_discounted_rewards: ",current_batch_discounted_rewards,", len: ", len(current_batch_discounted_rewards), writeToFile=True, debug_channel="rollouts_dump")
                assert(len(current_batch_discounted_rewards)==len(current_batch_observations))
                advantages = []
                for i in range(len(current_batch_discounted_rewards)):
                    self.log("Computing advantage #",i," for batch #",batch, writeToFile=True, debug_channel="advantages")
                    q_value = current_batch_discounted_rewards[i]
                    value_prediction = value_model(np.array([current_batch_observations[i]])).numpy()

                    #Predict state value (NOTICE: we use flatten() in case value prediction is a tuple)
                    advantage_value = (q_value - value_prediction).flatten()[0]
                    advantages.append(advantage_value)

                self.log("Advantages for batch #",batch,": ",advantages, writeToFile=True, debug_channel="advantages")

                return np.array(advantages)

            current_batch_advantages = compute_advantages(current_batch_discounted_rewards, current_batch_observations, self.state_value)
            #Normalize advantages
            current_batch_advantages = (current_batch_advantages - current_batch_advantages.mean())/(current_batch_advantages.std() + 1e-8)

            #4.3) Current batch actions one hot
            current_batch_actions_one_hot = tf.one_hot(current_batch_actions, self.env.get_action_shape(), dtype="float64")

            #4.4) Compute current batch policy gradient
            #args to pass to the loss function in order to get the gradient
            args = {
                "observation" : current_batch_observations,
                "actions_one_hot" : current_batch_actions_one_hot,
                "advantage" : current_batch_advantages
            }

            #g vector
            policy_gradient = self.policy.get_flat_gradients(self.surrogate_function, args)
            self.log("policy_gradient: ", policy_gradient, writeToFile=True, debug_channel="training")

            #4.5) Compute new policy using latest policy (update with latest batch) and old policy (the one that dates back to the latest rollout)
            new_policy_action_probabilities = tf.nn.softmax(self.new_model(current_batch_observations, tensor=True))
           
            #Calcola la vecchia policy usando la vecchia rete neurale salvata
            old_policy_action_probabilities = tf.nn.softmax(self.policy(current_batch_observations, tensor=True))

            #4.6) Mean Kullback-Leibler divergence
            mean_kl_div = mean_kl_divergence(old_policy_action_probabilities, new_policy_action_probabilities)
            self.log("mean_kl_divergence: ", mean_kl_div, writeToFile=True, debug_channel="training")
            
            self.log("Initiating conjugate gradients", writeToFile=True, debug_channel="training")
            #4.7) Use conjugate gradients to efficiently compute the gradient direction

            #H^-1 * g
            gradient_step_direction = self.conjugate_gradients(self.policy, policy_gradient, self.conjugate_gradients_damping, self.conjugate_gradients_iterations, args, logger=self.logger)
            self.log("gradient_step_direction: ", gradient_step_direction, writeToFile=True, debug_channel="training")

            #4.8) Fisher vectors product to solve the constrained optimization problem
            fvp = self.fisher_vectors_product(self.policy,gradient_step_direction,self.conjugate_gradients_damping, args, logger=self.logger).numpy()
            self.log("fisher_vectors_product: ", fvp, writeToFile=True, debug_channel="training")

            #4.9) Compute the maximum gradient step
            #From spinningup OpenAi implementation

            max_gradient_step = (np.sqrt(2*self.DELTA/np.dot(gradient_step_direction, fvp)+EPS)) * gradient_step_direction
            #max_gradient_step = (np.sqrt(2*self.DELTA/np.dot(gradient_step_direction,fvp.T))) * gradient_step_direction
            if np.isnan(max_gradient_step).any():
            #    self.log("NaN detected in max_gradient_step. Skippping training step to avoid NaN catastrophe!!!",debug_channel="learning")
            #    policy_loss=0.0
            #    value_loss=0.0
                continue
            self.log("max_gradient_step: ", max_gradient_step, debug_channel="training")

            
            #4.10) Perform line search to find the optimal gradient step
            def linesearch():
                theta = self.policy.get_flat_params()
                old_loss_value = self.surrogate_function(args).numpy()
                self.log("loss_value before", old_loss_value)

                for (_n_backtracks, step) in enumerate(self.backtrack_coeff**np.arange(self.backtrack_iters)):

                    #From spinningup OpenAi implementation
                    #theta_new = theta - step * max_gradient_step
                    theta_new = theta + step * max_gradient_step
                    
                    #NaN protection
                    if np.isnan(theta_new).any():
                        self.log("NaN detected in new parameters. Linesearch failed. Not updating parameters to avoid NaN catastrophe!!!",debug_channel="linesearch")
                        return False, theta, 0.0, old_loss_value
                    else:
                        self.new_model.set_flat_params(theta_new)

                    new_loss_value = self.surrogate_function(args).numpy()

                    new_policy_action_probabilities = tf.nn.softmax(self.new_model(current_batch_observations, tensor=True))
                    #self.log("new_policy_action_probabilities: ",new_policy_action_probabilities, writeToFile=True, debug_channel="linesearch")
                    
                    mean_kl_div = mean_kl_divergence(old_policy_action_probabilities,new_policy_action_probabilities).numpy()

                    improvement = (old_loss_value - new_loss_value)

                    self.log("improvement: ", improvement, writeToFile=True, debug_channel="linesearch")
                    
                    #if mean_kl_div <= self.DELTA and improvement >= 0:
                    #https://medium.com/@jonathan_hui/rl-trust-region-policy-optimization-trpo-part-2-f51e3b2e373a verso la fine
                    if mean_kl_div <= self.DELTA and new_loss_value >=0:
                    #if mean_kl_div <= self.DELTA and new_loss_value>=0 and (old_loss_value<0 or (old_loss_value>=0 and new_loss_value <= old_loss_value)):
                        self.log("Linesearch worked at ", _n_backtracks, ", New policy loss value: ", new_loss_value, writeToFile=True, debug_channel="linesearch")
                        return True, theta_new, mean_kl_div, new_loss_value
                    if _n_backtracks == self.backtrack_iters - 1:
                        self.log("Linesearch failed. Mean kl divergence: ", mean_kl_div, ", Discarded policy loss value: ", new_loss_value, writeToFile=True, debug_channel="linesearch")
                        return False, theta, mean_kl_div, old_loss_value

            linesearch_success, new_theta, mean_kl_div, policy_loss = linesearch()
       
            history = self.state_value.fit(current_batch_observations, current_batch_discounted_rewards, epochs=self.value_function_fitting_epochs, verbose=0)
            value_loss = history.history["loss"][-1]
            self.log("Current batch value loss: ",value_loss,writeToFile=True,debug_channel="batch_info")

            if linesearch_success:
                self.log("Linesearch successful, updating policy parameters", writeToFile=True, debug_channel="linesearch")

                #NaN protection
                if np.isnan(new_theta).any():
                    self.log("NaN detected in new parameters. Not updating parameters to avoid NaN catastrophe!!!",debug_channel="learning")
                else: 
                    self.policy.set_flat_params(new_theta)

            self.log("END OF TRAINING BATCH #", batch, debug_channel="batch_info")
            self.log("BATCH STATS: Reward: ",current_batch_reward,", Mean KL: ",mean_kl_div," Policy loss: ", policy_loss, ", Value loss: ", value_loss, ", linesearch_success: ", linesearch_success, ", epsilon: ", self.epsilon, debug_channel="batch_info")
        
        #mean_reward = tf.reduce_mean(rewards).numpy()
        if len(rewards)>0: 
            max_reward = max(rewards)
            mean_kl_div = tf.reduce_mean(mean_kl_divs).numpy()
            #mean_value_loss = tf.reduce_mean(value_losses).numpy()
            #mean_policy_loss = tf.reduce_mean(policy_losses).numpy()
        else:
            max_reward=-self.batch_size
            mean_kl_div = 0.0
            #mean_value_loss = 0.0
            #mean_policy_loss = 0.0
        
        self.log("END OF TRAINING STEP #", episode)
        self.log("TRAINING STEP STATS: Max reward: ",max_reward,", Mean KL: ",mean_kl_div," Policy loss: ", policy_loss, ", Value loss: ", value_loss, debug_channel="batch_info")

        self.steps_since_last_rollout += 1
        self.epsilon = self.epsilon - self.epsilon_decrease_factor

        return max_reward, mean_kl_div, value_loss, policy_loss

    #the ACT function: produces an action from the current policy
    def act(self,observation):

        #Trasforma la observation in un vettore riga [[array]]
        observation = observation[np.newaxis, :]
        #self.log("observation: ",observation,", len: ",len(observation), writeToFile=True, debug_channel="act")

        #Estrai la prossima azione dalla NN della policy
        policy_output = self.policy(observation)
        action_probabilities = tf.nn.softmax(policy_output).numpy().flatten()
        #self.log("action_probabilities: ",action_probabilities,", len: ",len(action_probabilities), writeToFile=True, debug_channel="act")

        action = np.random.choice(range(action_probabilities.shape[0]), p=action_probabilities)
        #self.log("chosen_action: ",action, writeToFile=True, debug_channel="act")

        exploitation_chance = np.random.uniform(0,1)
        if self.epsilon_greedy and exploitation_chance < self.epsilon:
            if self.epsilon_greedy and np.random.uniform(0,1) < 0.8 and self.last_action!=None:
                action = self.last_action
            else:
            #perform a random action
                action = np.random.randint(0,self.env.get_action_shape())
        
        self.last_action = action

        return action, action_probabilities

    def learn(self, nEpisodes, episodesBetweenModelBackups=-1, start_from_episode=0):
        initial_time = time.time()
        history=[]
        loss_values = []
        for episode in range(start_from_episode,nEpisodes):
            self.log("Episode #", episode, writeToFile=True, debug_channel="learning")

            max_reward, mean_kl_div, mean_value_loss, mean_policy_loss = self.training_step(episode)            

            self.log_history(
                max_reward,
                mean_kl_div,
                mean_value_loss,
                mean_policy_loss
            )
            #env.render_agent(agent)
            
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

    def log_history(self, max_reward, kl_div, value_loss, policy_loss):
        self.logger.log_history(max_reward, kl_div, value_loss, policy_loss)

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
        "rollouts",
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

    agent = TRPOAgent(env,logger,steps_per_rollout=2048,steps_between_rollouts=1, rollouts_per_sampling=16, multithreaded_rollout=True, batch_size=4096, DELTA=0.01,
    debug_rollouts=True, debug_act=True, debug_training=True, debug_model=True, debug_learning=True)
    
    #initial_time = time.time()
    #env.collect_rollouts(agent,10,1000)
    #first_time = time.time()
    #env.collect_rollouts_multithreaded(agent,10,1000,15)
    #second_time = time.time()

    #print("Non-multithreaded: ",first_time-initial_time,", multithreaded: ",second_time-first_time)
    #agent.load_weights(5)

    #agent.training_step(0)
    history = agent.learn(500,episodesBetweenModelBackups=5,start_from_episode=0)
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