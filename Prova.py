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
    value_learning_rate=1e-1,
    value_function_fitting_epochs=5,
    epsilon_greedy=True,
    epsilon=0.2,
    epsilon_decrease_factor=0.000,

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
    
    def train_step(self, episode, obs_all, Gs_all, actions_all, action_probs_all, total_reward, best_reward, entropy, t0):
        def surrogate_loss(theta=None):
			if theta is None:
				model = self.model
			else:
				model = self.tmp_model
				assign_vars(self.tmp_model, theta)

			#Calcola la nuova policy
			logits = model(obs)
			action_prob = tf.nn.softmax(logits)
			action_prob = tf.reduce_sum(actions_one_hot * action_prob, axis=1)

			#Calcola la vecchia policy usando la vecchia rete neurale salvata
			old_logits = self.model(obs)
			old_action_prob = tf.nn.softmax(old_logits)
			old_action_prob = tf.reduce_sum(actions_one_hot * old_action_prob, axis=1).numpy() + 1e-8
			
			prob_ratio = action_prob / old_action_prob # pi(a|s) / pi_old(a|s)

			#Calcola la loss function: ha anche aggiunto un termine di entropia
			loss = tf.reduce_mean(prob_ratio * advantage) + self.ent_coeff * entropy
			return loss

		def kl_fn(theta=None):
			if theta is None:
				model = self.model
			else:
				model = self.tmp_model
				assign_vars(self.tmp_model, theta)
			logits = model(obs)
			action_prob = tf.nn.softmax(logits).numpy() + 1e-8
			old_logits = self.model(obs)
			old_action_prob = tf.nn.softmax(old_logits)
			return tf.reduce_mean(tf.reduce_sum(old_action_prob * tf.math.log(old_action_prob / action_prob), axis=1))

		def hessian_vector_product(p):
			def hvp_fn(): 
				kl_grad_vector = flatgrad(kl_fn, self.model.trainable_variables)
				grad_vector_product = tf.reduce_sum(kl_grad_vector * p)
				return grad_vector_product

			fisher_vector_product = flatgrad(hvp_fn, self.model.trainable_variables).numpy()
			return fisher_vector_product + (self.cg_damping * p)

		def conjugate_grad(Ax, b):
			"""
			Conjugate gradient algorithm
			(see https://en.wikipedia.org/wiki/Conjugate_gradient_method)
			"""
			x = np.zeros_like(b)
			r = b.copy() # Note: should be 'b - Ax(x)', but for x=0, Ax(x)=0. Change if doing warm start.
			p = r.copy()
			old_p = p.copy()
			r_dot_old = np.dot(r,r)
			for _ in range(self.cg_iters):
				z = Ax(p)
				alpha = r_dot_old / (np.dot(p, z) + 1e-8)
				old_x = x
				x += alpha * p
				r -= alpha * z
				r_dot_new = np.dot(r,r)
				beta = r_dot_new / (r_dot_old + 1e-8)
				r_dot_old = r_dot_new
				if r_dot_old < self.residual_tol:
					break
				old_p = p.copy()
				p = r + beta * p
				if np.isnan(x).any():
					print("x is nan")
					print("z", np.isnan(z))
					print("old_x", np.isnan(old_x))
					print("kl_fn", np.isnan(kl_fn()))
			return x

		def linesearch(x, fullstep):
			fval = surrogate_loss(x)
			for (_n_backtracks, stepfrac) in enumerate(self.backtrack_coeff**np.arange(self.backtrack_iters)):
				xnew = x + stepfrac * fullstep
				newfval = surrogate_loss(xnew)
				kl_div = kl_fn(xnew)
				if np.isnan(kl_div):
					print("kl is nan")
					print("xnew", np.isnan(xnew))
					print("x", np.isnan(x))
					print("stepfrac", np.isnan(stepfrac))
					print("fullstep",  np.isnan(fullstep))
				if kl_div <= self.delta and newfval >= 0:
					print("Linesearch worked at ", _n_backtracks)
					return xnew
				if _n_backtracks == self.backtrack_iters - 1:
					print("Linesearch failed.", kl_div, newfval)
			return x

		#Calcola il numero di batch da dare in pasto alla rete neurale
		NBATCHES = len(obs_all) // self.BATCH_SIZE 
		if len(obs_all) < self.BATCH_SIZE:
			NBATCHES += 1
		#Per ogni batch, recupera:
		for batch_id in range(NBATCHES):
			#il vettore degli stati
			obs = obs_all[batch_id*self.BATCH_SIZE: (batch_id + 1)*self.BATCH_SIZE]
			#il vettore dei discounted cumulative reward
			Gs = Gs_all[batch_id*self.BATCH_SIZE: (batch_id + 1)*self.BATCH_SIZE]
			#il vettore delle azioni
			actions = actions_all[batch_id*self.BATCH_SIZE: (batch_id + 1)*self.BATCH_SIZE]
			#il vettore delle policy
			action_probs = action_probs_all[batch_id*self.BATCH_SIZE: (batch_id + 1)*self.BATCH_SIZE]

			#CALCOLO ADVANTAGE FUNCTION, vedi: http://178.79.149.207/posts/trpo.html
			print(obs.shape)
			Vs = self.value_model(obs).numpy().flatten()
			# advantage = Gs
			advantage = Gs - Vs
			advantage = (advantage - advantage.mean())/(advantage.std() + 1e-8)
			actions_one_hot = tf.one_hot(actions, self.envs[0].action_space.n, dtype="float64")
			
			#CALCOLARE IL POLICY GRADIENT
			#Calcola la loss function da ottimizzare
			policy_loss = surrogate_loss()
			#Calcola il gradiente della loss function
			policy_gradient = flatgrad(surrogate_loss, self.model.trainable_variables).numpy()
			


			#Trova la direzione di massima crescita con il metodo del conjugate_gradient
			step_direction = conjugate_grad(hessian_vector_product, policy_gradient)
			shs = .5 * step_direction.dot(hessian_vector_product(step_direction).T)
			lm = np.sqrt(shs / self.delta) + 1e-8
			fullstep = step_direction / lm
			
			if np.isnan(fullstep).any():
				print("fullstep is nan")
				print("lm", lm)
				print("step_direction", step_direction)
				print("policy_gradient", policy_gradient)
			
			oldtheta = flatvars(self.model).numpy()

			theta = linesearch(oldtheta, fullstep)


			if np.isnan(theta).any():
				print("NaN detected. Skipping update...")
			else:
				assign_vars(self.model, theta)

			kl = kl_fn(oldtheta)

			history = self.value_model.fit(obs, Gs, epochs=5, verbose=0)
			value_loss = history.history["loss"][-1]


			print(f"Ep {episode}.{batch_id}: Rw_mean {total_reward} - Rw_best {best_reward} - PL {policy_loss} - VL {value_loss} - KL {kl} - epsilon {self.epsilon} - time {time.time() - t0}")
		#if self.value_model:
			#writer = self.writer
			#with writer.as_default():
				#tf.summary.scalar("reward", total_reward, step=episode)
				#tf.summary.scalar("best_reward", best_reward, step=episode)
				#tf.summary.scalar("value_loss", value_loss, step=episode)
				#tf.summary.scalar("policy_loss", policy_loss, step=episode)
		self.epsilon = self.epsilon_decay(self.epsilon)	

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
        #        action = np.random.randint(0,self.env.get_action_shape())
        
        #self.log("chosen_action: ",action, writeToFile=True, debug_channel="act")

        return action, action_probabilities

    def learn(self, nEpisodes, episodesBetweenModelBackups=-1, start_from_episode=0):
        initial_time = time.time()
        history_values = {"mean_reward":[],"value_loss":[],"policy_loss":[]}
        loss_values = []

        plot, subplots = plt.subplots(3, 1, constrained_layout=True)

        for episode in range(start_from_episode,nEpisodes):
            self.log("Episode #", episode, writeToFile=True, debug_channel="learning")

            if start_from_episode>0:
                self.load_weights(start_from_episode)

            rollout_until_success = False
            current_episode_steps_per_rollout = self.steps_per_rollout

            total_reward, mean_reward, value_loss, policy_loss = self.train_step()
            #history_entry = {"max_reward":max_reward,"value_loss":value_loss,"policy_loss":policy_loss}

            history_values["mean_reward"].append(mean_reward)
            history_values["value_loss"].append(value_loss)
            history_values["policy_loss"].append(policy_loss)

            self.log_history(history_values)
            self.plot_history(plot, subplots, history_values)

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
    
    env_name = "CartPole-v0"
    
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

    agent = TRPOAgent(env,logger,
    steps_per_rollout=1500,steps_between_rollouts=1, rollouts_per_sampling=100, 
    multithreaded_rollout=True, full_rollout_episodes=0, 
    single_batch_training = True, batch_size=500, 
    DELTA=0.03, epsilon=0.1, epsilon_greedy=True, 
    value_function_fitting_epochs=5, value_learning_rate=1e-2,
    backtrack_coeff=1e-2, backtrack_iters=10,
    render=False)
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