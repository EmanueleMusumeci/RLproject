import os
import glob
import gym
import numpy as np
import time
from collections import namedtuple

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

RolloutStatistics = namedtuple("RolloutStatistics","actions action_probabilities rewards discounted_rewards mean_discounted_rewards observations advantages size")
TrainingInfo = namedtuple("TrainingInfo","value_loss linesearch_successful mean_kl_divergence")

class TRPOAgent:
    def __init__(self, 
    env,
    logger,
    #Rollouts
    steps_per_rollout=0, steps_between_rollouts=0, rollouts_per_sampling=10, 
    
    #Training
    batch_size = 4096,

    #Coefficients
    conjugate_gradients_damping=0.001, kl_max_substitutive_param=0.01, backtrack_coeff=.6, backtrack_iters=10, gamma = 0.99 ,

    #Debug
    debug_rollouts=False, debug_act=False, debug_training=False, debug_model=False, debug_learning=False,

    #Others
    model_backup_dir="TRPO_project/Models"):

        assert(env!=None), "You need to create an environment first"
        self.env = env
        self.model_backup_dir = model_backup_dir

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
        self.kl_max_substitutive_param = kl_max_substitutive_param
        self.backtrack_coeff = backtrack_coeff
        self.backtrack_iters = backtrack_iters

        #********
        #ROLLOUTS
        #********
        self.rollouts_per_sampling = rollouts_per_sampling
        self.steps_per_rollout = steps_per_rollout
        self.steps_between_rollouts = steps_between_rollouts
        self.steps_to_new_rollout = 0
        self.steps_from_last_rollout = self.steps_between_rollouts

        self.rollout_statistics = []

        #********
        #TRAINING
        #********
        self.batch_size = batch_size

        #TRAINING

        #This will get updated every training_step
        self.policy = Policy(self.env, self.logger, debug=debug_model)

        #This will be updated for every batch cycle inside the training step, in order to compute the ratio
        #between the new and the old policy in the surrogate function
        self.new_model = self.policy.clone()

        self.state_value = Value(self.env, self.logger, debug=debug_model)

        self.theta_old = self.policy.get_flat_params()
        if steps_per_rollout==0: self.steps_per_rollout = self.env.max_episode_steps

    def collect_rollout_statistics(self):
        rollouts, done = env.collect_rollouts(self,self.rollouts_per_sampling,self.steps_per_rollout)

        def collect_actions(rollouts):
            actions = []
            action_probabilities = []
            #rollouts is a list of lists (each one of these lists is a single rollout)
            for rollout in rollouts:
                #rollout is a list of RolloutTuple namedtuples
                for tupl in rollout:
                    actions.append(tupl.action)
                    action_probabilities.append(tupl.action_probabilities)
            return actions, action_probabilities

        def collect_rewards(rollouts, gamma):
            rewards = []
            discounted_rewards = []
            mean_discounted_rewards = []
            for rollout in rollouts:
                discounted_reward = 0
                current_gamma = 1
                for tupl in rollout:
                    rewards.append(tupl.reward)
                    discounted_reward += tupl.reward * current_gamma
                    current_gamma *= gamma

                    #contain the total discounted reward (Q-value) for each class
                    discounted_rewards.append(discounted_reward)
                mean_discounted_rewards.append(np.mean(discounted_rewards))
            return rewards, discounted_rewards, mean_discounted_rewards

        def collect_observations(rollouts):
            observations = []
            for rollout in rollouts:
                for tupl in rollout:
                    observations.append(tupl.observation)
            return observations

        def compute_advantages(discounted_rewards, observations, value_model):
            self.log("observations: ",observations,", len: ", len(observations), writeToFile=True, debug_channel="rollouts")
            self.log("discounted_rewards: ",discounted_rewards,", len: ", len(discounted_rewards), writeToFile=True, debug_channel="rollouts")
            assert(len(discounted_rewards)==len(observations))
            advantages = []
            for i in range(len(discounted_rewards)):
                q_value = discounted_rewards[i]
                value_prediction = value_model(np.array([observations[i]]))
                #for some reason value_prediction is a tuple
                advantage_value = (q_value - value_prediction).ravel()[0]
                advantages.append(advantage_value)
            return advantages

        actions, action_probabilities = collect_actions(rollouts)
        rewards, discounted_rewards, mean_discounted_rewards = collect_rewards(rollouts, self.gamma)
        observations = collect_observations(rollouts)
        advantages = compute_advantages(discounted_rewards, observations, self.state_value)
        assert(len(actions)==len(action_probabilities)==len(rewards)==len(discounted_rewards)==len(observations)==len(advantages))
        size = len(actions)

        statistics = RolloutStatistics(
            actions,
            action_probabilities,
            rewards,
            discounted_rewards,
            mean_discounted_rewards, #(Q_values)
            observations,
            advantages,
            size
            )

        return statistics, done


    #TODO: implement the surrogate function
    def surrogate_function(self, args):
        current_batch_observations = args["observation"]
        actions_one_hot = args["actions_one_hot"]
        advantage = args["advantage"]
        
        #Calcola la nuova policy usando la rete neurale aggiornata ad ogni batch cycle del training step
        new_policy_action_probabilites = tf.nn.softmax(self.new_model(current_batch_observations, tensor=True))
        #Calcola la vecchia policy usando la vecchia rete neurale salvata
        old_policy_action_probabilites = tf.nn.softmax(self.policy(current_batch_observations, tensor=True))
        
        #new_policy_action_probabilites = tf.Variable(args["new_policy_action_probabilites"])
        #old_policy_action_probabilites = tf.Variable(args["old_policy_action_probabilites"])
        self.log("actions_one_hot: ",actions_one_hot, writeToFile=True, debug_channel="training")
        self.log("new_policy_action_probabilities: ",new_policy_action_probabilites, writeToFile=True, debug_channel="training")
        self.log("old_policy_action_probabilities: ",old_policy_action_probabilites, writeToFile=True, debug_channel="training")

#    #TODO: Valutare se axis=0 o axis=1
        new_policy_action_probabilites = tf.reduce_sum(actions_one_hot * new_policy_action_probabilites, axis=1)
        old_policy_action_probabilites = tf.reduce_sum(actions_one_hot * old_policy_action_probabilites, axis=1)

        self.log("reduced new_policy_action_probabilities: ",new_policy_action_probabilites, writeToFile=True, debug_channel="training")
        self.log("reduced old_policy_action_probabilities: ",old_policy_action_probabilites, writeToFile=True, debug_channel="training")

        policy_ratio = tf.math.divide(new_policy_action_probabilites, old_policy_action_probabilites) # (Schulman et al., 2017) Equation(14)
        #logger.print("policy_ratio: ", policy_ratio)
        #Calcola la loss function: ha anche aggiunto un termine di entropia
        loss = tf.reduce_mean(tf.math.multiply(policy_ratio, advantage))
        self.log("loss value: ", loss, writeToFile=True, debug_channel="training")
        return loss

    def training_step(self, episode):

        if self.steps_from_last_rollout == self.steps_between_rollouts:
            self.steps_from_last_rollout = 0
            self.rollout_statistics, done = self.collect_rollout_statistics()

            #Change theta_old old policy params once every steps_between_rollouts rollouts
            theta = self.policy.get_flat_params()
            self.theta_old = theta
        

#DONE#TODO:Spostare tutto ciò che segue in un ciclo che cicla su tutti i batch
#    #TODO:Capire perché se metto la generazione dei logits tramite softmax fuori dalla surrogate function il gradiente è nullo
#          (per evitare di ripeterla due volte) 
        number_of_batches = max(self.rollout_statistics.size // self.batch_size, 1) 
        #the max covers the case where the size of data collected in the rollouts is lower than the batch_size

        actions = np.array_split(self.rollout_statistics.actions, number_of_batches)

        action_probabilities = np.array_split(self.rollout_statistics.action_probabilities, number_of_batches)
        rewards = np.array_split(self.rollout_statistics.rewards, number_of_batches)
        Q_values = np.array_split(self.rollout_statistics.mean_discounted_rewards, number_of_batches)
        discounted_rewards = np.array_split(self.rollout_statistics.discounted_rewards, number_of_batches)

        observations = np.array_split(self.rollout_statistics.observations, number_of_batches)

        advantages = np.array(self.rollout_statistics.advantages)
        advantages = (advantages - advantages.mean())/(advantages.std() + 1e-8)
        advantages = np.array_split(advantages, number_of_batches)

        for batch in range(number_of_batches):

            current_batch_actions = actions[batch]
            current_batch_observations = observations[batch]
            current_Q_values = Q_values[batch]
            current_batch_advantages = advantages[batch]
            current_batch_discounted_rewards = discounted_rewards[batch]

            #self.new_model.set_flat_params(np.array(1,len(self.policy.get_flat_params())+2, dtype=float))
            #FOR TESTING
            #to_fill = np.ones_like(self.policy.get_flat_params(), dtype = float)
            # training, to_fill)
            #self.policy.set_flat_params(to_fill)

            current_batch_actions_one_hot = tf.one_hot(current_batch_actions, self.env.get_action_shape(), dtype="float64")


            #args to pass to the loss function in order to get the gradient
            args = {
                "observation" : current_batch_observations,
                "actions_one_hot" : current_batch_actions_one_hot,
                "advantage" : current_batch_advantages
            }

            policy_gradient = self.policy.get_flat_gradients(self.surrogate_function, args)
            self.log("policy_gradient: ", policy_gradient, writeToFile=True, debug_channel="training")

            #Calcola la nuova policy usando la rete neurale aggiornata ad ogni batch cycle del training step
            new_policy_action_probabilities = tf.nn.softmax(self.new_model(current_batch_observations, tensor=True))
            #Calcola la vecchia policy usando la vecchia rete neurale salvata
            old_policy_action_probabilities = tf.nn.softmax(self.policy(current_batch_observations, tensor=True))

            mean_kl_div = mean_kl_divergence(old_policy_action_probabilities, new_policy_action_probabilities)
            self.log("mean_kl_divergence: ", mean_kl_div, writeToFile=True, debug_channel="training")
            
            gradient_step_direction = conjugate_gradients(self.policy, -policy_gradient, self.conjugate_gradients_damping, 10, old_policy_action_probabilities, new_policy_action_probabilities)
            self.log("gradient_step_direction: ", gradient_step_direction, writeToFile=True, debug_channel="training")

            fvp = fisher_vectors_product(self.policy,old_policy_action_probabilities,new_policy_action_probabilities,gradient_step_direction,self.conjugate_gradients_damping).numpy()
            self.log("fisher_vectors_product: ", fvp, writeToFile=True, debug_channel="training")


            shs = 0.5 * np.dot(gradient_step_direction,fvp.T)


            #shs = .5 * step_direction.dot(hessian_vector_product(step_direction).T)



            lm = np.sqrt(shs / self.kl_max_substitutive_param) + 1e-8


            max_gradient_step = gradient_step_direction / lm

            #neggdotstepdir = (-loss_grad * stepdir).sum(0, keepdim=True)
            #print(("lagrange multiplier:", lm[0], "grad_norm:", loss_grad.norm()))
            def linesearch():
                theta = self.policy.get_flat_params()
                loss_value = self.surrogate_function(args)
                self.log("loss_value before", loss_value)

                for (_n_backtracks, stepfrac) in enumerate(self.backtrack_coeff**np.arange(self.backtrack_iters)):

                    theta_new = theta + stepfrac * max_gradient_step
                    self.new_model.set_flat_params(theta_new)

                    new_loss_value = self.surrogate_function(args).numpy()

                    new_policy_action_probabilities = tf.nn.softmax(self.new_model(current_batch_observations, tensor=True))
                    mean_kl_div = mean_kl_divergence(old_policy_action_probabilities,new_policy_action_probabilities).numpy()

                    improvement = loss_value - new_loss_value

                    #print_debug("improvement: ", improvement)
                    
                    if mean_kl_div <= self.kl_max_substitutive_param and new_loss_value >= 0:
                        self.log("Linesearch worked at ", _n_backtracks, writeToFile=True, debug_channel="training")
                        return True, theta_new, mean_kl_div
                    if _n_backtracks == self.backtrack_iters - 1:
                        self.log("Linesearch failed. Mean kl divergence: ", mean_kl_div, ", New (policy) loss value: ", new_loss_value, ", theta new: ", theta_new, writeToFile=True, debug_channel="training")
                        return False, theta, mean_kl_div

            #success, new_theta = linesearch(self.policy, max_gradient_step, self.backtrack_coeff, self.backtrack_iters, mean_kl_div, self.kl_max_substitutive_param, self.surrogate_function, args, self.new_model)
            success, new_theta, mean_kl_div = linesearch()
                    
            self.policy.set_flat_params(new_theta)
            
            history = self.state_value.fit(current_batch_observations, current_batch_discounted_rewards, epochs=5, verbose=0)
            value_loss = history.history["loss"][-1]

        #    #TODO:find a way to also train the value network

            self.log("\n\n\nEND OF TRAINING BATCH #", batch,"\n\n\n")
            
        self.log("\n\n\nEND OF TRAINING STEP #", episode,"\n\n\n")


        self.steps_from_last_rollout += 1

        return TrainingInfo(value_loss, success, mean_kl_div)

    #the ACT function: produces an action from the current policy
    def act(self,observation):
        #Trasforma la observation in un vettore riga [[array]]
        observation = observation[np.newaxis, :]
        self.log("observation: ",observation,", len: ",len(observation), writeToFile=True, debug_channel="act")

        policy_output = self.policy(observation)
        action_probabilities = tf.nn.softmax(policy_output).numpy().ravel()
        self.log("action_probabilites: ",action_probabilities,", len: ",len(action_probabilities), writeToFile=True, debug_channel="act")

        action = np.random.choice(range(action_probabilities.shape[0]), p=action_probabilities)
        self.log("chosen_action: ",action, writeToFile=True, debug_channel="act")
        #Estrai la prossima azione dalla NN della policy
        #action, action_probabilities = self.policy(observation)
        return action, action_probabilities

#TODO decidere se applicare la softmax all'output della rete neurale o mettere uno strato di output softmax
#(preferire la seconda se l'output è uguale alla prima)
    
    def learn(self, nEpisodes, episodesBetweenModelBackups=-1):
        initial_time = time.time()
        history=[]
        loss_values = []
        for episode in range(nEpisodes):
            self.log("Episode #", episode, writeToFile=True, debug_channel="learning")
    #        if episode>0:
                #print_debug(self,self.debug_learning, "Previous loss value: ", loss_values[episode-1])
    
            training_info = self.training_step(episode)
            self.log("Training info: ",training_info, writeToFile=True, debug_channel="learning")

            history.append(training_info)
            loss_values.append(training_info.value_loss)

            #env.render_agent(agent)
            
            #plt.plot(loss_values)
            #plt.show()

            import os
            print(os.getcwd())
            if episodesBetweenModelBackups!=-1 and episode!=0 and episode%episodesBetweenModelBackups == 0:
                self.save_policy_weights(episode)
                self.save_value_weights(episode)

            
    def save_policy_weights(self,episode):
        filename = self.env.environment_name+"/Policy."+self.env.get_environment_description()+"."+str(episode)
        self.policy.save_model_weights(self.model_backup_dir+"/"+filename)

    def save_value_weights(self,episode):
        filename = self.env.environment_name+"/Value."+self.env.get_environment_description()+"."+str(episode)
        self.state_value.save_model_weights(self.model_backup_dir+"/"+filename)

    def log(self, *strings, writeToFile=False, debug_channel="Generic"):
        if self.logger!=None:
            self.logger.print(strings, writeToFile=writeToFile, debug_channel=debug_channel)

    def load_weights(self, episode, dir=""):
        if dir=="": dir = self.model_backup_dir
        policy_filename = self.env.environment_name+"/Policy."+self.env.get_environment_description()+"."+str(episode)
        self.policy.load_model_weights(dir+"/"+policy_filename)

        value_filename = self.env.environment_name+"/Value."+self.env.get_environment_description()+"."+str(episode)
        self.state_value.load_model_weights(dir+"/"+value_filename)

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
        #"act",
        #"training",
        "learning",
        #"model",
        #"EnvironmentRegister",
        #"environment"
        ]

    logger = Logger(name=env_name,log_directory="TRPO_project/Models/"+env_name,debugChannels=channels)

    env = Environment(env_name,logger,use_custom_env_register=True, debug=True, show_preprocessed=False)
    #env = Environment(env_name,None,use_custom_env_register=True, debug=True, show_preprocessed=False)

    agent = TRPOAgent(env,logger,gamma=0.99,steps_per_rollout=100,steps_between_rollouts=10,rollouts_per_sampling=10, 
    debug_rollouts=True, debug_act=True, debug_training=True, debug_model=True, debug_learning=True)
    #agent = TRPOAgent(env,None,gamma=0.99,steps_per_rollout=1,steps_between_rollouts=1,rollouts_per_sampling=1, 
    #debug_rollouts=True, debug_act=True, debug_training=True, debug_model=True, debug_learning=True)

    #agent.learn(100,5)

    #loss = agent.training_step(0)
    
    #rollouts, _ = env.collect_rollouts(agent, 1, 2)
    #roll, done = agent.collect_self.rollout_statistics()
    #obs = env.reset()
    #action, _ = agent.act(obs)
    
    agent.load_weights(60)
    
    env.render_agent(agent)
    #env.close()

#    #TODO:RISOLVERE PROBLEMA NAN E INF SULLA DIVERGENZA (SOMMARE UN VALORE INFINITESIMO)