import os
import glob
import gym
import numpy as np
import time
from collections import namedtuple

import tensorflow as tf
from tensorflow import GradientTape
from tensorflow import keras

from utils import *
from Environment import Environment, RolloutTuple
from Models import Policy, Value

RolloutStatistics = namedtuple("RolloutStatistics","actions action_probabilities rewards discounted_rewards mean_discounted_reward observations advantages")

class TRPOAgent:
    def __init__(self, env, gamma, steps_per_rollout=0, steps_between_rollouts=0, rollouts_per_sampling=10):

        self.env = env

        self.rollouts_per_sampling = rollouts_per_sampling
        self.steps_per_rollout = steps_per_rollout

        self.steps_between_rollouts = steps_between_rollouts
        self.steps_to_new_rollout = 0

        #Gamma
        assert(gamma<1)
        assert(gamma>0)
        self.gamma = gamma

        self.policy = Policy(env)
        self.state_value = Value(self.env)

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
            for rollout in rollouts:
                current_discounted_reward = 0
                current_gamma = 1
                for tupl in rollout:
                    rewards.append(tupl.reward)
                    current_discounted_reward += tupl.reward * current_gamma
                    current_gamma *= gamma
                    discounted_rewards.append(current_discounted_reward)
            
            mean_discounted_reward = current_discounted_reward/len(rollout)
            return rewards, discounted_rewards, mean_discounted_reward

        def collect_observations(rollouts):
            observations = []
            for rollout in rollouts:
                for tupl in rollout:
                    observations.append(tupl.observation)
            return observations

        def compute_advantages(discounted_rewards, observations, value_model):
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
        rewards, discounted_rewards, mean_discounted_reward = collect_rewards(rollouts, self.gamma)
        observations = collect_observations(rollouts)
        advantages = compute_advantages(discounted_rewards, observations, self.state_value) 

        statistics = RolloutStatistics(
            actions, 
            action_probabilities, 
            rewards, 
            discounted_rewards, 
            mean_discounted_reward, 
            observations, 
            advantages
            )

        return statistics, done


    #TODO: implement the surrogate function
    def surrogate_function(self):
        return 0.0

    def training_step(self, model, get_kl, max_kl, damping):

        rollout_statistics, done = self.collect_rollout_statistics()

        theta = self.policy.get_flat_params()
        if self.steps_between_rollouts>0: self.theta_old = theta

        actions = rollout_statistics.actions

        #NOTICE: this is an array(...)
        action_probabilities = rollout_statistics.action_probabilities
        rewards = rollout_statistics.rewards
        discounted_rewards = rollout_statistics.discounted_rewards

        #NOTICE: this is an array(...)
        observations = rollout_statistics.observations
        advantages = rollout_statistics.advantages
        mean_discounted_reward = rollout_statistics.mean_discounted_reward

        loss_function = self.surrogate_function()

        #TODO: Find a way to compute the gradients
        #grads = torch.autograd.grad(loss, model.parameters())
        #loss_grad = torch.cat([grad.view(-1) for grad in grads]).data
        policy_gradient = self.policy.get_flat_gradient(loss_function)

        if False:'''
        stepdir = conjugate_gradients(Fvp, -loss_grad, 10)

        shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0, keepdim=True)

        lm = torch.sqrt(shs / max_kl)
        fullstep = stepdir / lm[0]

        neggdotstepdir = (-loss_grad * stepdir).sum(0, keepdim=True)
        print(("lagrange multiplier:", lm[0], "grad_norm:", loss_grad.norm()))

        prev_params = get_flat_params_from(model)
        success, new_params = linesearch(model, get_loss, prev_params, fullstep,
                                        neggdotstepdir / lm[0])
        
        set_flat_params_to(self.policy, new_params)
        #find a way to also train the value network
        '''
        return 0.0

    #the ACT function: produces an action from the current policy
    def act(self,observation):
        #Trasforma la observation in un vettore riga [[array]]
        observation = observation[np.newaxis, :]
        policy_output = self.policy(observation)
        action_probabilities = tf.nn.softmax(policy_output).numpy().ravel()
        action = np.random.choice(range(action_probabilities.shape[0]), p=action_probabilities)

        #Estrai la prossima azione dalla NN della policy
        #action, action_probabilities = self.policy(observation)
        return action, action_probabilities

#TODO decidere se applicare la softmax all'output della rete neurale o mettere uno strato di output softmax
#(preferire la seconda se l'output è uguale alla prima)

    if False: '''
    def learn(self, nEpisodes, episodesBetweenModelBackups=-1):
        initial_time = time.time()
        for episode in range(nEpisodes):
            rollouts = self.env.collect_rollouts()
            loss_value, max_reward = self.training_step(episode, rollouts)
            if episodesBetweenModelBackups!=-1 and episode%episodesBetweenModelBackups == 0:
                self.save_policy_weights()
    '''
    

if __name__ == '__main__':
    env = Environment("MountainCar-v0",use_custom_env_register=True,show_debug_info=True, show_preprocessed=True)
    print(env.rendering_delay)
    agent = TRPOAgent(env,gamma=0.99,steps_per_rollout=2,steps_between_rollouts=15,rollouts_per_sampling=2)
    #rollouts, _ = env.collect_rollouts(agent, 1, 2)
    #print(rollouts)
    roll, done = agent.collect_rollout_statistics()
    print(roll)
    obs = env.reset()
    action, _ = agent.act(obs)
    env.render_agent(agent)
    env.close()
    