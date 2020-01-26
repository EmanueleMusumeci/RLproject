import math
import numpy as np
import inspect

import tensorflow as tf
from tensorflow import keras
from Logger import Logger

EPS = 1e-8

def nn_model(input_shape, output_shape, convolutional=False):
	model = keras.Sequential()
	if convolutional:
		model.add(keras.layers.Lambda(lambda x: tf.cast(tf.image.resize(tf.image.rgb_to_grayscale(x), size=(32,32)), dtype=tf.float64)/256., input_shape=input_shape))
		model.add(keras.layers.Conv2D(10, (3, 3), activation='relu'))
		model.add(keras.layers.MaxPooling2D((3, 3)))
		model.add(keras.layers.Conv2D(5, (3, 3), activation='relu'))
		model.add(keras.layers.MaxPooling2D((3, 3)))
		model.add(keras.layers.Flatten())
	# else:
	model.add(keras.layers.Dense(64, input_shape=input_shape, activation='relu'))
	model.add(keras.layers.Dense(64, activation='relu'))
	model.add(keras.layers.Dense(output_shape))
	return model

def flatgrad(loss_fn, var_list):
	with tf.GradientTape() as t:
		loss = loss_fn()
	grads = t.gradient(loss, var_list, unconnected_gradients=tf.UnconnectedGradients.ZERO)
	return tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)

def assign_vars(model, theta):
		"""
		Create the process of assigning updated vars
		"""
		#crea la lista delle shape di ogni strato della rete neurale
		shapes = [v.shape.as_list() for v in model.trainable_variables]
		#calcola il numero complessivo di parametri della rete neurale
		size_theta = np.sum([np.prod(shape) for shape in shapes])
		# self.assign_weights_op = tf.assign(self.flat_weights, self.flat_wieghts_ph)
		start = 0
		for i, shape in enumerate(shapes): #per ogni strato
			#calcola il numero di parametri di quello strato
			size = np.prod(shape)
			#estrai i parametri relativi a quello strato dal vettore theta
			param = tf.reshape(theta[start:start + size], shape)
			#assegna tali parametri allo strato corrispondente nella rete neurale model
			model.trainable_variables[i].assign(param)

			#accumula il numero di parametri assegnati
			start += size
			#verifica con una assertion che il numero di parametri assegnati sia corretto
		assert start == size_theta, "messy shapes"

def flatvars(model):
	return tf.concat([tf.reshape(v, [-1]) for v in model.trainable_variables], axis=0)

####################################

def mean_kl_divergence(old_policy_probabilities,new_policy_probabilities, logger=None):
    kl = tf.reduce_mean(tf.reduce_sum(old_policy_probabilities * tf.math.log(old_policy_probabilities / new_policy_probabilities),axis=1))
    if math.isinf(kl):
        if logger!=None:
            log("Inf divergence: old_policy_probabilities: ",old_policy_probabilities,", new_policy_probabilities: ", new_policy_probabilities,writeToFile=True,debug_channel="utils_kl", skip_stack_levels=2, logger=logger)
    elif math.isnan(kl):
        if logger!=None:
            log("nan divergence: old_policy_probabilities: ",old_policy_probabilities,", new_policy_probabilities: ", new_policy_probabilities,writeToFile=True,debug_channel="utils_kl", skip_stack_levels=2, logger=logger)
    else:
        if logger!=None:
            log("Divergence: ",kl,", old_policy_probabilities: ",old_policy_probabilities,", new_policy_probabilities: ", new_policy_probabilities,writeToFile=True,debug_channel="utils_kl", skip_stack_levels=2, logger=logger)
    return kl


def caller_name(skip=2):
    """Get a name of a caller in the format module.class.method
    
       `skip` specifies how many levels of stack to skip while getting caller
       name. skip=1 means "who calls me", skip=2 "who calls my caller" etc.
       
       An empty string is returned if skipped levels exceed stack height
    """
    stack = inspect.stack()
    start = 0 + skip
    if len(stack) < start + 1:
      return ''
    parentframe = stack[start][0]    
    
    name = []
    module = inspect.getmodule(parentframe)
    # `modname` can be None when frame is executed directly in console
    # TODO(techtonik): consider using __main__
    if module:
        name.append(module.__name__)
    # detect classname
    if 'self' in parentframe.f_locals:
        # I don't know any way to detect call from the object method
        # XXX: there seems to be no way to detect static method call - it will
        #      be just a function call
        name.append(parentframe.f_locals['self'].__class__.__name__)
    codename = parentframe.f_code.co_name
    if codename != '<module>':  # top level usually
        name.append( codename ) # function or a method
    del parentframe
    return ".".join(name)

def print_debug(obj=None,debug=True,*strings):
    if not debug: return
    if obj!=None and hasattr(obj,"debug"):
        if obj.debug: 
            print("[",caller_name(),"] ", end="")
            for s in strings:
                print(str(s), end="")
            print()
    else:
        print("[",caller_name(),"]: ", end="")
        for s in strings:
            print(str(s), end="")
        print()

def log(*strings, writeToFile=False, debug_channel="Generic", skip_stack_levels=2, logger=None):
    strings = [str(s) for s in strings]
    if logger!=None:
        logger.print(strings, writeToFile=writeToFile, debug_channel=debug_channel, skip_stack_levels=skip_stack_levels)

def GAE(value_model, current_batch_observations, current_batch_rewards, GAMMA, LAMBDA):
    def discount(rewards, GAMMA):
        discounted_rewards = []
        discounted_reward=0
        for reward in reversed(rewards):
            discounted_reward = reward + self.GAMMA*discounted_reward
            discounted_rewards.insert(0, discounted_reward)
        return discounted_rewards
    
    discounted_rewards = []
    advantages = []

    state_values = value_model(current_batch_observations).numpy().flatten()
    self.log("State values: ", state_values, debug_channel="advantage")
        # Compute discounted rewards with a 'bootstrapped' final value.
    rs_bootstrap = [] if current_batch_rewards == [] else current_batch_rewards + [state_values[-1]]
    discounted_rewards.extend(discount(rs_bootstrap, GAMMA)[:-1])


    # Compute advantages for each environment using Generalized Advantage Estimation;
    # see eqn. (16) of [Schulman 2016].
    #ValueError: operands could not be broadcast together with shapes (4096,) (4095,) 
    #delta_t = current_batch_rewards + GAMMA*state_values[1:] - state_values[:-1]
    #Eq. 16 https://arxiv.org/pdf/1506.02438.pdf
    delta_t = current_batch_rewards[:-1] + GAMMA*state_values[1:] - state_values[:-1]
    advantages = discount(delta_t, GAMMA*LAMBDA)
    advantages.insert(0,GAMMA*state_values[0])

    return advantages