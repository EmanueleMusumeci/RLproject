import numpy as np
#from tensorflow.keras import layers, optimizers, losses, models
#from tensorflow.keras.layers import Dense, Sequential
#sets to float64 to avoid compatibility issues between numpy 
# (whose default is float64) and keras(whose default is float32)

from Environment import Environment
from Logger import Logger

import tensorflow as tf
from tensorflow import keras as keras


class Policy:
    def __init__(self, environment, logger, debug = False):

        self.logger = logger

        self.input_shape = environment.get_observation_shape()
        self.output_shape = environment.get_action_shape()
        self.environment = environment

        self.debug=debug

        #Determine if we're going to use a Convolutional NN (Image analysis required)
        self.convolutional = environment.type in ["atari"]
        
        self.log("input_shape: ", self.input_shape,", output_shape: ",self.output_shape, ", convolutional: ",self.convolutional, writeToFile=True, debug_channel="model")

        self.model = keras.Sequential()
        if not self.convolutional:
            self.model.add(keras.layers.Dense(64, activation="tanh",input_shape=self.input_shape))
            self.model.add(keras.layers.Dense(32, activation="tanh"))
            self.model.add(keras.layers.Dense(self.output_shape))
        else:
            self.model.add(keras.layers.Conv2D(10, (3, 3), input_shape=self.input_shape, activation='relu'))
            self.model.add(keras.layers.MaxPooling2D((3, 3)))
            self.model.add(keras.layers.Conv2D(5, (3, 3), activation='relu'))
            self.model.add(keras.layers.MaxPooling2D((3, 3)))
            self.model.add(keras.layers.Flatten())
            self.model.add(keras.layers.Dense(self.output_shape))            

		#crea la lista delle shape di ogni strato della rete neurale
        self.shape_list = [layer.shape.as_list() for layer in self.model.trainable_variables]

		#calcola il numero complessivo di parametri della rete neurale
        self.number_of_parameters = np.sum([np.prod(shape) for shape in self.shape_list])
        self.logger.print("number_of_parameters: ",self.number_of_parameters, writeToFile=True, debug_channel="model")
        stringlist=[]
        self.model.summary(print_fn=lambda x: stringlist.append(x))
        model_summary = "\n".join(stringlist)
        self.logger.print(model_summary,writeToFile=True,debug_channel="model_summary")
    
    def __call__(self, observation):
        if self.convolutional:
            observation = np.array(observation)
            #Add an axis if the image has only 1 channel (grayscale) because the shape has to be like 
            # (batch size, img_height, img_width, channels) but if the image is grayscale, the shape is
            # (batch_size, img_height, img_width)
            if self.input_shape[2]==1:
                observation = observation.reshape(observation.shape[0], observation.shape[1], observation.shape[2], 1)
            if self.input_shape[2]==3:
                observation = observation.reshape(observation.shape[0], observation.shape[1], observation.shape[2], 3)
 
            observation = tf.cast(observation,'float64')
        
        return self.model(np.array(observation))

    def get_flat_params(self, numpy=True):
        params = tf.concat([tf.reshape(v, [-1]) for v in self.model.trainable_variables], axis=0)
        if numpy:
            return params.numpy()
        else:
            return params

    def set_flat_params(self, flat_params):

        current_layer=0
        assigned_parameters = 0
        shape_list = [layer.shape.as_list() for layer in self.model.trainable_variables]
        for shape in shape_list: 
            #per ogni strato
            #calcola il numero di parametri di quello strato
            layer_size = np.prod(shape)
            #estrai i parametri relativi a quello strato dal vettore theta
            theta = tf.reshape(flat_params[assigned_parameters:assigned_parameters + layer_size], shape)
            #assegna tali parametri allo strato corrispondente nella rete neurale model
            self.model.trainable_variables[current_layer].assign(theta)

            #accumula il numero di parametri assegnati
            assigned_parameters += layer_size
            current_layer+=1

        #verifica con una assertion che il numero di parametri assegnati sia corretto
        assert assigned_parameters == self.number_of_parameters

    #Computes the gradients wrt the weights of the neural network
    def get_flat_gradients(self, function, args=None):
        #create a tensorflow array out of the numpy one containing the nn parameters
        trainable_variables = self.model.trainable_variables

        with tf.GradientTape() as t:
            f= function()
        
        gradient = t.gradient(f, trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            #UnconnectedGradients.ZERO serve a evitare che ti restituisca None quando il grafo dei gradienti non è connesso
        return tf.concat([tf.reshape(v, [-1]) for v in gradient], axis=0)

    def save_model_weights(self,filename):
        self.model.save_weights(filename + ".h5")

    def load_model_weights(self, filename):
        self.model.load_weights(filename + ".h5")

    def clone(self):
        cloned_model = Policy(self.environment, self.logger)
        cloned_model.set_flat_params(self.get_flat_params())
        return cloned_model

    def log(self, *strings, writeToFile=False, debug_channel="Generic"):
        print(strings)
        if self.logger!=None:
            self.logger.print(strings, writeToFile=writeToFile, debug_channel=debug_channel)

class Value:
    def __init__(self, environment, logger, value_lr, debug=False):
        
        self.logger = logger

        self.input_shape = environment.get_observation_shape()
        self.output_shape = 1
        self.environment = environment
        self.value_lr = value_lr
        self.debug=debug

        #Determine if we're going to use a Convolutional NN (Image analysis required)
        self.convolutional = environment.type in ["atari"]

        self.log("input_shape: ", self.input_shape,", output_shape: ",self.output_shape, ", convolutional: ",self.convolutional, writeToFile=True, debug_channel="model")
        self.model = keras.Sequential()
        if not self.convolutional:
            self.model.add(keras.layers.Dense(64, activation="relu", input_shape=self.input_shape))
            self.model.add(keras.layers.Dense(64, activation="relu"))
            self.model.add(keras.layers.Dense(self.output_shape))
            adam = keras.optimizers.Adam(learning_rate=value_lr)
            self.model.compile(loss="mean_squared_error", optimizer=adam)
        else:
            self.model.add(keras.layers.Conv2D(10, (3, 3), input_shape=self.input_shape, activation='relu'))
            self.model.add(keras.layers.MaxPooling2D((3, 3)))
            self.model.add(keras.layers.Conv2D(5, (3, 3), activation='relu'))
            self.model.add(keras.layers.MaxPooling2D((3, 3)))
            self.model.add(keras.layers.Flatten())
            self.model.add(keras.layers.Dense(self.output_shape))          
            adam = keras.optimizers.Adam(learning_rate=value_lr)
            self.model.compile(loss="mean_squared_error", optimizer=adam)

		#crea la lista delle shape di ogni strato della rete neurale
        self.shape_list = [layer.shape.as_list() for layer in self.model.trainable_variables]

		#calcola il numero complessivo di parametri della rete neurale
        self.number_of_parameters = np.sum([np.prod(shape) for shape in self.shape_list])
        self.logger.print("number_of_parameters: ",self.number_of_parameters, writeToFile=True, debug_channel="model")
        stringlist=[]
        self.model.summary(print_fn=lambda x: stringlist.append(x))
        model_summary = "\n".join(stringlist)
        self.logger.print(model_summary,writeToFile=True,debug_channel="model_summary")
    
    def __call__(self, observation):
        if self.convolutional:
            observation = np.array(observation)
            #Add an axis if the image has only 1 channel (grayscale) because the shape has to be like 
            # (batch size, img_height, img_width, channels) but if the image is grayscale, the shape is
            # (batch_size, img_height, img_width)
            if self.input_shape[2]==1:
                observation = observation.reshape(observation.shape[0], observation.shape[1], observation.shape[2], 1)
            if self.input_shape[2]==3:
                observation = observation.reshape(observation.shape[0], observation.shape[1], observation.shape[2], 3)

            observation = tf.cast(observation,'float64')
        
        val = self.model.predict(np.array(observation)).flatten()
        return val
    
    def fit(self, observations, discounted_rewards, epochs=5, verbose=0):
        if self.convolutional:
            observations = np.array(observations)
            #Add an axis if the image has only 1 channel (grayscale) because the shape has to be like 
            # (batch size, img_height, img_width, channels) but if the image is grayscale, the shape is
            # (batch_size, img_height, img_width)
            if self.input_shape[2]==1:
                observations = observations.reshape(observations.shape[0], observations.shape[1], observations.shape[2], 1)
            if self.input_shape[2]==3:
                observations = observations.reshape(observations.shape[0], observations.shape[1], observations.shape[2], 3)
        return self.model.fit(np.array(observations),np.array(discounted_rewards),epochs=epochs,verbose=verbose)

    def get_flat_params(self):
	    return tf.concat([tf.reshape(v, [-1]) for v in self.model.trainable_variables], axis=0).numpy()

    def set_flat_params(self, flat_params):

        current_layer=0
        assigned_parameters = 0
        for shape in self.shape_list: 
            #per ogni strato
            #calcola il numero di parametri di quello strato
            layer_size = np.prod(shape)
            #estrai i parametri relativi a quello strato dal vettore theta
            theta = tf.reshape(flat_params[assigned_parameters:assigned_parameters + layer_size], shape)
            #assegna tali parametri allo strato corrispondente nella rete neurale model
            self.model.trainable_variables[current_layer].assign(theta)

            #accumula il numero di parametri assegnati
            assigned_parameters += layer_size
            current_layer+=1

        #verifica con una assertion che il numero di parametri assegnati sia corretto
        assert assigned_parameters == self.number_of_parameters
    

    def save_model_weights(self,filename):
        self.model.save_weights(filename + ".h5")

    def load_model_weights(self, filename):
        self.model.load_weights(filename + ".h5")
        
    def clone(self):
        cloned_model = Value(self.environment, self.logger, self.value_lr)
        cloned_model.set_flat_params(self.get_flat_params())
        return cloned_model

    def log(self, *strings, writeToFile=False, debug_channel="Generic"):
        if self.logger!=None:
            self.logger.print(strings, writeToFile=writeToFile, debug_channel=debug_channel)

if __name__ =="__main__":
    import numpy
    import sys
    from TRPOAgent import *
    #numpy.set_printoptions(threshold=sys.maxsize)

    env_name = "MsPacman-ram-v0"

    logger = Logger(name=env_name, debugChannels=["model","model_summary"])
    env = Environment(env_name, logger, True)
    agent = TRPOAgent(env,logger)