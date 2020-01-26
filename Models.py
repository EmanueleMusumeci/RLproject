import numpy as np
#from tensorflow.keras import layers, optimizers, losses, models
#from tensorflow.keras.layers import Dense, Sequential
#sets to float64 to avoid compatibility issues between numpy 
# (whose default is float64) and keras(whose default is float32)

from Environment import Environment
from Logger import Logger

import tensorflow as tf
from tensorflow import keras as keras

#    #TODO:Get NN shape from CustomEnvironmentRegister
#DONE#TODO:Use Atari preprocessing wrapper to preprocess the image
#    #TODO:Make both models subclasses of a NNEstimator class, to collect common methods

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
            self.model.add(keras.layers.Dense(64, input_shape=self.input_shape, activation="relu"))
            self.model.add(keras.layers.Dense(64, activation="relu"))
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
        self.log("number_of_parameters: ",self.number_of_parameters, writeToFile=True, debug_channel="model")
        print(self.model.summary())
    
    def __call__(self, observation, tensor=False):
        if self.convolutional:
            #self.log("observation.shape: ",observation.shape,writeToFile=True, debug_channel="model")
            #Add an axis if the image has only 1 channel (grayscale) because the shape has to be like 
            # (batch size, img_height, img_width, channels) but if the image is grayscale, the shape is
            # (batch_size, img_height, img_width)
            if self.input_shape[2]==1:
                observation = observation.reshape(observation.shape[0], observation.shape[1], observation.shape[2], 1)
        
        #self.log("shape fed to the neural network: ",observation.shape,writeToFile=True, debug_channel="model")
        #if tensor:

        if self.convolutional: #observation.astype('uint8')
            observation = tf.cast(observation,'float64')
        #self.log("Observation.dtype: ",observation.dtype, debug_channel="model")
        
        return self.model(observation)
        #else:
            #return self.model.predict(observation)

    def get_flat_params(self, numpy=True):
        params = tf.concat([tf.reshape(v, [-1]) for v in self.model.trainable_variables], axis=0)
        if numpy:
            return params.numpy()
        else:
            return params

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

    #Computes the gradients wrt the weights of the neural network
    #args is a dictionary containing all remaining arguments to be passed to the function as:
    # ({"parameter_name":parameter_value}). The function is supposed to unpack them
    def get_flat_gradients(self, function, args=None):
        #create a tensorflow array out of the numpy one containing the nn parameters
        trainable_variables = self.model.trainable_variables
        #observation = tf.Variable(observation)
#        self.log("trainable variables:\n",self.model.trainable_variables, writeToFile=True, debug_channel="model")

        with tf.GradientTape() as t:
                f= function()
        
        gradient = t.gradient(f, trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO) 
            #UnconnectedGradients.ZERO serve a evitare che ti restituisca None quando il gradiente è zero
 #       self.log("gradient:\n",gradient, writeToFile=True, debug_channel="model")
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
            self.model.add(keras.layers.Dense(64, input_shape=self.input_shape, activation="relu"))
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
        self.log("number_of_parameters: ",self.number_of_parameters, writeToFile=True, debug_channel="model")
        print(self.model.summary())
    
    def __call__(self, observation, tensor=False):
        if self.convolutional:
            #self.log("observation.shape: ",observation.shape,writeToFile=True, debug_channel="model")
            #Add an axis if the image has only 1 channel (grayscale) because the shape has to be like 
            # (batch size, img_height, img_width, channels) but if the image is grayscale, the shape is
            # (batch_size, img_height, img_width)
            if self.input_shape[2]==1:
                observation = observation.reshape(observation.shape[0], observation.shape[1], observation.shape[2], 1)
            if self.input_shape[2]==3:
                observation = observation.reshape(observation.shape[0], observation.shape[1], observation.shape[2], 3)
        
        #print("A")

        #self.log("shape fed to the neural network: ",observation.shape,writeToFile=True, debug_channel="model")
        #if tensor:
        if self.convolutional: #observation.astype('uint8')
            observation = tf.cast(observation,'float64')

        #print("B")
        #print("Obs shape: ", observation.shape, ", Known shape: ", self.input_shape)
        val = self.model(observation).numpy().flatten()
        #print("VALUE PREDICTION: ", val)
        return val
        #else:
            #return self.model.predict(observation)
    
    def fit(self, observations, discounted_rewards, epochs=5, verbose=0):
        if(len(observations.shape)==3):    
            observations = observations.reshape(observations.shape[0], observations.shape[1], observations.shape[2], self.input_shape[2])
        return self.model.fit(observations,discounted_rewards,epochs=epochs, verbose=verbose)

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
    #numpy.set_printoptions(threshold=sys.maxsize)

    env_name = "MountainCar-v0"

    logger = Logger(name=env_name,log_directory="TRPO_project/Testing/Model")
    env = Environment(env_name, logger, True,True)
    policy = Policy(env, logger)
    value = Value(env, logger, 1e-3)
    policy_params = policy.get_flat_params()
    value_params = value.get_flat_params()
    print("policy_params:\n",policy_params)
    print("value_params:\n",value_params)
    
    if False:
        '''
        zeroed_policy_params = np.zeros_like(policy_params)
        policy.set_flat_params(zeroed_policy_params)
        zeroed_policy_params = policy.get_flat_params()
        print("zeroed policy_params:\n",zeroed_policy_params, ", len: ",len(zeroed_policy_params))

        zeroed_value_params = np.zeros_like(value_params)
        value.set_flat_params(zeroed_value_params)
        zeroed_value_params = value.get_flat_params()
        print("zeroed value_params:\n",zeroed_value_params, ", len: ",len(zeroed_value_params))

        x = np.arange(1,len(policy_params)+1,dtype=np.float64)
        print("x: ", x)
        x = tf.convert_to_tensor(x)
        print("x(tensor): ", x)
        with tf.GradientTape() as t:
            t.watch(x)
            function = tf.pow(x,2)
            print(function)
        grad = t.gradient(function, x)
        print(grad)
        '''

    x = np.arange(1,len(policy_params)+1,dtype=np.float64)
    print("x: ", x)
    policy.set_flat_params(x)
    policy_params = policy.get_flat_params()
    print("policy_params before gradient: ", policy_params)
    
    def fn1(x):
        return tf.pow(x,2)
    
    def fn2(x,args):
        y= args["y"]
        return tf.pow(x,2)*y
    function = fn1
    #gradients = policy.get_flat_gradients(function, args={"y":1})
    gradients = policy.get_flat_gradients(function)
    print("gradients: ", gradients)

    if False:
        '''
        x = np.arange(1,len(value_params)+1,dtype=np.float64)
        print("x: ", x)
        value.set_flat_params(x)
        value_params = value.get_flat_params()
        print("value_params before gradient: ", value_params)

        function = fn
        gradients = value.get_flat_gradients(function)
        print("gradients: ", gradients)
        '''