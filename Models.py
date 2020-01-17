import tensorflow as tf
from tensorflow import keras as keras
#from tensorflow.keras import layers, optimizers, losses, models
#from tensorflow.keras.layers import Dense, Sequential

from Environment import Environment

##TODO:Get NN shape from CustomEnvironmentRegister
##TODO:Use Atari preprocessing wrapper to preprocess the image

class Policy:
    def __init__(self, environment):
        self.input_shape = environment.get_observation_shape()
        self.output_shape = environment.get_action_shape()
        self.model = keras.Sequential()
        if environment.type in ["box","mujoco"]:
            self.model.add(keras.layers.Dense(64, input_shape=self.input_shape, activation="relu"))
            self.model.add(keras.layers.Dense(64, activation="relu"))
            self.model.add(keras.layers.Dense(self.output_shape))
        elif environment.type=="atari":
            self.model.add(keras.layers.Dense(64,input_shape=self.input_shape, activation="relu"))
            self.model.add(keras.layers.Dense(64, activation="relu"))
            self.model.add(keras.layers.Dense(self.output_shape))
        else: print("Environment type not recognized")
    
    def __call__(self, observation):
        return self.model(observation)

    def get_flat_params(self):
        parameters = []
        for layer in self.model.layers:
            parameters.append(layer.get_weights())
        return parameters

    def get_flat_gradient(self, function):
        return []

class Value:
    def __init__(self, environment):
        self.input_shape = environment.get_observation_shape()
        self.output_shape = 1
        self.model = keras.Sequential()
        self.model.add(keras.layers.Dense(64,input_shape=self.input_shape,activation="relu"))
        self.model.add(keras.layers.Dense(1, activation="relu"))
    
    def __call__(self, observation):
        prediction = self.model.predict(observation)
        print(prediction)
        return prediction

