from pysc2.agents import base_agent
from pysc2.lib import actions, features, units

import numpy as np
import random
import time

from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Activation, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
from tensormod import ModifiedTensorBoard


action_space = []


class honoursAgent(base_agent.BaseAgent):
    def __init__(self):
        super(honoursAgent, self).__init__()


        self.model = self.create_model()

        self.target_model = self.create_model()
        self.weights = self.model.get_weights()

        self.target_model.set_weights(self.weights)

    def step(self, obs):
        super(honoursAgent, self).step(obs)


        self.minerals = obs.observation.player.minerals

        return self.train_drone(obs)

    def no_op(self,obs):
        return actions.FUNCTIONS.no_op()

    def get_units_by_type(self,obs, unit_type):
        return [unit for unit in obs.observation.raw_units
            if unit.unit_type == unit_type 
            and unit.alliance == features.PlayerRelative.SELF]

    def select_larva(self, obs):
        larvae = self.get_units_by_type(obs,units.Zerg.Larva)
        if len(larvae) > 0:
            larva = random.choice(larvae)
        return larva

    def train_drone(self, obs):
        if self.minerals >= 50:
            larva = self.select_larva(obs)
            if len(larva) > 0:
                    return actions.RAW_FUNCTIONS.Train_Drone_quick("now",larva.tag)
        return actions.RAW_FUNCTIONS.no_op()



    def create_model(self):
        ## create NN model
        nn_input = Input(shape=(64, 64))
        nn_layer1 = Dense(100, activation="relu")(nn_input)
        nn_layer2 = Dense(50, activation="relu", name="dense_end")(nn_layer1)

        nn_flatten = Flatten()(nn_layer2)

        ## create Conv model
        conv_input = Input(shape=(64, 64, 1))
        conv_layer_input = Conv2D(64, (3, 3), activation="relu")(conv_input)
        conv_layer_input_pool = MaxPooling2D(pool_size=(3, 3))(conv_layer_input)

        conv_layer1 = Conv2D(64, (3, 3), activation="relu")(conv_layer_input_pool)
        conv_layer1_pool = MaxPooling2D(pool_size=(3, 3))(conv_layer1)

        conv_layer2 = Conv2D(64, (3, 3), activation="relu")(conv_layer1_pool)
        conv_layer2_pool = MaxPooling2D(pool_size=(3, 3), name="conv_end")(conv_layer2)

        conv_flatten = Flatten()(conv_layer2_pool)

        ## merge models
        concatenated = concatenate([nn_flatten, conv_flatten])

        ## model output
        merged_layer = Dense(80, activation="relu")(concatenated)
        out = Dense((len(action_space)), activation="linear")(merged_layer)

        ##model construction + compile
        merged_model = Model([nn_input, conv_input], out)
        merged_model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
        return merged_model


