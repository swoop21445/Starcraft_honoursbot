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




class honoursAgent(base_agent.BaseAgent):
    def __init__(self):
        super(honoursAgent, self).__init__()

        self.action_space = []
        self.main_base = []

        self.model = self.create_model()

        self.target_model = self.create_model()
        self.weights = self.model.get_weights()

        self.target_model.set_weights(self.weights)

    def step(self, obs):
        super(honoursAgent, self).step(obs)

        self.minerals = obs.observation.player.minerals

        hatchery = self.get_units_by_type(obs,units.Zerg.Hatchery)
        if len(hatchery) > 0:
            self.main_base_left = (hatchery[0].x < 32)

        return random.choice([self.train_drone(obs), self.train_overlord(obs), self.build_pool(obs), self.train_zergling(obs), self.attack(obs)])

    def no_op(self,obs):
        return actions.RAW_FUNCTIONS.no_op()

    def get_units_by_type(self,obs, unit_type):
        return [unit for unit in obs.observation.raw_units
            if unit.unit_type == unit_type 
            and unit.alliance == features.PlayerRelative.SELF]

    def get_distances(self, obs, unit_list, target_xy):
        units_xy = [(unit.x, unit.y) for unit in unit_list]
        return np.linalg.norm(np.array(units_xy) - np.array(target_xy), axis=1)
    
    def select_larva(self, obs):
        larva = self.get_units_by_type(obs,units.Zerg.Larva)
        if len(larva) > 0:
            larva = random.choice(larva)
        return larva
    
    def select_builder(self, obs, drones, location):
        distance_to_target = self.get_distances(obs, drones, location)
        drone = drones[np.argmin(distance_to_target)]
        return drone

    def train_overlord(self,obs):
        if self.minerals >= 100:
            larva = self.select_larva(obs)
            if len(larva) > 0:
                return actions.RAW_FUNCTIONS.Train_Overlord_quick("now",larva.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def train_drone(self, obs):
        if self.minerals >= 50:
            larva = self.select_larva(obs)
            if len(larva) > 0:
                    return actions.RAW_FUNCTIONS.Train_Drone_quick("now",larva.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def build_pool(self,obs):
        if self.minerals >= 200:
            target_location = (22,21) if self.main_base_left else (35,42)
            drones = self.get_units_by_type(obs,units.Zerg.Drone)
            if len(drones) > 1:
                drone = self.select_builder(obs, drones, target_location)
            return actions.RAW_FUNCTIONS.Build_SpawningPool_pt("now", drone.tag, target_location)
        return actions.RAW_FUNCTIONS.no_op()

    def train_zergling(self,obs):
        if self.minerals >= 50:
            if len(self.get_units_by_type(obs,units.Zerg.SpawningPool)) > 0:
                larva = self.select_larva(obs)
                if len(larva) > 0:
                    return actions.RAW_FUNCTIONS.Train_Zergling_quick("now", larva.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def attack(self, obs):
        zerglings = self.get_units_by_type(obs,units.Zerg.Zergling)
        if len(zerglings)> 0:
            target_location = (35,42) if self.main_base_left else (22,21)
            distances = self.get_distances(obs,zerglings,target_location)
            zergling = zerglings[np.argmax(distances)]
            return actions.RAW_FUNCTIONS.Attack_pt("now", zergling.tag, (target_location[0],target_location[1]))
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
        out = Dense((len(self.action_space)), activation="linear")(merged_layer)

        ##model construction + compile
        merged_model = Model([nn_input, conv_input], out)
        merged_model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
        return merged_model


