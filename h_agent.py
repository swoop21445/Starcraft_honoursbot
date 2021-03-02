from pysc2.agents import base_agent
from pysc2.lib import actions, features, units

import numpy as np
import random
import time
from collections import deque
from tqdm import tqdm


import tensorflow as tf

from keras.callbacks import TensorBoard
from keras.utils import normalize
from keras.layers.merge import concatenate
from keras.models import Model, load_model, save_model
from keras.layers import Dense, Input, Activation, Flatten, Conv2D, MaxPooling2D
from tensormod import ModifiedTensorBoard

max_stored_states = 50_000
min_stored_states = 1000  # changed for test ease 10000 = normal
minibatch_size = 50
update_value = 1  # changed for testing ease 5 = normal
model_name = "extractors"
terminate_value = 20_000


class honoursAgent(base_agent.BaseAgent):
    def __init__(self):
        super(honoursAgent, self).__init__()

        self.tensorboard = ModifiedTensorBoard(
            log_dir="logs/{}-{}".format(model_name, int(time.time())), model_name=model_name)

        self.stored_states = deque(maxlen=max_stored_states)
        self.target_update_counter = -1
        self.numb_game = 0

        state_len = ["minerals",
                     "gas",
                     "supply",
                     "supply_cap",
                     "army_supply",
                     "worker_supply",
                     "idle_workers",
                     "larva_count",
                     "queens_count",
                     "queen_energy",
                     "game_loops "]

        self.nn_input_shape = len(state_len)

        action_space = ["self.no_op(obs)",
                        "self.train_drone(obs)",
                        "self.build_pool(obs)",
                        "self.train_overlord(obs)",
                        "self.train_zergling(obs)",
                        "self.attack(obs)",
                        "attack_expansion(obs",
                        "self.train_queen(obs)",
                        "self.queen_inject(obs)",
                        "build_vespene_extractor"]

        self.model_output_len = len(action_space)

        try:
            self.model = Model.load_model("models/" + model_name)
        except:
            self.model = self.create_model()
        self.target_model = self.create_model()
        self.weights = self.model.get_weights()

        self.target_model.set_weights(self.weights)

        self.epsilon = 0.99
        # for speed of training epsilon decay is reduced should be 0.9999975
        self.epsilon_decay = 0.9999975
        self.discount = 0.99

    def reset(self):
        super(honoursAgent, self).reset()
        self.main_base = []
        self.state = "begin"

        self.target_update_counter += 1
        if self.numb_game != 0:
            print("Game reward = " + str(self.reward_debug))
        self.numb_game += 1
        print("Game Number " + str(self.numb_game) + " Starting....")

        if self.target_update_counter > update_value:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
        self.train()

    def step(self, obs):
        super(honoursAgent, self).step(obs)

        self.build_state(obs)
        reward = obs.observation.score_cumulative[0]
        # remove in full release
        self.reward_debug = reward
        units_map = self.populate_map(obs)
        numerical_state = self.build_state(obs)

        penalty = self.ovy_overproduction_penalty(obs)
        reward = reward - penalty

        place_holder_actions = 0

        # model prediction or random action based on epsilon
        action_index = self.choice_maker(obs, numerical_state, units_map)

        if self.state == "begin":
            self.state = [numerical_state, units_map, place_holder_actions, reward, [
                numerical_state, units_map, place_holder_actions, reward]]
            self.old_state = [numerical_state, units_map,
                              place_holder_actions, reward, self.state]
        else:
            self.state = [numerical_state, units_map,
                          action_index, reward, self.old_state]

        self.hatchery = self.get_units_by_type(obs, units.Zerg.Hatchery)
        if len(self.hatchery) > 0:
            self.main_base_left = (self.hatchery[0].x < 32)

        self.update_state_mem(self.state, reward)
        self.old_state = self.state

        action = self.action_number_matcher(obs, action_index)

        if self.game_loops > terminate_value:
            action = self.civil_war(obs)

        return action

    def choice_maker(self, obs, numerical_state, units_map):
        if random.random() > self.epsilon:
            choice = self.model.predict((numerical_state, units_map))
            choice = np.argmax(choice)
        else:
            choice = random.randint(0, (self.model_output_len - 1))
        self.epsilon = self.epsilon * self.epsilon_decay
        return choice

    def action_number_matcher(self, obs, action_number):
        if action_number == 0:
            return self.no_op(obs)
        elif action_number == 1:
            return self.train_drone(obs)
        elif action_number == 2:
            return self.build_pool(obs)
        elif action_number == 3:
            return self.train_overlord(obs)
        elif action_number == 4:
            return self.train_zergling(obs)
        elif action_number == 5:
            return self.attack(obs)
        elif action_number == 6:
            return self.attack_expansion(obs)
        elif action_number == 7:
            return self.train_queen(obs)
        elif action_number == 8:
            return self.queen_inject(obs)
        elif action_number == 9:
            return self.build_vespene_extractor(obs)
        else:
            raise Exception("Action scope error")

    def no_op(self, obs):
        return actions.RAW_FUNCTIONS.no_op()

    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.alliance == features.PlayerRelative.SELF]

    def get_units_by_type_neutral(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type]

    # script taken from here https://gist.github.com/skjb/6764df2e1bba282730a893f38b8d449e#file-learning_agent_step1e-py
    def get_distances(self, obs, unit_list, target_xy):
        units_xy = [(unit.x, unit.y) for unit in unit_list]
        return np.linalg.norm(np.array(units_xy) - np.array(target_xy), axis=1)

    def select_larva(self, obs):
        larva = self.get_units_by_type(obs, units.Zerg.Larva)
        if len(larva) > 0:
            larva = random.choice(larva)
        return larva

    def select_builder(self, obs, drones, location):
        distance_to_target = self.get_distances(obs, drones, location)
        drone = drones[np.argmin(distance_to_target)]
        return drone

    def train_overlord(self, obs):
        if self.minerals >= 100:
            larva = self.select_larva(obs)
            if len(larva) > 0:
                return actions.RAW_FUNCTIONS.Train_Overlord_quick("now", larva.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def train_drone(self, obs):
        if self.minerals >= 50:
            larva = self.select_larva(obs)
            if len(larva) > 0:
                return actions.RAW_FUNCTIONS.Train_Drone_quick("now", larva.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def build_pool(self, obs):
        if self.minerals >= 200:
            target_location = (22, 21) if self.main_base_left else (35, 42)
            drones = self.get_units_by_type(obs, units.Zerg.Drone)
            if len(drones) > 0:
                drone = self.select_builder(obs, drones, target_location)
                return actions.RAW_FUNCTIONS.Build_SpawningPool_pt("now", drone.tag, target_location)
        return actions.RAW_FUNCTIONS.no_op()

    def train_zergling(self, obs):
        if self.minerals >= 50:
            if len(self.get_units_by_type(obs, units.Zerg.SpawningPool)) > 0:
                larva = self.select_larva(obs)
                if len(larva) > 0:
                    return actions.RAW_FUNCTIONS.Train_Zergling_quick("now", larva.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def attack(self, obs):
        zerglings = self.get_units_by_type(obs, units.Zerg.Zergling)
        if len(zerglings) > 0:
            target_location = (35, 42) if self.main_base_left else (22, 21)
            zerglings = [unit.tag for unit in zerglings]
            return actions.RAW_FUNCTIONS.Attack_pt("queued", zerglings, (target_location[0], target_location[1]))
        return actions.RAW_FUNCTIONS.no_op()

    def attack_expansion(self, obs):
        zerglings = self.get_units_by_type(obs, units.Zerg.Zergling)
        if len(zerglings) > 0:
            target_location = (15, 50) if self.main_base_left else (40, 15)
            zerglings = [unit.tag for unit in zerglings]
            return actions.RAW_FUNCTIONS.Attack_pt("queued", zerglings, (target_location[0], target_location[1]))
        return actions.RAW_FUNCTIONS.no_op()

# use queue checks when multiple hatcheries are implemented
    def train_queen(self, obs):
        if self.minerals >= 150:
            hatcheries = self.get_units_by_type(obs, units.Zerg.Hatchery)
            if len(hatcheries) > 0:
                return actions.RAW_FUNCTIONS.Train_Queen_quick("now", hatcheries[0].tag)
        return actions.RAW_FUNCTIONS.no_op()

    def queen_inject(self, obs):
        if self.queen_energy > 25:
            try:
                return actions.RAW_FUNCTIONS.Effect_InjectLarva_unit("now", self.queen, self.hatchery[0].tag)
            except:
                return actions.RAW_FUNCTIONS.no_op()
        return actions.RAW_FUNCTIONS.no_op()

    def build_vespene_extractor(self, obs):
        if self.minerals > 25:
            drones = self.get_units_by_type(obs, units.Zerg.Drone)
            hatchery = self.get_units_by_type(obs, units.Zerg.Hatchery)
            if len(drones) > 0 and len(hatchery) > 0:
                geysers = self.get_units_by_type_neutral(obs, units.Neutral.VespeneGeyser)
                
                distance = self.get_distances(obs, geysers, (hatchery[0].x, hatchery[0].y))
                extractors_num = len(self.get_units_by_type(obs,units.Zerg.Extractor))
                geysers_sorted = np.sort(distance)
                # adjust for extractors thazt could have already been made
                if len(geysers_sorted) == extractors_num:
                    return actions.RAW_FUNCTIONS.no_op()
                target_geyser = geysers_sorted[extractors_num]
                target_geyser = np.where(distance == target_geyser)
                target_geyser = target_geyser[0][0]
                geyser = geysers[target_geyser]
                drone = self.select_builder(obs, drones, (geyser.x, geyser.y))
                return actions.RAW_FUNCTIONS.Build_Extractor_unit("now", drone.tag, geyser.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def get_queen_energy_status(self, obs):
        queens = self.get_units_by_type(obs, units.Zerg.Queen)
        if len(queens) > 0:
            queen_energy = [unit.energy for unit in queens]
            queen = np.argmax(queen_energy)
            queen_energy = queens[queen].energy
            return queen_energy, len(queens), queens[queen].tag
        return 0, 0, 0

    def civil_war(self, obs):
        hatchery = self.get_units_by_type(obs, units.Zerg.Hatchery)
        spawning_pool = self.get_units_by_type(obs, units.Zerg.SpawningPool)
        extractor = self.get_units_by_type(obs, units.Zerg.Extractor)
        if len(hatchery) > 0 or len(spawning_pool) > 0 or len(extractor) > 0:
            drones = self.get_units_by_type(obs, units.Zerg.Drone)
            queens = self.get_units_by_type(obs, units.Zerg.Queen)
            queen_tags = [unit.tag for unit in queens]
            if len(queen_tags) > 0:
                attacker_tags = queen_tags
            elif len(drones) > 0:
                drone_tags = [unit.tag for unit in drones]
                attacker_tags = drone_tags
            else:
                return actions.RAW_FUNCTIONS.no_op()
            if len(hatchery) > 0:
                return actions.RAW_FUNCTIONS.Attack_unit("now", attacker_tags, hatchery[0].tag)
            elif len(spawning_pool) > 0:
                return actions.RAW_FUNCTIONS.Attack_unit("now", attacker_tags, spawning_pool[0].tag)
            else:
                return actions.RAW_FUNCTIONS.Attack_unit("queued", attacker_tags, extractor[0].tag)
        return actions.RAW_FUNCTIONS.no_op()

    def build_state(self, obs):
        self.minerals = obs.observation.player.minerals
        self.gas = obs.observation.player.vespene
        self.supply = obs.observation.player.food_used
        self.supply_cap = obs.observation.player.food_cap
        self.army_supply = obs.observation.player.food_army
        self.worker_supply = obs.observation.player.food_workers
        self.idle_workers = obs.observation.player.idle_worker_count
        self.larva_count = obs.observation.player.larva_count
        self.queen_energy, self.queens_count, self.queen = self.get_queen_energy_status(
            obs)
        self.game_loops = obs.observation.game_loop
        state = (float(self.minerals), float(self.gas), float(self.supply), float(self.supply_cap), float(
            self.army_supply), float(self.worker_supply), float(self.idle_workers), float(self. larva_count), float(self.queens_count), float(self.queen_energy), float(self.game_loops))
        state = np.asarray(state)
        state = np.reshape(state, (1, self.nn_input_shape))
        return state

    def populate_map(self, obs):
        unit_map = np.zeros(shape=(1, 64, 64, 2))
        for unit in obs.observation.raw_units:
            if unit.alliance == features.PlayerRelative.SELF:
                unit_map[0][unit.x][unit.y][0] = 1
            else:
                unit_map[0][unit.x][unit.y][1] = 1
        unit_map = unit_map.astype(np.float)
        return unit_map

    def ovy_overproduction_penalty(self, obs):
        # solving overlord overproduction by removing score for overlords over 200 supply
        if self.supply_cap == 200:
            overlords = self.get_units_by_type(obs, units.Zerg.Overlord)
            num_overlords = len(overlords)
            # 25 overlords + 1 hatchery is just over 200 control (control produces supply)
            overproduction = num_overlords - 25
            # reward penalty
            penalty = (overproduction * 100)
            return penalty
        return 0

    def update_state_mem(self, state, reward):
        self.stored_states.append(state)
        # 160 is the longest posible action for zerg
        if len(self.stored_states) >= 160:
            past_state_index = len(self.stored_states) - 160
            # reward has the 4th position in the state list
            self.stored_states[past_state_index][3] = reward

    def create_model(self):
        # create NN model
        # numerical state input
        nn_input = Input(shape=(self.nn_input_shape,), name="nn_input")

        nn_layer1 = Dense(100, activation="relu")(nn_input)
        nn_layer2 = Dense(50, activation="relu", name="dense_end")(nn_layer1)

        nn_flatten = Flatten()(nn_layer2)

        # create Conv model
        # map input
        conv_input = Input(shape=(64, 64, 2), name="conv_input")

        conv_layer_1 = Conv2D(64, (3, 3), activation="relu")(conv_input)
        conv_pool_1 = MaxPooling2D(pool_size=(3, 3))(conv_layer_1)

        conv_layer2 = Conv2D(64, (3, 3), activation="relu")(conv_pool_1)
        conv_layer2_pool = MaxPooling2D(pool_size=(3, 3))(conv_layer2)

        conv_layer3 = Conv2D(64, (3, 3), activation="relu")(conv_layer2_pool)
        conv_layer3_pool = MaxPooling2D(
            pool_size=(3, 3), name="conv_end")(conv_layer3)

        conv_flatten = Flatten()(conv_layer3_pool)

        # merge models
        concatenated = concatenate([nn_flatten, conv_flatten])

        # model output
        merged_layer = Dense(80, activation="relu")(concatenated)
        out = Dense(self.model_output_len, activation="linear")(merged_layer)

        # model construction + compile
        merged_model = Model(
            inputs=[nn_input, conv_input], outputs=out, name="merged_model_1")
        merged_model.compile(loss="mse", optimizer="adam",
                             metrics=["accuracy"])

        merged_model.summary()

        return merged_model

    # code adapted from here: https://pythonprogramming.net/training-deep-q-learning-dqn-reinforcement-learning-python-tutorial/
    def train(self):
        if len(self.stored_states) < min_stored_states:
            return
        minibatch = random.sample(self.stored_states, minibatch_size)

        old_qs = []
        future_qs = []

        old_states = np.array([transition[4][0] for transition in minibatch])
        old_maps = np.array([transition[4][1] for transition in minibatch])
        for x in range(0, len(old_states)):
            q = self.model.predict((old_states[x], old_maps[x]))
            old_qs.append(q)

        new_states = np.array([transition[0] for transition in minibatch])
        new_maps = np.array([transition[1] for transition in minibatch])
        for x in range(0, len(new_states)):
            q = self.model.predict((new_states[x], new_maps[x]))
            future_qs.append(q)

        x = []
        y = []

        for index, (new_state, unit_map, action, reward, old_state) in enumerate(minibatch):

            max_future_q = np.max(future_qs[index])
            new_q = reward + self.discount * max_future_q

            current_qs = old_qs[index]
            current_qs[0][action] = new_q

            x.append((old_state[0], old_state[1]))
            y.append(current_qs)

        for index in range(0, len(y)):
            # callbacks = [self.tensorboard] removed
            self.model.fit(
                x[index], y[index], batch_size=minibatch_size, verbose=0, shuffle=False)
        ##self.model.save("models/" + model_name)
