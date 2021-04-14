import sys
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units

import csv



class BayesAgent(base_agent.BaseAgent):
    def __init__(self):
        super(BayesAgent, self).__init__()
        self.games=0

    def reset(self):
        self.games += 1
        if self.games > 5:
            sys.exit("games complete")
            

    def step(self, obs):
        #1961 is for the total number of differant units in sc2
        enemy_state = [0]*1961
        enemy_units = self.get_units_by_enemy(obs)
        if len(enemy_units) != 0:
            for unit in enemy_units:
                unit_id = unit[0]
                enemy_state[unit_id] += 1
            with open('opponent_data.csv',mode = 'a') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=',')
                csv_writer.writerow(enemy_state + ["13"])
        return actions.RAW_FUNCTIONS.no_op()

    def get_units_by_enemy(self, obs):
        return [unit for unit in obs.observation.raw_units
                if unit.alliance == features.PlayerRelative.ENEMY]