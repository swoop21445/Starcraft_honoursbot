from pysc2.agents import base_agent
from pysc2.lib import actions, features, units



class BayesAgent(base_agent.BaseAgent):
    def __init__(self):
        super(BayesAgent, self).__init__()

    def step(self, obs):
        return actions.RAW_FUNCTIONS.no_op()