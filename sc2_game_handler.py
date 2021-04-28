from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features

from h_agent import honoursAgent

import sys
from absl import flags
FLAGS = flags.FLAGS
FLAGS(sys.argv)


def main():
  agent = honoursAgent()
  try:
    while True:
      with sc2_env.SC2Env(
          map_name="Simple64",
          players=[sc2_env.Agent(sc2_env.Race.zerg),
                   sc2_env.Bot(sc2_env.Race.terran,
                               sc2_env.Difficulty.very_easy)],
          agent_interface_format=features.AgentInterfaceFormat(
              action_space=actions.ActionSpace.RAW,
              use_raw_units=True,
              raw_resolution=64),
          step_mul=16,
          game_steps_per_episode=15000,) as env:
          
        agent.setup(env.observation_spec(), env.action_spec())
        
        timesteps = env.reset()
        agent.reset()
        
        while True:
          step_actions = [agent.step(timesteps[0])]
          if timesteps[0].last():
            break
          timesteps = env.step(step_actions)
      
  except KeyboardInterrupt:
    pass
  
main()