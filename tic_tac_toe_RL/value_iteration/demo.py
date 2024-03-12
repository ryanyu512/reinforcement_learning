from game import *
from agent import *

HUMAN = 1
AGENT = 2

#initialise agent
agent = Agent(epsilon = 1e-10)

#training
agent.iterate(who_first = AGENT)

#initialise game
game = TIC_TAC_TOE(agent = agent)

#start playing
game.human_vs_agent(who_first = AGENT)
