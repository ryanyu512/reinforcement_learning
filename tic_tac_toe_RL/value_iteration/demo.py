from game import *
from agent import *

HUMAN = 1
AGENT = 2

#initialise agent
agent = Agent(epsilon = 1e-10)

#training
agent.iterate(who_first = HUMAN)

#initialise game
game = TIC_TAC_TOE(agent = agent)
game.human_vs_agent(who_first = HUMAN)
