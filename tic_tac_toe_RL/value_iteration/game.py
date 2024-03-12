import os
import numpy as np

HUMAN = 1
AGENT = 2

class TIC_TAC_TOE():
    def __init__(self, agent, n_grid = 9):
        self.agent = agent
        self.agent.load_policy()
        self.n_grid = n_grid
        self.env    = None

    def initialise(self):
        self.env = np.zeros((self.n_grid,))

    def who_win(self):

        '''
        0 1 2
        3 4 5
        6 7 8
        '''

        #check vertical and horizontal line
        for i in range(3):
            if self.env[i*3] == self.env[i*3 + 1] and self.env[i*3] == self.env[i*3 + 2] and self.env[i*3] > 0:
                return self.env[i*3]
            if self.env[i] == self.env[i + 3] and self.env[i] == self.env[i + 6]  and self.env[i] > 0:
                return self.env[i]
        
        #check diagonal
        if self.env[0] == self.env[4] and self.env[0] == self.env[8]  and self.env[0] > 0:
            return self.env[0]
        if self.env[2] == self.env[4] and self.env[2] == self.env[6]  and self.env[2] > 0:
            return self.env[2]
        
        return None

    def is_end_game(self):

        env = list(self.env)
        if env.count(0) == 0:
            print(f"no winner!")
            return True
        
        winner = self.who_win()
        if winner is not None:
            print(f"winner: {'human!' if winner == 1 else 'computer!'}")
            return True
    
        return False
    
    def human_vs_agent(self, who_first):

        #initialise game
        self.initialise()

        #decide who first
        # next_turn = np.random.binomial(1, 0.5)
        next_turn = who_first
        print(f"{'human first!' if next_turn == HUMAN else 'agent first!'}")

        cnt = 0
        while True:
            
            if self.is_end_game():
                break

            s = tuple(self.env)
            if next_turn == HUMAN:
                print("enter (0 - 8) to represent your location")
                print("0 1 2\n3 4 5\n6 7 8")
                human_input = int(input())
                self.env[human_input] = HUMAN

                next_turn = AGENT
                print(f"human takes {human_input}")
            else:
                agent_input = self.agent.step(s)
                self.env[agent_input] = AGENT

                next_turn = HUMAN
                print(f"agent takes {agent_input}")

            print(f"==== round: {cnt} ====")
            c = 0
            for i in range(self.env.shape[0]):
                if c < 3:
                    print(f"{self.env[i]}", end = " ")
                c += 1

                if c >= 3:
                    c = 0
                    print()

            cnt += 1





