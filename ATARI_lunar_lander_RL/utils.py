import matplotlib.pyplot as plt
import numpy as np
import gym

def plotLearning(x, scores, filename, epsilons = None, window = 20, lines = None):
    fig = plt.figure()
    
    N = len(scores)
    running_avg = np.empty(N)
    
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
    
    ax = fig.add_subplot(111,label="2", frame_on = False)
    ax.scatter(x, running_avg, color = "C1")
    ax.axes.get_xaxis().set_visible(False)
    ax.yaxis.tick_right()
    ax.set_ylabel('Score', color= "C1")
    ax.yaxis.set_label_position("right")
    ax.tick_params(axis='y',colors="C1")
            
    plt.savefig(filename)