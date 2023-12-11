import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def plotLearning(x, scores, epsilons, filename, window = 20, lines = None):
    fig = plt.figure()
    ax  = fig.add_subplot(111,label="1")
    ax2 = fig.add_subplot(111,label="2", frame_on = False)
    
    ax.plot(x, epsilons, color = "C0")
    ax.set_xlabel("Game", color = "C0")
    ax.set_ylabel("Epsilon", color = "C0") 
    ax.tick_params(axis ='x', colors= "C0")
    ax.tick_params(axis ='y', colors = "C0")
    
    N = len(scores)
    running_avg = np.empty(N)
    
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
        
    ax2.scatter(x, running_avg, color = "C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color= "C1")
    ax2.yaxis.set_label_position("right")
    ax2.tick_params(axis='y',colors="C1")
    
    if lines is not None:
        for line in lines:
            plt.axvline(x = line)
            
    plt.savefig(filename)
    
def plotLearningNoEpilson(scores, filename, x=None, window=5):
    N = len(scores)
    running_avg = np.empty(N)
    
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
        
    if x is None:
        x = [i for i in range(N)]
        
    plt.ylabel('Score')
    plt.xlabel('Game')
    plt.plot(x, running_avg)
    plt.savefig(filename)
    
def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):
    
    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)