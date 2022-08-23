# analysis of chemotaxer for Neha's CEWiT poster
# haily merritt
# time of development: spring 2022

# import dependencies
import pickle
import matplotlib.pyplot as plt
import numpy as np
import netneuro
import agent
import CTRNN

# set Parameters
step_size = 0.01
locations = [[50,30,90],[20,60]]
n_agent = 3
duration = 2500
n_iter = 10
sizes = [2,4,6,8]
#sizes = [2,4,6,8,10]

# define function to run task and plot neural output
def runTask(size,param):
    final_dist = np.zeros((n_agent,n_iter))
    neural = np.zeros((n_agent,size,duration,n_iter))
    for iter in range(n_iter):
        for a in range(n_agent):
            # create agent
            i = agent.Agent(size,step_size)
            i.locateAgent(locations[0])
            i.setParameters(param[a][iter])

            # create food
            food = agent.Food()
            food.locateFood(locations[1])

            # initialize time
            time = 0

            while time < duration:

                 # update sensors
                 i.sense(food)
                 # move agent
                 i.tax(duration)
                 # record neural activity
                 neural[a,:,time,iter] = i.network.outputs
                 # increase time
                 time += 1

            final_dist[a,iter] = i.distance(food)
    ind = 1
    time_hist = np.array(range(duration))
    for a in range(n_agent):
        for i in range(n_iter):
            for s in range(size):
                plt.subplot(n_agent,n_iter,ind)
                plt.plot(time_hist,neural[a,s,:,i],'c',alpha=0.75)
                plt.plot(time_hist,neural[a,s,:,i],'m',alpha=0.75)
            ind += 1
    fname = "neuralOutputs_" + str(size) + ".png"
    plt.savefig(fname)
    #plt.show()
    return final_dist

# import evolved parameters
two = pickle.load( open("top3Genotypes_2.p", "rb") )
four = pickle.load( open("top3Genotypes_4.p", "rb") )
six = pickle.load( open("top3Genotypes_6.p", "rb") )
eight = pickle.load( open("top3Genotypes_8.p", "rb") )
#ten = pickle.load( open("top3Genotypes_10.p", "rb") )

# run all sizes and plot neural output
param = [two,four,six,eight]
dists = np.zeros((n_agent,n_iter,len(sizes)))
for size in range(len(sizes)):
    dists[:,:,size] = runTask(sizes[size],param[size])

plt.violinplot(np.reshape(dists,(n_agent*n_iter,len(sizes))))
plt.show()
