# handling code for evolution of an adaptive chemotaxer
# haily merritt
# time of development: summer 2021

# THIS VERSION:
# - needs adjustment to accommodate a network size other than two
# - currently initializes agent and nutrient at one of the paired locations on
#   list derived from Agmon and Beer (2014), but can iterate through

# NEEDS:
# - kill agents if they cross boundary -> give fitness of 0
# - average fitness over time

### SET UP
# import packages
import random
import numpy as np
import matplotlib.pyplot as plt
import evalg as ea
import agent
import pandas as pd
import pickle
import multiprocessing as mp

### DEFINE PARAMETER VALUES
# for evolutionary algorithm
#size = 2                                # number of neurons in CTRNN
mutProb = 1                             # num of mutations per genotype
mutSize = 0.25                          # amount to mutate gene
popSize = 200                            # size of population
maxGen = 500                             # maximum number of generations to evolve
#genotypeLength = size + size + size**2  # length of genotype as determined by number of network parameters
#genotypeLength = int(size*2 + (size**2-size)/2)# assume size even, create bilateral symmetry
numClones = 2
# for chemotaxis
duration = 2500                         # duration of each trial
tau = 1                                 # time constant, subject to evalg
step_size = 0.01                        # integration size
envWidth = 100                          # width of environment
maxDist = np.sqrt(envWidth**2 + envWidth**2)    # maximum distance agent could be from nutrient in environment
locations = [[50,30,90],[20,60]]        # starting locations of agent and nutrient
#locations = [[[50,30,90],[20,60]], [[50,30,180],[20,60]], [[50,30,0],[20,60]], [[50,30,0],[40,80]], [[50,30,270],[45,80]], [[30,30,90],[10,75]], [[50,30,270],[10,80]], [[50,20,90],[15,80]], [[50,10,180],[40,80]], [[30,30,180],[10,80]], [[30,30,0],[10,90]]]
reps = len(locations)                   # number of times simulation should be repeated if using multipleRuns
nsave = 3 # number of genotypes to save at the end

### DEFINE FUNCTIONS
# take the second element for sort
def take_second(elem):
    return elem[1]

# evaluation function for chemotaxis
def evaluate(agent_params, locations):
    # evaluate fitness given performance on single trial

    #create agent
    individual = agent.Agent(size, step_size)
    individual.locateAgent(locations[0])
    individual.setParameters(agent_params)

    # create nutrient
    food = agent.Food()
    food.locateFood(locations[1])

    # initialize time
    time = 0

    while time < duration:

        # update agent's sensors
        individual.sense(food)
        # agent moves
        individual.tax(duration)
        #increase time
        time += 1

    return individual.distance(food) # gives a minimizing fitness function

    #maybe return network params

# we're not going to use this function for now
"""
def evaluateMulti(reps, agent_params, locations):
    # simulate multiple chemotaxing agents

    rep = 0
    dists = []


    while rep < reps:

        dists.append(evaluate(agent_params, locations[rep]))

        rep += 1

    return sum(dists)/reps
"""

def evolution(size):
    # initialize stats
    gen = 0
    genMin = []
    genMax = []
    genAvg = []
    genStd = []
    generation = []

    # create evolutionary algorithm, initializes random starting population
    evo = ea.evolutionaryAlg(size, popSize, mutProb, maxGen, evaluate, numClones, mutSize, genotypeLength)
    # get fitness of each individual in population
    evo.getFitnesses(locations)
    bestFitOfGen = min(evo.fitnesses)

    print(" ~ begin evolution ~ ")

    while bestFitOfGen > 0 and gen < maxGen:

        # A new generation
        gen = gen + 1
        print("-- generation %i --" % gen)

        # identify individuals with best fitnesses to clone, a selection function
        evo.getClones(fitnessType = "lowest")
        # all other individuals will get mutated
        evo.getMutants()
        # mutate everyone not identified for mutation
        evo.mutate()
        # get new population from mutants and clones
        evo.getNextGen()
        # have next population chemotax
        evo.getFitnesses(locations)

        # save stats from generations
        length = len(evo.pop)
        mean = sum(evo.fitnesses) / length
        sum2 = sum(x*x for x in evo.fitnesses)
        std = abs(sum2 / length - mean**2)**0.5
        maxFitn = max(evo.fitnesses)
        bestFitOfGen = min(evo.fitnesses)

        genMin.append(bestFitOfGen)
        genMax.append(maxFitn)
        genAvg.append(mean)
        genStd.append(std)
        generation.append(gen)

        #print("  min %s" % bestFitOfGen)
        #print("  max %s" % maxFitn)
        #print("  avg %s" % mean)
        #print("  std %s" % std)

    print("-- evolution complete --")
    if bestFitOfGen == 0:
        print("best fitness achieved")
    if gen == maxGen:
        print("maximum number of generations achieved")
    #print("best fit individual: %f" % min(genMin))

    # save genotype of top 3 individuals with best fitness
    bestfits = sorted(evo.fitnesses)[:nsave]
    bestGenotypes =[]
    [bestGenotypes.append(evo.pop[evo.fitnesses.index(g)]) for g in bestfits]

    # plot changes in fitnesses over evolutionary time
    #plt.plot(generation, genMax, label = "max fitness")
    #plt.plot(generation, genMin, label = "min fitness")
    #plt.plot(generation, genAvg, label = "avg fitness")
    #plt.ylabel("final distance from nutrient")
    #plt.xlabel("generation")
    #plt.legend()
    #plt.show()
    #plt.savefig("fitnessOverEvoTime.png")

    return bestGenotypes


#evolution()
sizes = [2,4,6,8,10]
niter = 100
ntop = 3
topGenotypes = np.zeros((ntop,niter,len(sizes)))
for size in sizes:
    genotypeLength = int(size*2 + (size**2-size)/2)
    topGenotypes = np.zeros((ntop,niter,genotypeLength))
    print("size:",size)
    for iter in range(niter):
        topGenotypes[:,iter,:] = evolution(size)
        print("iteration:",iter)

    filename = "top3Genotypes_" + str(size) + ".p"
    pickle.dump(topGenotypes, open(filename,"wb"))



# transfer to bigred3
# remote access, navigate to dir, then in terminal:
#scp hmerritt@ip:/Users/...pathToFile.../file file
