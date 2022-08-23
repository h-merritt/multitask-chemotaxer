# handling code for evolution of an adaptive chemotaxer
# haily merritt
# time of development: summer 2021

### GOALS:
#   1. evolve an agent that can chemotax to two nutrients, thus action switching
#   2. compare agent's performance across multiple trials

# notes
#boundaries: kill agents if they cross boundary -> give fitness of 0
# average fitness over time
#check if alive throughout trial/for loop

### SET UP
# import packages
import random
import numpy as np
import matplotlib.pyplot as plt
import evalg as ea
import agent

### DEFINE PARAMETER VALUES
# for evolutionary algorithm
size = 2                                # number of neurons in CTRNN
mutProb = 1                             # num of mutations per genotype
mutSize = 0.25                          # amount to mutate gene
popSize = 10                             # size of population
maxGen = 10                             # maximum number of generations to evolve
genotypeLength = size + size + size**2  # length of genotype as determined by number of network parameters
numClones = 2
# for chemotaxis
duration = 2500                         # duration of each trial
tau = 1                                 # time constant, subject to evalg
step_size = 0.01                        # integration size
envWidth = 100                          # width of environment
maxDist = np.sqrt(envWidth**2 + envWidth**2)    # maximum distance agent could be from nutrient in environment
locations = [[[50,30,90],[20,60]], [[50,30,180],[20,60]], [[50,30,0],[20,60]], [[50,30,0],[40,80]], [[50,30,270],[45,80]], [[30,30,90],[10,75]], [[50,30,270],[10,80]], [[50,20,90],[15,80]], [[50,10,180],[40,80]], [[30,30,180],[10,80]], [[30,30,0],[10,90]]]
reps = len(locations)                   # number of times simulation should be repeated if using multipleRuns


### DEFINE FUNCTIONS
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

    while time < duration and individual.alive == True:
        print("hello world")
        # update agent's sensors
        individual.sense(food)
        # agent moves
        individual.tax(duration)
        #increase time
        time += 1

    if individual.alive == False:
        fit = maxDist
    else:
        fit = individual.distance(food)

    return fit


def evaluateMulti(reps, agent_params, locations):
    # simulate multiple chemotaxing agents

    rep = 0
    dists = []


    while rep < reps:

        dists.append(evaluate(agent_params, locations[rep]))

        rep += 1

    return sum(dists)/reps



def evolution():

    # initialize stats
    gen = 0
    genMin = []
    genMax = []
    genAvg = []
    genStd = []
    generation = []

    # create evolutionary algorithm, initializes random starting population
    evo = ea.evolutionaryAlg(size, popSize, mutProb, maxGen, evaluateMulti, numClones, mutSize, genotypeLength)
    # get fitness of each individual in population
    evo.getFitnesses(reps, locations)
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
        evo.getFitnesses(reps, locations)

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

        print("  min %s" % bestFitOfGen)
        #print("  max %s" % maxFitn)
        #print("  avg %s" % mean)
        #print("  std %s" % std)

    print("-- evolution complete --")
    if bestFitOfGen == 0:
        print("best fitness achieved")
    if gen == maxGen:
        print("maximum number of generations achieved")
    print("best fit individual: %f" % min(genMin))
    #save genMax, genMin, genAvg, and genStd

    # plot changes in fitnesses over evolutionary time
    plt.plot(generation, genMax, label = "max fitness")
    plt.plot(generation, genMin, label = "min fitness")
    plt.plot(generation, genAvg, label = "avg fitness")
    plt.ylabel("final distance from nutrient")
    plt.xlabel("generation")
    plt.legend()
    plt.show()

evolution()
