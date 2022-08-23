# haily merritt
# generic evolutionary algorithm

#import packages
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import math

# define an evolutionary algorithm
class evolutionaryAlg:

    def __init__(self, size, popSize, mutProb, maxGen, evaluate, numClones, mutSize, genotypeLength):

        self.popSize = popSize               # size of population
        self.mutProb = mutProb               # probability of mutation
        self.maxGen = maxGen                 # max number of generations to evolve
        self.evaluate = evaluate             # function used for evaluation of fitnesses
        self.pop = np.random.random((popSize, genotypeLength))   # list of all genotypes of population
        self.fitnesses = []                  # list of fitnesses for all individuals in population
        self.clones = []                     # list of all individuals to clone
        self.mutants = []                    # list of all individuals to mutate
        self.numClones = numClones           # number of individuals to be cloned
        self.mutSize = mutSize               # amount to mutate a given gene
        self.size = size                     # number of neurons in network
        self.genotypeLength = genotypeLength # number of parameters of network


    # get fitnesses
    def getFitnesses(self, locations):

        self.fitnesses = [self.evaluate(genotype, locations) for genotype in self.pop]

    # get clones
    def getClones(self, fitnessType):
        # identify best performing individuals to clone
        # best performing = highest fitness
        if fitnessType == "highest":
            reverse = True
        elif fitnessType == "lowest":
            reverse = False

        # take the second element for sort
        def take_second(elem):
            return elem[1]

        self.clones = []

        elites = sorted(zip(self.pop, self.fitnesses), reverse = reverse, key = take_second)[:self.numClones]

        [self.clones.append(copy.deepcopy(elites[ind][0])) for ind in range(self.numClones)]

    # get mutants
    def getMutants(self):

        self.mutants = []

        numMutants = self.popSize - self.numClones

        [self.mutants.append(self.pop[ind]) for ind in random.sample(range(0,len(self.pop)-1), numMutants)]

    # mutate the mutants
    def mutate(self):

        for genotype in self.mutants:

            genotype += np.random.normal(0.0, self.mutSize, size = self.genotypeLength)

    # define selection function
    def getNextGen(self):

        self.pop = self.clones + self.mutants
        self.fitnesses = []
