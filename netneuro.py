# network neuroscientific analyses
# time of development: spring 2022 by haily merritt

# import dependencies
from scipy import stats
import numpy as np


# functional connectivity
def getFC(timeseries):
    # takes in time series from N>1 neurons
    # return functional connectivity matrix
    # computed using Pearson correlation,
    # such that each element in the fc matrix is Pearson's r

    # z-score time series of each neuron
    timeseries = [stats.zscore(x) for x in timeseries]

    # initialize fc matrix
    n = len(timeseries) # number of nodes
    fc = np.zeros((n,n))

    # fill matrix with correlation coefficients (ie weights)
    i_ind = 0
    for i in timeseries:
        j_ind=0
        for j in timeseries:
            fc[i_ind,j_ind] = stats.pearsonr(i,j)[0]
            fc[j_ind,i_ind] = stats.pearsonr(i,j)[0]
            j_ind+=1
        i_ind+=1

    return fc


# edge time series
def getETS(timeseries):
    # takes in time series from N neurons
    # returns edge time series

    # z-score time series of each neuron
    timeseries = [stats.zscore(x) for x in timeseries]

    # initalize ets array
    n = len(timeseries)
    t = len(timeseries[0])
    ets = np.zeros((n**2,t))

    # compute edge time series
    ind = 0
    for i in timeseries:
        for j in timeseries:
            ets[ind,:] = i*j
            ind+=1

    return ets


# edge functional connectivity
def getEFC(timeseries):
    # takes in time series from N neurons
    # returns edge functional connectivity matrix
    # computed using Pearson correlation,
    # such that each element in the efc matrix is Pearson's r

    # get edge time series
    ets = getETS(timeseries)

    # initialize efc matrix
    n = len(timeseries) # number of nodes
    efc = np.zeros((n**2,n**2))

    # fill matrix with edge functional connectivity from edge time series
    i_ind = 0
    for i in ets:
        j_ind = 0
        for j in ets:
            efc[i_ind,j_ind] = stats.pearsonr(i,j)[0]
            efc[j_ind,i_ind] = stats.pearsonr(i,j)[0]
            j_ind+=1
        i_ind+=1

    return efc
