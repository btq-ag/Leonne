#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trustPartitioner.py

This script implements a classical trust-based network partitioning algorithm.
It analyzes trust relationships between nodes and networks to optimize the
partitioning of nodes into networks based on trust values and security thresholds.

The algorithm includes:
- Trust calculation between nodes and networks
- Jump phase: Moving nodes to networks with higher trust
- Abandon phase: Isolating nodes with insufficient internal trust
- Network-level trust computation

Author: Jeffrey Morais, BTQ
"""

import numpy as np
import string # For naming the networks with letter strings


# %% Defining the trust functions at different scales

# Defining the trust function between nodes and networks
def r(trustAinB, forward=True):
    
    # Computing the r_i(B) values for all nodes i in A
    if forward == True:
        
        # Initializing the network and empty trust values
        networkSize = len(trustAinB) #|A|
        trustValues = np.zeros(networkSize)
        for i in range(networkSize):
            trustValues[i] =  1/(networkSize) * np.sum(trustAinB[i])
            # {r_i(B),...,r_k(B)}
             
    # Computing the r_A(j) values for all nodes j in B
    if forward == False:
        
        # Initializing the network and empty trust values
        networkSize = len(trustAinB.T) #|B|
        trustValues = np.zeros(networkSize)
        for j in range(networkSize):
            trustValues[j] =  1/(networkSize) * np.sum(trustAinB.T[j])
            # {r_A(l),...,r_A(m)}
    
    # Returning the desired trust array
    return trustValues

# Defining the trust function between networks   
def rr(trustAinB, trustBinA, forward=True):
    
    # Computing the r_i(N) sums amongst the networks
    trustIinB = r(trustAinB) # {r_i(B),...,r_k(B)}
    trustJinA = r(trustBinA) # {r_l(A),...,r_m(A)}
    
    # Computing the trust between networks as r(A,B)
    if forward == True:
        networkTrustAinB = np.max(trustIinB)
        
        # Returning the network wide trust of A in B
        return networkTrustAinB
    
    # Computing the trust between networks as r(B,A)
    if forward == False:
        networkTrustBinA = np.max(trustJinA)
        
        # Returning the network wide trust of B in A
        return networkTrustBinA

# Defining an ordered combination of networks and empty sets
def networkCombiner(inputNetworks,inputEmptySets):
    
    # Removing any empty sets
    emptyRemover = lambda set: [i for i in set if i]

    # Extracting values from empty sets
    emptyValues = sorted(list(inputEmptySets.values()))
    
    # Extracting non-empty networks
    extractedNetworks = emptyRemover(inputNetworks)
    extractedEmptySets = emptyRemover(emptyValues)
    
    # Combining them into a total collection of networks
    collectedNetworks = extractedNetworks + extractedEmptySets
    
    # Ordering individual networks
    networkOrdering = [sorted(i) for i in collectedNetworks]
    
    # Returning the combined ordered network
    return networkOrdering


# %% Defining the networks, their trust, and their security - コンプリート

# Defining the networks and their elements
N = [0,1,2] # Network N with nodes: i,j,k = 0,1,2
M = [3,4] # Network M with nodes: l,m = 3,4
networks = [N,M]

# Defining all trust matrix permutations
trustNinN = np.array([
    [0,1/2,1/4],
    [1/3,0,1/4],
    [1/2,1,0]
])
trustNinM = np.array([
    [1,1/5],
    [1/3,1/2],
    [1/3,1]
])
trustMinN = np.array([
    [1/7,1/2,1/3],
    [1,1/4,1/2]
])
trustMinM = np.array([
    [0,1/4],
    [1/3,0]
])

# Combining trust matrices into one total one
systemTrust = np.hstack((
    np.vstack((trustNinN,trustMinN)), np.vstack((trustNinM,trustMinM))
))

# Defining the security parameters of the nodes in the network
securityN = {0:1/6,1:1/2,2:1/3}
securityM = {3:1/2,4:1/4}
nodeSecurity = securityN | securityM

# Defining the network wide security parameters via most lenient n/2 nodes
networkSecurity = {}
for networkIndex, network in enumerate(networks):
    
    # Get the security values for the current network
    securityValues = [nodeSecurity[node] for node in network]
    
    # Sort the security values in increasing order
    orderedSecurity = sorted(securityValues)
    
    # Find the most lenient security value among the top n/2 nodes
    mostLenientSecurity = np.max(orderedSecurity[len(orderedSecurity) // 2:])
    
    # Assign the most lenient security value to the corresponding network
    networkSecurity[networkIndex] = mostLenientSecurity


# %% Defining the full generalized network partitioning function

# Generalizing the network partition function to arbitrary networks
def networkPartitioner(inputNetworks, inputTrust, nodesSecurity, networksSecurity):
    
    # Initializing information of the networks provided
    networksCopy = [j.copy() for j in inputNetworks]
    networkSizes = [len(i) for i in inputNetworks]
    numberOfNetworks = len(networkSizes)
    networkNames = list(string.ascii_uppercase)[:numberOfNetworks]
    networkEmptySets = {k:[] for k in range(np.sum(networkSizes))}

    # Printing initial configuration of the networks
    print('--- Original state of networks ---')
    for i in range(len(inputNetworks)):
        print('Initial network', networkNames[i],':', inputNetworks[i])
    
    # Extracting the trust block matrices
    blockMatrices = []
    sizeSum = lambda alpha: sum(networkSizes[i] for i in range(alpha))
    for i in range(numberOfNetworks):
        for j in range(numberOfNetworks):
            blockMatrices.append(inputTrust[sizeSum(i):sizeSum(i+1),sizeSum(j):sizeSum(j+1)])
            
    # Computing the trust sums from the perspectives of the nodes and the networks
    nodeSums = []
    networkSums = []
    for i in blockMatrices:
        nodeSums.append(r(i))
    for i in blockMatrices:
        networkSums.append(r(i,False))
        
    # Iterating through the network to compute optimal jump locations
    optimalLocations = []
    for n in range(len(inputNetworks)):
    
        # Printing the information of the current network
        networkLength = len(inputNetworks[n])
        targetTrusts = nodeSums[n*len(inputNetworks):(n+1)*len(inputNetworks)]
        
        # Initializing jump locations for nodes in network n
        jumpLocations = []
        
        # Iterating through the nodes of the networks
        for i in range(networkLength):
            for j in targetTrusts:
                jumpLocations.append(j[i])

        # Making them into q-plets
        pletFormer = lambda lst, q: [lst[i:i + q] for i in range(0, len(lst), q)]
        orderedLocations = pletFormer(jumpLocations,numberOfNetworks)
        
        # Computing optimal locations
        optimalLocationsTemp = [loc.index(min(loc)) for loc in orderedLocations]
        optimalLocations.append(optimalLocationsTemp)

    # Returning complete list of optimal locations for all nodes in each network
    print('Full optimal locations:', optimalLocations)
    
    # Performing potential jumps for every node in every networks
    for n in range(len(inputNetworks)):
    
        # Printing configuration of the current network
        currentNetwork = inputNetworks[n]
        currentCopy = networksCopy[n]
        currentOptimal = optimalLocations[n]
        
        # Defining external networks trusts in this one
        externalTrusts = []
        trustExtractor = lambda lst, q: [[lst[i] for i in range(start, len(lst), q)] for start in range(q)]
        externalTrusts = trustExtractor(networkSums,numberOfNetworks)[n]
        
        # Iterating through the nodes of the networks
        for i in currentNetwork:

            # Only looking at nodes originally in network
            if i in currentCopy:
                
                # Extracting the index of node i
                nodeIndex = currentNetwork.index(i)
                
                # Selecting optimal location for node i in network n
                location = currentOptimal[nodeIndex]
                locationNetwork = networksCopy[location]

                # Non-trivial location condition
                if location != n:
                    print('Node', currentNetwork[nodeIndex], 'in network', networkNames[n], 'wants to jump network',networkNames[location]+'!')
                    
                    # Seeing validity of trust condition [in function this should be networksSecurity]
                    if externalTrusts[location][nodeIndex] <= networksSecurity[location]:
                        
                        # Move node from one network to another
                        jumpNode = currentCopy.pop(currentCopy.index(i))
                        locationNetwork.insert(len(locationNetwork), jumpNode)
                        print('Node', currentNetwork[nodeIndex], 'has jumped to network',networkNames[location]+'!')
    
    # Printing the outcome of all potential jump events                                
    print('Networks after jumping events:', networksCopy)
    
    # Performing abandons for every network
    for n in range(len(inputNetworks)):
        
        # Printing configuration of the current network
        currentNetwork = inputNetworks[n]
        currentCopy = networksCopy[n]

        # Defining the trusts of nodes in their own network
        internalTrusts = nodeSums[n*(numberOfNetworks+1)]
        
        # Iterating through the nodes of the networks
        for i in currentNetwork:
            
            # Only looking at nodes originally in the network
            if i in currentCopy:
                
                # Extracting the index of node i
                nodeIndex = currentNetwork.index(i)
                
                # Seeing validity of trust conditions [in function this should be nodesSecurity]
                if internalTrusts[nodeIndex] > nodesSecurity[i]:
                    
                    # Move node i to ∅_i
                    jumpNode = currentCopy.pop(currentCopy.index(i))
                    networkEmptySets[i].insert(0,jumpNode)
                    print('Node', i, 'in network', networkNames[n], 'has abandoned its network!')

    # Printing the outcome of all potential abandon events
    print('Networks after abandon events:', networksCopy) 
    print('Emtpy sets after evolution:', networkEmptySets)
    
    # Creating new networks from old ones
    newNetworks = networkCombiner(networksCopy,networkEmptySets)
    print('Updated networks:', newNetworks)
    
    return newNetworks, inputTrust, nodesSecurity, networksSecurity

# Testing the function above on our network configuration
testOutput = networkPartitioner(networks,systemTrust,nodeSecurity,networkSecurity)


# %% Testing the code with N = 3 networks

# Defining the networks and their elements
N = [0,1,2] # Network N with nodes: i,j,k = 0,1,2
M = [3,4] # Network M with nodes: l,m = 3,4
P = [5] # Network P with nodes: p = 5
networks = [N,M,P]

# Defining random total trust matrix
systemTrust = np.random.rand(6,6)

# Defining the security parameters of the nodes in the network
securityN = {0:1/6,1:1/2,2:1/3}
securityM = {3:1/2,4:1/4}
securityP = {5:1/7}
nodeSecurity = securityN | securityM | securityP

# Defining the network wide security parameters via most lenient n/2 nodes
networkSecurity = {}
for networkIndex, network in enumerate(networks):
    
    # Get the security values for the current network
    securityValues = [nodeSecurity[node] for node in network]
    
    # Sort the security values in increasing order
    orderedSecurity = sorted(securityValues)
    
    # Find the most lenient security value among the top n/2 nodes
    mostLenientSecurity = np.max(orderedSecurity[len(orderedSecurity) // 2:])
    
    # Assign the most lenient security value to the corresponding network
    networkSecurity[networkIndex] = mostLenientSecurity

# Testing the function above on our network configuration
testOutput = networkPartitioner(networks,systemTrust,nodeSecurity,networkSecurity)


# %% Testing the code with N = 4 networks

# Defining the networks and their elements
N = [0,1,2] # Network N with nodes: i,j,k = 0,1,2
M = [3,4] # Network M with nodes: l,m = 3,4
P = [5,6,7] # Network P with nodes: p,q,r = 5,6,7
Q = [8,9] # Network Q with nodes: a,b = 8,9
networks = [N,M,P,Q]

# Defining random total trust matrix
systemTrust = np.random.rand(10,10)

# Defining the security parameters of the nodes in the network
securityN = {0:1/6,1:1/2,2:1/3}
securityM = {3:1/2,4:1/4}
securityP = {5:1/7,6:1/8,7:1/2}
securityQ = {8:1/3,9:1/4}
nodeSecurity = securityN | securityM | securityP | securityQ

# Defining the network wide security parameters via most lenient n/2 nodes
networkSecurity = {}
for networkIndex, network in enumerate(networks):
    
    # Get the security values for the current network
    securityValues = [nodeSecurity[node] for node in network]
    
    # Sort the security values in increasing order
    orderedSecurity = sorted(securityValues)
    
    # Find the most lenient security value among the top n/2 nodes
    mostLenientSecurity = np.max(orderedSecurity[len(orderedSecurity) // 2:])
    
    # Assign the most lenient security value to the corresponding network
    networkSecurity[networkIndex] = mostLenientSecurity

# Testing the function above on our network configuration
testOutput = networkPartitioner(networks,systemTrust,nodeSecurity,networkSecurity)


# %%
