#%% Importing modules - コンプリート

"""
generalizedPermutations.py

Implementation of generalized permutation algorithms with visualizations.

Author: Jeffrey Morais, BTQ
"""

# Standard mathematics modules
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import networkx as nx
from tqdm import tqdm

# Group theoretic modules
from sympy.utilities.iterables import multiset_permutations
from sympy.combinatorics import Permutation, PermutationGroup
from sympy.combinatorics.named_groups import (SymmetricGroup, CyclicGroup, DihedralGroup)

# Set output directory to the current directory
output_dir = os.path.dirname(os.path.abspath(__file__))

# Set up styles for visualization
plt.style.use('dark_background')
COLORS = {
    'permutation': '#4CAF50',  # Green
    'orbit': '#F44336',        # Red
    'neutral': '#2196F3',      # Blue
    'symmetric': '#FFC107',    # Yellow/Gold
    'background': '#111111',
    'edge': '#555555',
    'highlight': '#9C27B0'     # Purple
}


#%% Defining modern & general Fisher Yates shuffles for different symmetry groups - コンプリート

# Defining the modern Fisher Yates swapping function
def FisherYates(inputSet,extraInfo=True):
    
    # Making a copy of the input to no overwrite it
    inputArray = inputSet.copy()
    
    # Printing the original state of the array
    if extraInfo == True:
        print('Initial array state: ', inputArray)
    
    # Looping over the different included endpoints of the array
    for endpoint in reversed(
        range(
            1, len(inputArray)
        )
    ):
        # Select a random array element (bounded by included endpoints)
        randomIndex = np.random.randint(0, endpoint+1)
        randomElement = inputArray[randomIndex]
        if extraInfo == True:
            print('Array element', randomElement, 'of index', randomIndex, 'selected.')
            
        # Swapping the randomly selected element with the current array endpoint
        if extraInfo == True:
            print('Swapped element', randomElement, 'with', inputArray[endpoint], end='.\n')
        inputArray[randomIndex], inputArray[endpoint] = inputArray[endpoint], inputArray[randomIndex]
        if extraInfo == True:
            print('New array state: ', inputArray)
            
    # Presenting the final state of the array
    if extraInfo == True:
        print('Final array state: ', inputArray)
        
    return inputArray


#%% Performing different group actions on sample input sets - コンプリート

# Selecting a random permutation of {0,...,n} as out input array
inputState = [0,1,2,3,4]
possiblePermutations = possiblePermutations = list(
    multiset_permutations(inputState)
)
selectedPermutation = possiblePermutations[
    np.random.randint(0,len(possiblePermutations))
]
print('Selected permutation for different groups: ', selectedPermutation)


# Peforming different transformations for different groups
print('--- Modern Fisher Yates ---')
FisherYates(selectedPermutation)
print('--- Generalized Fisher Yates (Permutation Group) ---')


# %% Computing probability distribution of group action for different Fisher Yates shuffles - コンプリート

# Defining a plotting function for a fixed amount of iterations
def groupDistribution(shuffleAlgorithm, inputSet, iterationSize, extraInfo=False):
    
     #Defining all possible unique permutations of the input set
    permutationArray = list(
        multiset_permutations(inputSet)
    )

    # Initialize an empty list to tally up the occurences of given permutations
    tallySet = list(
        np.zeros(
            len(permutationArray), dtype=int
        )
    )
    
    # Iterating through iterations of shuffles
    for iteration in range(1,iterationSize+1):
        
        # Performing a random permutation of the input set
        outputSet = shuffleAlgorithm(inputSet,False)
        if extraInfo == True:
            print('Permuted set: ',outputSet)
        
        # Checking if it is one of the permutations and if so add to the tally
        for permutation in range(
            len(permutationArray)
        ):
            if outputSet == permutationArray[permutation]:
                tallySet[permutation] += 1
            
    # Printing output tally set
    if extraInfo == True:           
        print('Final tally: ',tallySet)

    # Plotting results
    permutationIndexSet = list(
        np.arange(
            1, len(permutationArray)+1
        )
    )
    averageOccurence = np.mean(tallySet)/iterationSize

    fig, axs = plt.subplots(2, 1, figsize=(10, 5), constrained_layout=True)

    ax = axs[0]
    ax.scatter(permutationIndexSet,np.array(tallySet)/iterationSize, color='magenta',label='Frequency')
    ax.axhline(y=averageOccurence, color='black', label = 'Normalized Average Probability')
    ax.axhline(y=1/len(permutationArray), color='gray', label = 'Expected Average Probability', linestyle = 'dotted')
    ax.set_ylabel('Permutation Probability')
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.set_title(f'{iterationSize} Independent Shuffles with {shuffleAlgorithm.__name__}')
    ax.legend()
    ax.grid()
    ax.set_axisbelow(True)

    ax = axs[1]
    ax.scatter(permutationIndexSet,np.abs(
        ( np.array(tallySet)/iterationSize ) - averageOccurence
    ), color ='aqua', label='Normalized Residues')
    ax.set_xlabel('Unique Permutations')
    ax.set_ylabel('Logarithmic Variance')
    ax.set_yscale('log')
    ax.grid()
    ax.set_axisbelow(True)
    
    return


# Computing probability distribution for different group actions
print('--- Modern Fisher Yates ---')
groupDistribution(FisherYates,selectedPermutation,100000)
print('--- Generalized Fisher Yates (Permutation Group) ---')


#%% Defining general Fisher Yates shuffles with orbit probabilities - コンプリート

# Defining the generalized orbit Fisher Yates swapping function (G = permutation)
def permutationOrbitFisherYates(inputSet,extraInfo=True):
    
    # Initializing the set and permutation group with the sympy library
    inputArray = inputSet.copy()
    initializedSet = Permutation(inputArray)
    permutationGroup = PermutationGroup([initializedSet])
    
    # Printing the original state of the array
    if extraInfo == True:
        print('Initial array state: ', inputArray)
        
    # Initializing inverse orbit size list
    inverseOrbitSizes = list(
        np.zeros(
            len(inputArray)
        )
    )
    
    # Looping over the different included endpoints of the array
    for endpoint in reversed(
        range(
            0, len(inputArray) #CHANGED TO 0 INSTEAD OF 1
        )
    ):
        # Look at the orbit of the last element of the list
        lastElement = inputArray[endpoint]
        if extraInfo == True:
            print('Current last element: ', lastElement)
        lastElementOrbit = list(
            permutationGroup.orbit(lastElement)
        )
        if extraInfo == True:
            print('Orbit of last element: ', lastElementOrbit)
        
        # Defining the intersection of the orbit and the unselected list entries
        overlapList = list(
            set(inputArray[:endpoint+1]) & set(lastElementOrbit)
        )
        overlapSize = len(overlapList)
        if extraInfo == True:
            print('Transition overlap set: ', overlapList)
            print('Transition set size: ', overlapSize)
        inverseOrbitSizes[endpoint] = 1/overlapSize
        if extraInfo == True:
            print('Inverse Transition Size: ', 1/overlapSize)
        
        # Selecting a random element of the intersection/transition set
        randomIndex = np.random.randint(
            0, len(overlapList)
            )
        randomElement = overlapList[randomIndex]
        randomElementInputIndex = inputArray.index(randomElement)
        if extraInfo == True:
            print('Array element', randomElement, 'of index', randomElementInputIndex, 'selected.')
            
        # Swapping the randomly selected element with the current array endpoint
        if extraInfo == True:
            print('Swapped element', randomElement, 'with', lastElement, end='.\n')
        inputArray[randomElementInputIndex], inputArray[endpoint] = inputArray[endpoint], inputArray[randomElementInputIndex]
        if extraInfo == True:
            print('New array state: ', inputArray)
            
    # Presenting the final state of the array
    if extraInfo == True:
        print('Final array state: ', inputArray)
        
    # Multiplying all elements of inverseOrbitSizes for final permutation probality
    permutationProbability = np.prod(inverseOrbitSizes)
    if extraInfo == True:     
        print('Final inverse orbit sizes: ', inverseOrbitSizes)
        print('Final permutation probability: ', permutationProbability)
        
    return inputArray, permutationProbability


#%% Performing different group actions on sample input sets - コンプリート

# Selecting a random permutation of {0,...,n} as out input array
inputState = [0,1,2,3,4]
possiblePermutations = possiblePermutations = list(
    multiset_permutations(inputState)
)
selectedPermutation = possiblePermutations[
    np.random.randint(0,len(possiblePermutations))
]
print('Selected permutation for different groups: ', selectedPermutation)


# Peforming different transformations for different groups
print('--- Generalized Fisher Yates (Permutation Group) ---')
permutationOrbitFisherYates(selectedPermutation)
print('--- Generalized Fisher Yates (Symmetric Group) ---')


# %% Computing probability distribution of group action with orbit probabilities - コンプリート

# Defining a plotting function for a fixed amount of iterations
def groupOrbitDistribution(shuffleAlgorithm, inputSet, iterationSize, extraInfo=False):
    
     #Defining all possible unique permutations of the input set
    permutationArray = list(
        multiset_permutations(inputSet)
    )

    # Initialize an empty list to tally up the occurences of given permutations
    tallySet = list(
        np.zeros(
            len(permutationArray), dtype=int
        )
    )
    
    # Initialize an empty list of permutation probabilities
    probabilitySet = list(
        np.zeros(
            len(permutationArray), dtype=int
        )
    )
    
    # Iterating through iterations of shuffles
    for iteration in range(1,iterationSize+1):
        
        # Performing a random permutation of the input set
        fisherOutput = shuffleAlgorithm(inputSet,False)
        outputSet = fisherOutput[0]
        if extraInfo == True:
            print('Permuted set: ',outputSet)
        
        # Checking if it is one of the permutations and if so add to the tally
        for permutation in range(
            len(permutationArray)
        ):
            if outputSet == permutationArray[permutation]:
                tallySet[permutation] += 1
                probabilitySet[permutation] += fisherOutput[1]
            
    # Printing output tally set
    if extraInfo == True:           
        print('Final tally: ',tallySet)
        
    # Computed the predicted occurences based on inverse probabilities
    for i in range(
            len(permutationArray)
        ):
            probabilitySet[i] = (probabilitySet[i]/(tallySet[i]+0.0000000000000001)) # Added small number to avoid zero division    # Plotting results
    permutationIndexSet = list(
        np.arange(
            1, len(permutationArray)+1
        )
    )
    averageOccurence = np.mean(tallySet)/iterationSize

    fig, axs = plt.subplots(2, 1, figsize=(10, 5), constrained_layout=True, facecolor=COLORS['background'])

    ax = axs[0]
    ax.scatter(permutationIndexSet,np.array(tallySet)/iterationSize, color='magenta',label='Frequency')
    ax.axhline(y=averageOccurence, color='white', label = 'Normalized Average Probability')
    ax.scatter(permutationIndexSet,probabilitySet, color='gray', label = 'Expected Averages', linestyle = 'dotted')
    ax.set_ylabel('Permutation Probability')
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.set_title(f'{iterationSize} Independent Shuffles with {shuffleAlgorithm.__name__}')
    ax.legend()
    ax.grid(color='gray', alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_facecolor(COLORS['background'])
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.tick_params(colors='white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.title.set_color('white')

    ax = axs[1]
    ax.scatter(permutationIndexSet,np.abs(
        ( np.array(tallySet)/iterationSize ) - averageOccurence
    ), color ='aqua', label='Normalized Residues')
    ax.set_xlabel('Unique Permutations')
    ax.set_ylabel('Logarithmic Variance')
    ax.set_yscale('log')
    ax.grid(color='gray', alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_facecolor(COLORS['background'])
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.tick_params(colors='white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{shuffleAlgorithm.__name__}_orbit_distribution.png'), dpi=300, bbox_inches='tight')
    
    print('Amount of unique selected permutations: ', np.count_nonzero(tallySet))
    
    return


# Computing probability distribution for different group actions
print('--- Generalized Fisher Yates (Permutation Group) ---')
groupOrbitDistribution(permutationOrbitFisherYates,selectedPermutation,100000)
print('--- Generalized Fisher Yates (Symmetric Group) ---')


#%% Defining general non-degenerate Fisher Yates shuffles with orbit probabilities - コンプリート

# Defining the generalized orbit Fisher Yates swapping function (G = symmetric)
def symmetricNDFisherYates(inputSet,extraInfo=True):
    
    # Initializing the set and permutation group with the sympy library
    inputArray = inputSet.copy()
    symmetricGroup = SymmetricGroup(
        len(inputArray)
    )
    
    # Printing the original state of the array
    if extraInfo == True:
        print('Initial array state: ', inputArray)
        
    # Initializing inverse orbit size list
    inverseOrbitSizes = list(
        np.zeros(
            len(inputArray)
        )
    )
    
    # Looping over the different included endpoints of the array
    for endpoint in reversed(
        range(
            1, len(inputArray) #CHANGED BACK TO 1
        )
    ):
        # Look at the orbit of the last element of the list
        lastElement = inputArray[endpoint]
        if extraInfo == True:
            print('Current last element: ', lastElement)
        lastElementOrbit = list(
            symmetricGroup.orbit(lastElement)
        )
        if extraInfo == True:
            print('Orbit of last element: ', lastElementOrbit)
            
        # Look at the orbit (from the stabilizer subgroup) of the last element of the list
        stabilizerSubGroup = symmetricGroup.stabilizer(lastElement)
        stabilizerOrbit = list(
            stabilizerSubGroup.orbit(lastElement)
        )
        if extraInfo == True:
            print('Stabilizer orbit of last element: ', stabilizerOrbit)
        
        # Defining the intersection of the orbit (removing stabilizer orbit) and the unselected list entries
        overlapList = list(
            set(inputArray[:endpoint+1]) & ( set(lastElementOrbit) - set(stabilizerOrbit) )
        )
        overlapSize = len(overlapList)
        if extraInfo == True:
            print('Transition overlap set: ', overlapList)
            print('Transition set size: ', overlapSize)
        inverseOrbitSizes[endpoint] = 1/overlapSize
        if extraInfo == True:
            print('Inverse Transition Size: ', 1/overlapSize)
            
        # Removing problematic first entry
        inverseOrbitSizes[0] = 1.0
        
        # Selecting a random element of the intersection/transition set
        randomIndex = np.random.randint(
            0, len(overlapList)
            )
        randomElement = overlapList[randomIndex]
        randomElementInputIndex = inputArray.index(randomElement)
        if extraInfo == True:
            print('Array element', randomElement, 'of index', randomElementInputIndex, 'selected.')
            
        # Swapping the randomly selected element with the current array endpoint
        if extraInfo == True:
            print('Swapped element', randomElement, 'with', lastElement, end='.\n')
        inputArray[randomElementInputIndex], inputArray[endpoint] = inputArray[endpoint], inputArray[randomElementInputIndex]
        if extraInfo == True:
            print('New array state: ', inputArray)
            
    # Presenting the final state of the array
    if extraInfo == True:
        print('Final array state: ', inputArray)
        
    # Multiplying all elements of inverseOrbitSizes for final permutation probality
    permutationProbability = np.prod(inverseOrbitSizes)
    if extraInfo == True:     
        print('Final inverse orbit sizes: ', inverseOrbitSizes)
        print('Final permutation probability: ', permutationProbability)
        
    return inputArray, permutationProbability


#%% Performing different non-degenerate group actions on sample input sets - コンプリート

# Selecting a random permutation of {0,...,n} as out input array
inputState = [0,1,2,3,4]
possiblePermutations = possiblePermutations = list(
    multiset_permutations(inputState)
)
selectedPermutation = possiblePermutations[
    np.random.randint(0,len(possiblePermutations))
]
print('Selected permutation for different groups: ', selectedPermutation)


# Peforming different transformations for different groups
print('--- Generalized ND Fisher Yates (Symmetric Group) ---')
symmetricNDFisherYates(selectedPermutation)
print('--- Generalized ND Fisher Yates (Cyclic Group) ---')


# %% Computing probability distribution of non-degenerate group action with orbit probabilities - コンプリート

# Computing probability distribution for different group actions
print('--- Generalized ND Fisher Yates (Symmetric Group) ---')
groupOrbitDistribution(symmetricNDFisherYates,selectedPermutation,100000)
print('--- Generalized ND Fisher Yates (Cyclic Group) ---')


#%% Manually defining the Fisher Yates shuffles with orbit probabilities under the consensus group (symmetric group modulo degeneracies) - コンプリート

# Defining the consensus group Fisher Yates swapping function
def consensusFisherYates(inputSet,extraInfo=True):
    
    # Initializing the set to prevent it being overwritten
    inputArray = inputSet.copy()
    
    # Printing the original state of the array
    if extraInfo == True:
        print('Initial array state: ', inputArray)
        
    # Initializing inverse orbit size list
    inverseOrbitSizes = list(
        np.zeros(
            len(inputArray)
        )
    )
    
    # Looping over the different included endpoints of the array
    for endpoint in reversed(
        range(
            1, len(inputArray) #CHANGED BACK TO 1
        )
    ):
        # Look at the orbit of the last element of the list
        lastElement = inputArray[endpoint]
        if extraInfo == True:
            print('Current last element: ', lastElement)
        lastElementOrbit = sorted(inputArray[:endpoint+1])
        if extraInfo == True:
            print('Orbit of last element: ', lastElementOrbit)
            
        # Look at the orbit (from the stabilizer subgroup) of the last element of the list
        stabilizerOrbit = [lastElement]
        if extraInfo == True:
            print('Stabilizer orbit of last element: ', stabilizerOrbit)
        
        # Defining the intersection of the orbit (removing stabilizer orbit) and the unselected list entries
        overlapList = list(
            set(inputArray[:endpoint+1]) & ( set(lastElementOrbit) - set(stabilizerOrbit) )
        )
        overlapSize = len(overlapList)
        if extraInfo == True:
            print('Transition overlap set: ', overlapList)
            print('Transition set size: ', overlapSize)
        inverseOrbitSizes[endpoint] = 1/overlapSize
        if extraInfo == True:
            print('Inverse Transition Size: ', 1/overlapSize)
            
        # Removing problematic first entry
        inverseOrbitSizes[0] = 1.0
        
        # Selecting a random element of the intersection/transition set
        randomIndex = np.random.randint(
            0, len(overlapList)
            )
        randomElement = overlapList[randomIndex]
        randomElementInputIndex = inputArray.index(randomElement)
        if extraInfo == True:
            print('Array element', randomElement, 'of index', randomElementInputIndex, 'selected.')
            
        # Swapping the randomly selected element with the current array endpoint
        if extraInfo == True:
            print('Swapped element', randomElement, 'with', lastElement, end='.\n')
        inputArray[randomElementInputIndex], inputArray[endpoint] = inputArray[endpoint], inputArray[randomElementInputIndex]
        if extraInfo == True:
            print('New array state: ', inputArray)
            
    # Presenting the final state of the array
    if extraInfo == True:
        print('Final array state: ', inputArray)
        
    # Multiplying all elements of inverseOrbitSizes for final permutation probality
    permutationProbability = np.prod(inverseOrbitSizes)
    if extraInfo == True:     
        print('Final inverse orbit sizes: ', inverseOrbitSizes)
        print('Final permutation probability: ', permutationProbability)
        
    return inputArray, permutationProbability


# %% Computing probability distribution of non-degenerate group action with orbit probabilities - コンプリート

# Selecting a random permutation of {0,...,n} as out input array
inputState = [0,1,2,3,4]
possiblePermutations = possiblePermutations = list(
    multiset_permutations(inputState)
)
selectedPermutation = possiblePermutations[
    np.random.randint(0,len(possiblePermutations))
]
print('Selected permutation for different groups: ', selectedPermutation)

# Performing consensus shuffle on selected permutation
print('--- Consensus Fisher Yates (Consensus Group) ---')
consensusFisherYates(selectedPermutation)

# Computing probability distribution for the consensus group
print('--- Consensus Fisher Yates (Consensus Group) ---')
groupOrbitDistribution(consensusFisherYates,selectedPermutation,100000)


# %% Consctruction of the edge sets for proper permutation - コンプリート

# Defining the edge set
testEdgeSet = np.array(
    [
        [1,2],[2,1],[2,3],[3,2],[4,4]
    ]
)

# Defining the Fisher Yates swapping function for the edge set via the consensus group (manual: orbits superfluous)
def edgeFisherYates(inputEdgeSet, extraInfo=True):
    
    # Initializing the edge set and its columns
    edgeSet = np.copy(inputEdgeSet)
    leftColumnEdge = list(edgeSet.T[0])
    rightColumnEdge = list(edgeSet.T[1])
    
    # Printing original state of the edge sets and the vertex indices
    if extraInfo == True:
        print('Initial edge set of the graph:\n', edgeSet)
        print('Target list:\n', leftColumnEdge)
        print('Dual list:\n', rightColumnEdge)
        
    # Looping over the different included endpoints of the target list
    for endpoint in reversed(
        range(
            0, len(leftColumnEdge)-1
        )
    ):
        # Current last point and its dual in the right column
        lastPoint = leftColumnEdge[endpoint]
        dualPoint = rightColumnEdge[endpoint]
        if extraInfo == True:
            print('Current last element: ', lastPoint)
        
        # Randomly selecting points until they satistfy non-degeneracy
        for testIteration in range(
            0, 1000
        ):
        
            # Select a random array element (bounded by included endpoints)
            randomIndex = np.random.randint(0, endpoint+1)
            randomElement = leftColumnEdge[randomIndex]
            dualElement = leftColumnEdge[randomIndex]
            
            # Degeneracy condition
            if (lastPoint != randomElement) and (dualPoint != dualElement):
                if extraInfo == True:
                    print('Satifying element selected!')
                    print('Left edge element', randomElement, 'of index', randomIndex, 'selected.')
                break
            else:
                if extraInfo == True:
                    print('Non-satisfying element selected! Reselecting.')
                
        # Performing the swap with the randomly selected non-degenerate element
        leftColumnEdge[endpoint], leftColumnEdge[randomIndex] = leftColumnEdge[randomIndex], leftColumnEdge[endpoint]
        if extraInfo == True:
            print('Swapped element', randomElement, 'with', lastPoint, end='.\n')
        if extraInfo == True:
            print('New target list state: ', leftColumnEdge)
            
    # Presenting the final state of the target list
    if extraInfo == True:
        print('Final array state: ', leftColumnEdge)
            
    # Gluing back the pieces of the edge set together
    finalEdgeSet = np.column_stack(
        (leftColumnEdge,rightColumnEdge)
    )
    
    # Presenting the final state of the edge set
    if extraInfo == True:
        print('Final state of edge set:\n', finalEdgeSet)
        
    return finalEdgeSet, leftColumnEdge
    
# Performing a test of the new algorithm
edgeFisherYates(testEdgeSet)


# %% Computing probability distribution of group action for consensus Fisher Yates shuffles - コンプリート

# Defining a plotting function for a fixed amount of iterations
def edgeDistribution(shuffleAlgorithm, inputEdgeSet, iterationSize, extraInfo=False):
    
    # Defining the target list
    inputSet = list(
        inputEdgeSet.T[0]
    )
    
    # Defining all possible unique permutations of the input set
    permutationArray = list(
        multiset_permutations(inputSet)
    )
    print('Total possible permutations: ', len(permutationArray))
    print('All possible unique permutations: ', permutationArray)

    # Initialize an empty list to tally up the occurences of given permutations
    tallySet = list(
        np.zeros(
            len(permutationArray), dtype=int
        )
    )
    
    # Iterating through iterations of shuffles
    for iteration in range(1,iterationSize+1):
        
        # Performing a random permutation of the input set
        outputSet = shuffleAlgorithm(inputEdgeSet,False)[1]
        if extraInfo == True:
            print('Permuted set: ',outputSet)
        
        # Checking if it is one of the permutations and if so add to the tally
        for permutation in range(
            0,len(permutationArray)-1
        ):
            if outputSet == permutationArray[permutation]:
                tallySet[permutation] += 1
            
    # Printing output tally set
    if extraInfo == True:           
        print('Final tally: ',tallySet)

    # Plotting results
    permutationIndexSet = list(
        np.arange(
            1, len(permutationArray)+1
        )
    )
    averageOccurence = np.mean(tallySet)/iterationSize

    fig, axs = plt.subplots(2, 1, figsize=(10, 5), constrained_layout=True, facecolor=COLORS['background'])

    ax = axs[0]
    ax.scatter(permutationIndexSet,np.array(tallySet)/iterationSize, color='magenta',label='Frequency')
    ax.axhline(y=averageOccurence, color='white', label = 'Normalized Average Probability')
    ax.axhline(y=1/len(permutationArray), color='gray', label = 'Expected Average Probability', linestyle = 'dotted')
    ax.set_ylabel('Permutation Probability')
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.set_title(f'{iterationSize} Independent Shuffles with {shuffleAlgorithm.__name__}')
    ax.legend()
    ax.grid(color='gray', alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_facecolor(COLORS['background'])
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.tick_params(colors='white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.title.set_color('white')

    ax = axs[1]
    ax.scatter(permutationIndexSet,np.abs(
        ( np.array(tallySet)/iterationSize ) - averageOccurence
    ), color ='aqua', label='Normalized Residues')
    ax.set_xlabel('Unique Permutations')
    ax.set_ylabel('Logarithmic Variance')
    ax.set_yscale('log')
    ax.grid(color='gray', alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_facecolor(COLORS['background'])
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.tick_params(colors='white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{shuffleAlgorithm.__name__}_edge_distribution.png'), dpi=300, bbox_inches='tight')
    
    return


# %% Trying again for a larger set, and a trivial set
largeEdgeSet = np.array(
    [
        [1,1],[1,2],[2,1],[3,2],[3,3],[4,1],[5,2]
    ]
)
trivialEdgeSet = np.array(
    [
        [1,1],[2,2],[3,3]
    ]
)

print('--- Large Edge Set ---')
edgeDistribution(edgeFisherYates,largeEdgeSet,10000)
print('--- Trivial Edge Set ---')
edgeDistribution(edgeFisherYates,trivialEdgeSet,10000)


# %% Redefing the cosensus shuffle using Peter's notation

# Defining the consensus shuffle with a deterministic run time
def consensusShuffle(targetEdgeSet, extraInfo=True):
    
    # Initializing the edge set and its columns
    edgeSet = np.copy(targetEdgeSet)
    uColumn = list(edgeSet.T[0]) #\vec{v}
    vColumn = list(edgeSet.T[1]) #\vec{u}
    
    # Extra info: printing the original configuration of the set
    if extraInfo == True:
        print('Initial edge set of the graph:\n', edgeSet)
        print('Target list:\n', uColumn)
        print('Dual list:\n', vColumn)
        
    # Looping over the different endpoints of the unselected elements
    for i in reversed(
        range(
            1, len(uColumn) # Not starting at 0 because the alg is trivial for last element
        )
    ):
        # Defining the current ith endpoint of the unselected elements of U
        lastPoint = uColumn[i]
        dualLastPoint = vColumn[i]
        
        # Extra info: printing current selected endpoint
        if extraInfo == True:
            print('Current endpoint: ', lastPoint)
        
        # Initializing the transition set
        t = []
        
        # Constructing the transition set based on non-degenerate elements
        for k in range(
            0, i
            ):
            
            # Defining temporary jth elements in U and V columns
            kthPoint = uColumn[k]
            dualKthPoint = vColumn[k]
            
            # Appending non-degenerate points to the transition set
            if (lastPoint != kthPoint) and (dualLastPoint != dualKthPoint):
                t.append(kthPoint)
                
        # Extra info: printing the transition set
        if extraInfo == True:
            print('Current transition set: ', t)        
                
        # Randomly selecting a point in the transition set and swapping elements
        j = random.choice(t)
        jIndex = uColumn.index(j)
        uColumn[i], uColumn[jIndex] = uColumn[jIndex], uColumn[i]
        
        # Extra info: printing randomly selected element
        if extraInfo == True:
            print('Selected transition point: ', j)
            print('Selected point index in U: ', jIndex)
        
        # Extra info: printing swapped elements and new list
        if extraInfo == True:
            print('Swapped element', uColumn[i], 'with', uColumn[jIndex])
            print('Current target list: ', uColumn)
            print('Current dual list: ', vColumn)
        
    # Gluing back the pieces of the edge set together 
    finalEdgeSet = np.column_stack(
        (uColumn,vColumn)
    )
    
    # Extra info: presenting the final state of the edge set
    if extraInfo == True:
        print('Final state of edge set:\n', finalEdgeSet)
        
    return finalEdgeSet, uColumn
                

# %% Computing the probability distribution for different edge sets
print('--- Test Edge Set ---')
edgeDistribution(consensusShuffle,testEdgeSet,100000)
# print('--- Large Edge Set ---')
# edgeDistribution(consensusShuffle,largeEdgeSet,10000)
print('--- Trivial Edge Set ---')
edgeDistribution(consensusShuffle,trivialEdgeSet,100000)


# %% Enhanced visualization functions for network-based permutations - コンプリート

def visualize_permutation_network(shuffle_algorithm, input_set, iterations=100, network_type='random'):
    """
    Visualize permutations as network transformations
    
    Parameters:
    -----------
    shuffle_algorithm : function
        The permutation algorithm to visualize
    input_set : list
        The initial set to permute
    iterations : int
        Number of iterations for statistical visualization
    network_type : str
        Type of network to visualize ('random', 'small_world', 'scale_free')
    """
    # Create a network based on the type
    n_nodes = len(input_set)
    if network_type == 'random':
        G = nx.erdos_renyi_graph(n_nodes, 0.4)
    elif network_type == 'small_world':
        G = nx.watts_strogatz_graph(n_nodes, 4, 0.3)
    elif network_type == 'scale_free':
        G = nx.barabasi_albert_graph(n_nodes, 2)
    else:
        G = nx.complete_graph(n_nodes)
    
    # Run permutations and collect statistics
    permutation_counts = {}
    for i in tqdm(range(iterations), desc=f"Running {shuffle_algorithm.__name__} iterations"):
        # Apply shuffle algorithm
        result = shuffle_algorithm(input_set.copy(), False)
        if isinstance(result, tuple):
            perm = tuple(result[0])  # Convert to tuple for hashing
        else:
            perm = tuple(result)  # Convert to tuple for hashing
        
        # Count permutations
        if perm in permutation_counts:
            permutation_counts[perm] += 1
        else:
            permutation_counts[perm] = 1
    
    # Get the most common permutations
    top_permutations = sorted(permutation_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    most_common_perm = list(top_permutations[0][0])
    
    # Skip saving PNG images - we will only create animations
    
    # Create animation of permutation process (except for problematic algorithm)
    if shuffle_algorithm.__name__ != "permutationOrbitFisherYates":
        create_permutation_animation(shuffle_algorithm, input_set, G, network_type)
    else:
        print(f"Skipping animation for {shuffle_algorithm.__name__} on {network_type} network (known issue)")
    
    return permutation_counts

def create_permutation_animation(shuffle_algorithm, input_set, network, network_type, frames=20):
    """
    Create an animation showing the permutation process on a network
    """
    # Skip creating animations for the problematic algorithm
    if shuffle_algorithm.__name__ == "permutationOrbitFisherYates":
        print(f"Skipping animation for {shuffle_algorithm.__name__} on {network_type} network (known issue)")
        return
        
    G = network
    pos = nx.spring_layout(G)
    
    # Run the algorithm to get the final permutation
    result = shuffle_algorithm(input_set.copy(), False)
    if isinstance(result, tuple):
        final_perm = result[0]
    else:
        final_perm = result
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    def animate(i):
        ax.clear()
        
        # For the first frame, show initial state
        if i == 0:
            current_perm = input_set.copy()
        # For the last frame, show final state
        elif i == frames - 1:
            current_perm = final_perm
        # For intermediate frames, interpolate
        else:
            # Create a partial permutation for this frame
            current_perm = input_set.copy()
            fraction = i / (frames - 1)
            
            # Swap elements based on frame number
            for j in range(len(input_set)):
                if j < int(fraction * len(input_set)):
                    current_perm[j] = final_perm[j]
        
        # Draw the network edges
        nx.draw_networkx_edges(G, pos, ax=ax, width=1.0, alpha=0.5, edge_color=COLORS['edge'])
        
        # Draw nodes with colors based on permutation state
        node_colors = []
        for node in G.nodes():
            if node < len(current_perm) and current_perm[node] == node:  # Fixed point
                node_colors.append(COLORS['symmetric'])
            elif node < len(current_perm):  # Moved element
                node_colors.append(COLORS['permutation'])
            else:
                node_colors.append(COLORS['neutral'])
                
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=list(G.nodes()), 
                              node_color=node_colors, node_size=700, alpha=0.8)
        
        # Draw node labels
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=12, font_family='monospace', font_color='white')
        
        # Add arrows for permutation mapping
        for j, node in enumerate(G.nodes()):
            if j < len(current_perm) and current_perm[j] != j:
                target = current_perm[j]
                # Add a curved arrow showing permutation
                ax.annotate("", 
                          xy=pos[target], 
                          xytext=pos[j],
                          arrowprops=dict(arrowstyle="->", color=COLORS['permutation'], 
                                         connectionstyle="arc3,rad=0.3", lw=2))
        
        # Title and frame info
        ax.set_title(f"{shuffle_algorithm.__name__} on {network_type} network - Frame {i+1}/{frames}", 
                    color='white')
        return ax,
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=300, blit=False)
    
    # Save animation
    filename = f"{shuffle_algorithm.__name__}_{network_type}_network_animation.gif"
    anim.save(os.path.join(output_dir, filename), writer='pillow', fps=4, 
              dpi=100, savefig_kwargs={'facecolor': COLORS['background']})
    
    plt.close()

# %% Generate enhanced visualizations for different permutation algorithms - コンプリート

def generate_all_visualizations():
    """Generate all visualizations for the different permutation algorithms"""
    print("Generating enhanced permutation visualizations...")
    
    # Set up input data
    input_set = [0, 1, 2, 3, 4, 5, 6]
    
    # Define permutation algorithms to visualize
    algorithms = [
        FisherYates,
        # Skip permutationOrbitFisherYates which has corrupted GIFs
        symmetricNDFisherYates,
        consensusFisherYates
    ]
    
    # Define network types
    network_types = ['random', 'small_world', 'scale_free']
    
    # Generate visualizations for each algorithm on each network type
    for algorithm in algorithms:
        for network_type in network_types:
            print(f"\nVisualizing {algorithm.__name__} on {network_type} network...")
            visualize_permutation_network(algorithm, input_set, iterations=1000, network_type=network_type)
    
    print("\nAll visualizations complete!")

# Only run this if explicitly called
if __name__ == "__main__":
    generate_all_visualizations()
    cleanup_image_files()

def cleanup_image_files():
    """
    Clean up any excess PNG files while preserving distribution PNGs and animations
    """
    print("Cleaning up image files...")
    
    # Get all files in the directory
    for filename in os.listdir(output_dir):
        # Only keep orbit_distribution.png, edge_distribution.png, and animation GIFs
        if filename.endswith('.png') and not any(keyword in filename for keyword in 
                                               ['orbit_distribution.png', 'edge_distribution.png']):
            # This is not a distribution PNG, so delete it
            try:
                os.remove(os.path.join(output_dir, filename))
                print(f"Removed excess file: {filename}")
            except Exception as e:
                print(f"Error removing {filename}: {e}")
    
    print("Cleanup completed.")

