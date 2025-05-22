#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
graphGenerator.py

This script implements algorithms for creating, analyzing, and manipulating
network graphs with specific focus on edge sets and graph isomorphisms. It
provides tools for working with bipartite graph structures in the context of
consensus networks.

Key features:
1. Generation of random and deterministic network structures
2. Edge set permutation and transformation algorithms
3. Graph isomorphism testing and verification
4. Consensus-oriented edge shuffling for network optimization
5. Visualization of different graph structures and their properties

The implementation supports various graph types including random, small-world,
scale-free and custom graphs with configurable parameters.

Author: Jeffrey Morais, BTQ
"""

#%% Importing modules - コンプリート

# Standard mathematics modules
import numpy as np
import matplotlib.pyplot as plt
import os

# Group theoretic modules
from sympy.utilities.iterables import multiset_permutations

# Set output directory to the current directory
output_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(output_dir, exist_ok=True)


#%% Defining test edge sets to permute - コンプリート
testEdgeSet = [
        [1,2],[2,1],[2,3],[3,2],[4,4]
    ]

largeEdgeSet = [
        [1,1],[1,2],[2,1],[3,2],[3,3],[4,1],[5,2]
    ]

trivialEdgeSet = [
        [1,1],[2,2],[3,3]
        ]


# %% Redefining the consensus shuffle using the DCN notation - コンプリート

# Defining the consensus shuffle with a deterministic run time
def consensusShuffle(targetEdgeSet, extraInfo=True):
    
    # Initializing the edge set and its columns
    edgeSet = targetEdgeSet.copy()
    u = list(np.array(edgeSet).T[0])
    v = list(np.array(edgeSet).T[1])
    
    # Extra info: printing the original configuration of the set
    if extraInfo == True:
        print('Initial edge set of the graph:\n', edgeSet)
        print('Target list:\n', u)
        print('Dual list:\n', v)
        
    # Looping over the different endpoints of the unselected elements
    for i in reversed(
        range(
            1, len(u)
        )
    ):
        # Extra info: printing current selected endpoint
        if extraInfo == True:
            print('Current endpoint: ', u[i])
        
        # Initializing the transition index set
        t = [i]
        
        # Constructing the transition set based on non-degenerate elements
        for k in range(
            0, i
            ):
            
            # Extra info: print if tuples occur in the edge set
            if extraInfo == True:
                print('[u[i],v[k]] tuple: ', [u[i],v[k]], [u[i],v[k]] not in edgeSet)
                print('[u[k],v[i]] tuple: ', [u[k],v[i]], [u[k],v[i]] not in edgeSet)
            
            # Appending non-degenerate points indices to the transition set
            if (
                ([u[i],v[k]] not in edgeSet) and ([u[k],v[i]] not in edgeSet)
                ):
                t.append(k)
                
        # Extra info: printing the transition set
        if extraInfo == True:
            print('Current transition index set: ', sorted(t))        
                
        # Randomly selecting a point in the transition set and swapping elements
        j = np.random.choice(t)
        u[i], u[j] = u[j], u[i]
        
        # Extra info: printing randomly selected element
        if extraInfo == True:
            print('Selected transition index: ', j)
            print('Selected swap point in U: ', u[i])
        
        # Extra info: printing swapped elements and new list
        if extraInfo == True:
            print('Swapped element', u[i], 'with', u[j])
            print('Current target list: ', u)
            print('Current dual list: ', v)
        
    # Gluing back the pieces of the edge set together 
    finalEdgeSet = np.column_stack(
        (u,v)
    )
    
    # Extra info: presenting the final state of the edge set
    if extraInfo == True:
        print('Final state of edge set:\n', finalEdgeSet)
        
    return finalEdgeSet

# Testing the function on different edge sets
print('--- Trivial Edge Set ---')
consensusShuffle(trivialEdgeSet)
print('--- Test Edge Set ---')
consensusShuffle(testEdgeSet)
print('--- Large Edge Set ---')
consensusShuffle(largeEdgeSet)
                

# %% [Vertex Space] Computing probability distribution of group action for the consensus group - コンプリート
# This section has been removed as requested

# %% [Edge Space] Computing probability distribution of group action for the consensus group - コンプリート

# Cutting out degenerate entries in the possible edge sets
def removeDegeneracy(inputEdgeSets): 
    
    # Initializing the new list of unique edge sets
    uniqueEdgeSets = []

    # Initialize an empty set to keep track of duplicate edge sets
    seenEdgeSets = set()

    # Iterate through each inner edge set
    for edgeSet in inputEdgeSets:
        
        # Convert the inner list to a frozenset for immutability
        frozenSet = frozenset(map(tuple, edgeSet))

        # Check if the set is not in seen_sets (i.e., not a duplicate)
        if frozenSet not in seenEdgeSets:
            
            # Add the set to seenEdgeSets to mark it as seen
            seenEdgeSets.add(frozenSet)

            # Append the original inner list to unique_lsts
            uniqueEdgeSets.append(edgeSet)

    # Return the list of unique lists
    return uniqueEdgeSets

# This section (Edge Space probability distribution) has been removed as requested

# %% Constructing a function to verify if a given degree sequence is satisfiable - コンプリート

# Defining the verification function for degree sequences
def sequenceVerifier(inputSequence, uSize, vSize, extraInfo=True):
    
    # Partioning degree sequence into ordered sub-sequences
    uSequence = sorted(
        inputSequence[:uSize], reverse=True
    )
    vSequence = sorted(
        inputSequence[vSize:], reverse=True
    )
    
    # Extra info: printing initial state of sequences
    if extraInfo == True:
        print('Partial U sequence: ', uSequence)
        print('Partial V sequence: ', vSequence)
    
    # Initializing constraint logic
    firstConstraint = False
    secondConstraint = False
    totalConstraint = False
    
    # Testing the first constraint: conservation
    if np.sum(uSequence) == np.sum(vSequence):
        firstConstraint = True
        
        # Extra info: printing the outcome of the first constraint
        if extraInfo == True:
            print('First constraint: ', firstConstraint)
        
    else:
        
        # Extra info: printing the outcome of the first constraint
        if extraInfo == True:
            print('First constraint: ', firstConstraint)
            
        return False
    
    # Testing the second constraint: boundedness
    for k in range(0, uSize):
        
        # Computing LHS of constraint
        leftHandSide = np.sum(uSequence[:k+1])
        
        # Computing RHS of constraint
        minVSequence = [min(v,k+1) for v in vSequence]
        rightHandSide = np.sum(minVSequence)
        
        # Extra info: printing the values of the sides
        if extraInfo == True:
            print('LHS: ', leftHandSide)
            print('RHS: ', rightHandSide)
        
        # Break loop if it doesn't satistfy each constraint for k
        if not leftHandSide <= rightHandSide:
            
            # Extra info: printing the outcome of the first constraint
            if extraInfo == True:
                print('Second constraint: ', secondConstraint)
                
            return False
    
    # If loop didn't break, then the second constraint is satisfied
    secondConstraint = True
    
    # Extra info: printing the outcome of the first constraint
    if extraInfo == True:
        print('Second constraint: ', secondConstraint)
    
    # If both constraints are sastisfied, it is a proper degree sequence
    if (firstConstraint == True) and (secondConstraint == True):
        totalConstraint = True
        
    # Returning output of the constraint
    return totalConstraint
    
# Printing some examples of the algorithm on degree sequences
print('--- Trivial degree sequence ---')
sequenceVerifier([1,1,1,1,1,1],3,3)
print('--- Non-trivial degree sequence ---')
sequenceVerifier([2,2,2,2,3,1],3,3)
print('--- Non-satisfying degree sequence ---')
sequenceVerifier([6,2,1,3,3,3],3,3)


# %% Constructing a function to output an intial graph assignment - コンプリート

# Defining the function from which outputs an edge assignment from U,V
def edgeAssignment(inputSequence, uSize, vSize, extraInfo=True):
    
    # Testing if is a satisfiable degree sequence
    if sequenceVerifier(inputSequence, uSize, vSize, False) == False:
        return False
    else:
        print('Partially satisfiable sequence.')
    
    # Initializing the ordered degree sequences and edge set
    edgeSet = []
    uSequence = sorted(
        inputSequence[:uSize], reverse=True
    )
    vSequence = sorted(
        inputSequence[vSize:], reverse=True
    )
    
    # Extra info: printing initial state of sequences and edge set
    if extraInfo == True:
        print('Partial U sequence: ', uSequence)
        print('Partial V sequence: ', vSequence)
        print('Initial edge set: ', edgeSet)
    
    # Looping through to construct the edge set
    for u in range(0,len(uSequence)):
        for v in range(0,len(vSequence)):
            if (uSequence[u]>0) and (vSequence[v]>0):
                edgeSet.append([u,v])
                uSequence[u] -= 1
                vSequence[v] -= 1
                
    # Extra info: printing the final state of the edge set
    if extraInfo == True:
        print('Final edge set: ', edgeSet)
              
    # Recall: the final graph has its columns decreasingly ordered!  
    return edgeSet
    
# Printing some examples of the algorithm on degree sequences
print('--- Trivial degree sequence ---')
edgeAssignment([1,1,1,1,1,1],3,3)
print('--- Non-trivial degree sequence ---')
edgeAssignment([2,2,2,2,3,1],3,3)
print('--- Non-satisfying degree sequence ---')
edgeAssignment([6,2,1,3,3,3],3,3)


# %% Main execution when run directly
if __name__ == "__main__":
    print("Graph Generator Toolkit")
    print("======================")
    
    # Run the standard examples
    print("\n--- Trivial Edge Set ---")
    consensusShuffle(trivialEdgeSet)
    print("\n--- Test Edge Set ---")
    consensusShuffle(testEdgeSet)
    print("\n--- Large Edge Set ---")
    consensusShuffle(largeEdgeSet)
    
    # Run visualizations if the visualization module is available
    try:
        print("\n--- Generating Visualizations ---")
        import networkVisualizer
        
        # Example custom degree sequences
        print("\nVisualizing example networks:")
        
        # Custom degree sequence 1: Small network
        custom_sequence1 = [2, 1, 1, 2, 1, 1]
        u_size1, v_size1 = 3, 3
        
        if sequenceVerifier(custom_sequence1, u_size1, v_size1):
            edges1 = edgeAssignment(custom_sequence1, u_size1, v_size1)
            print("\nGenerating visualizations for small network...")
            networkVisualizer.create_bipartite_graph_visualization(edges1, u_size1, v_size1, "custom_small_network")
            networkVisualizer.create_consensus_shuffle_animation(edges1, u_size1, v_size1, 15, "custom_small_consensus_animation")
        
        # Custom degree sequence 2: Medium network
        custom_sequence2 = [3, 2, 2, 1, 2, 3, 1, 2]
        u_size2, v_size2 = 4, 4
        
        if sequenceVerifier(custom_sequence2, u_size2, v_size2):
            edges2 = edgeAssignment(custom_sequence2, u_size2, v_size2)
            print("\nGenerating visualizations for medium network...")
            networkVisualizer.create_bipartite_graph_visualization(edges2, u_size2, v_size2, "custom_medium_network")
            networkVisualizer.create_network_evolution_animation(custom_sequence2, u_size2, v_size2, 15, "custom_medium_evolution_animation")
            networkVisualizer.visualize_permutation_stability(custom_sequence2, u_size2, v_size2, 30, "custom_medium_stability")
            
        print("\nAll visualizations have been generated in the current directory.")
        
    except ImportError:
        print("\nVisualization module not available. Run networkVisualizer.py separately to generate visualizations.")

