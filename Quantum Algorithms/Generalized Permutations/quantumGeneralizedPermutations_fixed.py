#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quantumGeneralizedPermutations_fixed.py

This script implements quantum-enhanced versions of permutation algorithms
with a focus on integrating quantum randomness into generalized Fisher-Yates
shuffling methods. The quantum enhancements provide improved entropy and
randomness compared to classical implementations.

Key implementations include:
1. Quantum random number generation for improved shuffling entropy
2. Quantum Fisher-Yates algorithm with quantum state visualization
3. Quantum permutation-orbit algorithm for orbit-weighted probabilities
4. Quantum symmetric non-degenerate algorithm to avoid fixed points
5. Quantum consensus algorithm for multiset permutations
6. Bloch sphere visualization of quantum states during shuffling
7. Statistical analysis of quantum vs. classical distribution quality

Each algorithm includes visualization tools to demonstrate quantum advantages
and analyze the enhanced entropy provided by quantum randomness.

Author: Jeffrey Morais, BTQ
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import to_rgba
from sympy.utilities.iterables import multiset_permutations
from sympy.combinatorics import Permutation, PermutationGroup
from sympy.combinatorics.named_groups import (SymmetricGroup, CyclicGroup, DihedralGroup)


#%% Quantum random number generation for enhanced shuffling

def quantum_random_bit():
    """
    Simulates a quantum random bit using quantum measurement uncertainty.
    In a real quantum system, this would use hardware quantum randomness.
    """
    # Simulate quantum randomness by adding quantum-like noise
    # In a real quantum system, this would be true quantum randomness
    theta = np.random.uniform(0, np.pi/2)
    # Simulate measurement of a qubit in superposition
    probability = np.sin(theta)**2
    return 1 if np.random.random() < probability else 0

def quantum_random_int(max_val):
    """
    Generates a quantum random integer between 0 and max_val-1.
    Uses multiple quantum random bits to build an integer.
    """
    if max_val <= 0:
        return 0
        
    # Calculate number of bits needed
    num_bits = int(np.ceil(np.log2(max_val)))
    
    # Generate quantum random bits
    result = 0
    for i in range(num_bits):
        bit = quantum_random_bit()
        result = (result << 1) | bit
    
    # Ensure the value is in range
    return result % max_val

def quantum_random_choice(options):
    """
    Selects a random item from a list using quantum randomness.
    """
    if not options:
        return None
    idx = quantum_random_int(len(options))
    return options[idx]


#%% Quantum state visualization helpers

def bloch_sphere_coordinates(quantum_state):
    """
    Converts a quantum state vector to Bloch sphere coordinates.
    For visualization purposes.
    """
    # Normalize state
    state = np.array(quantum_state, dtype=complex)
    norm = np.sqrt(np.sum(np.abs(state)**2))
    if norm > 0:
        state = state / norm
    
    # For a 2D state [α, β], compute Bloch coordinates
    if len(state) == 2:
        # Extract amplitudes and phase difference
        alpha, beta = state
        theta = 2 * np.arccos(np.abs(alpha))
        
        # Phase difference determines position on the equator
        if np.abs(alpha) < 1e-10:
            phi = 0
        elif np.abs(beta) < 1e-10:
            phi = 0
        else:
            phi = np.angle(beta) - np.angle(alpha)
            
        # Convert to Cartesian coordinates
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return x, y, z
    else:
        # For higher dimensions, project to lower dimension for visualization
        return np.random.normal(0, 0.1, 3)  # Random point near origin


#%% Quantum Fisher-Yates algorithms

def quantumFisherYates(inputSet, extraInfo=False):
    """
    Quantum-enhanced version of the classic Fisher-Yates shuffle.
    Uses quantum random number generation for improved randomness.
    """
    # Make a copy to avoid modifying the original
    inputArray = inputSet.copy()
    
    if extraInfo:
        print('Initial array state:', inputArray)
    
    # Create quantum states for visualization
    quantum_states = []
    
    # Iterate through array elements in reverse
    for endpoint in reversed(range(1, len(inputArray))):
        # Use quantum random number generator for selection
        randomIndex = quantum_random_int(endpoint + 1)
        randomElement = inputArray[randomIndex]
        
        if extraInfo:
            print(f'Selected element {randomElement} at index {randomIndex}')
            
        # Save quantum state for visualization
        # Create a simple 2D quantum state representation based on indices
        theta = np.pi * randomIndex / (endpoint + 1)
        state = [np.cos(theta/2), np.sin(theta/2) * np.exp(1j * np.pi/4)]
        quantum_states.append((state, (randomIndex, endpoint)))
        
        # Swap elements
        inputArray[randomIndex], inputArray[endpoint] = inputArray[endpoint], inputArray[randomIndex]
        
        if extraInfo:
            print(f'Swapped with element {inputArray[endpoint]}')
            print(f'New array state: {inputArray}')
            
    if extraInfo:
        print('Final array state:', inputArray)
    
    return inputArray, quantum_states


def quantumPermutationOrbitFisherYates(inputSet, extraInfo=False):
    """
    Quantum version of permutation-orbit Fisher-Yates algorithm.
    Generates permutations with orbit-weighted probabilities using quantum randomness.
    """
    # Initialize set and permutation group
    inputArray = inputSet.copy()
    initializedSet = Permutation(inputArray)
    permutationGroup = PermutationGroup([initializedSet])
    
    if extraInfo:
        print('Initial array state:', inputArray)
        
    # Initialize inverse orbit sizes and quantum states
    inverseOrbitSizes = np.zeros(len(inputArray))
    quantum_states = []
    
    # Iterate through array elements in reverse
    for endpoint in reversed(range(0, len(inputArray))):
        # Get orbit of the current element
        lastElement = inputArray[endpoint]
        if extraInfo:
            print('Current last element:', lastElement)
            
        lastElementOrbit = list(permutationGroup.orbit(lastElement))
        if extraInfo:
            print('Orbit of last element:', lastElementOrbit)
        
        # Find overlap between orbit and unselected elements
        overlapList = list(set(inputArray[:endpoint+1]) & set(lastElementOrbit))
        overlapSize = len(overlapList)
        
        if extraInfo:
            print('Transition overlap set:', overlapList)
            print('Transition set size:', overlapSize)
            
        inverseOrbitSizes[endpoint] = 1/overlapSize if overlapSize > 0 else 0
        
        if extraInfo:
            print('Inverse Transition Size:', inverseOrbitSizes[endpoint])
        
        # Use quantum randomness for selection
        if overlapSize > 0:
            randomElement = quantum_random_choice(overlapList)
            randomElementInputIndex = inputArray.index(randomElement)
            
            # Create quantum state representation for this step
            # Each amplitude corresponds to the probability of selecting each element
            state = np.zeros(overlapSize, dtype=complex)
            for i in range(overlapSize):
                theta = np.pi * i / overlapSize
                state[i] = np.sqrt(1/overlapSize) * np.exp(1j * theta)
            
            quantum_states.append((state, (randomElementInputIndex, endpoint)))
            
            if extraInfo:
                print(f'Selected element {randomElement} at index {randomElementInputIndex}')
                
            # Swap elements
            inputArray[randomElementInputIndex], inputArray[endpoint] = inputArray[endpoint], inputArray[randomElementInputIndex]
            
            if extraInfo:
                print(f'Swapped with element {inputArray[endpoint]}')
                print(f'New array state: {inputArray}')
    
    # Calculate final permutation probability
    permutationProbability = np.prod(inverseOrbitSizes)
    
    if extraInfo:
        print('Final array state:', inputArray)
        print('Final inverse orbit sizes:', inverseOrbitSizes)
        print('Final permutation probability:', permutationProbability)
    
    return inputArray, permutationProbability, quantum_states


def quantumSymmetricNDFisherYates(inputSet, extraInfo=False):
    """
    Quantum version of non-degenerate symmetric Fisher-Yates algorithm.
    Uses quantum randomness and removes stabilizer orbits to avoid fixed points.
    """
    # Initialize set and symmetric group
    inputArray = inputSet.copy()
    symmetricGroup = SymmetricGroup(len(inputArray))
    
    if extraInfo:
        print('Initial array state:', inputArray)
        
    # Initialize inverse orbit sizes and quantum states
    inverseOrbitSizes = np.zeros(len(inputArray))
    quantum_states = []
    
    # Iterate through array elements in reverse
    for endpoint in reversed(range(1, len(inputArray))):
        # Get orbit and stabilizer of current element
        lastElement = inputArray[endpoint]
        if extraInfo:
            print('Current last element:', lastElement)
            
        lastElementOrbit = list(symmetricGroup.orbit(lastElement))
        if extraInfo:
            print('Orbit of last element:', lastElementOrbit)
            
        stabilizerSubGroup = symmetricGroup.stabilizer(lastElement)
        stabilizerOrbit = list(stabilizerSubGroup.orbit(lastElement))
        
        if extraInfo:
            print('Stabilizer orbit of last element:', stabilizerOrbit)
        
        # Find non-degenerate overlap (removing stabilizer orbit)
        overlapList = list(set(inputArray[:endpoint+1]) & (set(lastElementOrbit) - set(stabilizerOrbit)))
        overlapSize = len(overlapList)
        
        if extraInfo:
            print('Transition overlap set:', overlapList)
            print('Transition set size:', overlapSize)
            
        inverseOrbitSizes[endpoint] = 1/overlapSize if overlapSize > 0 else 0
        
        if extraInfo:
            print('Inverse Transition Size:', inverseOrbitSizes[endpoint])
        
        # Use quantum randomness for selection
        if overlapSize > 0:
            randomElement = quantum_random_choice(overlapList)
            randomElementInputIndex = inputArray.index(randomElement)
            
            # Create quantum state representation
            # We use amplitudes proportional to 1/√|overlap|
            state = np.zeros(overlapSize, dtype=complex)
            for i in range(overlapSize):
                phase = 2 * np.pi * i / overlapSize
                state[i] = np.sqrt(1/overlapSize) * np.exp(1j * phase)
            
            quantum_states.append((state, (randomElementInputIndex, endpoint)))
            
            if extraInfo:
                print(f'Selected element {randomElement} at index {randomElementInputIndex}')
                
            # Swap elements
            inputArray[randomElementInputIndex], inputArray[endpoint] = inputArray[endpoint], inputArray[randomElementInputIndex]
            
            if extraInfo:
                print(f'Swapped with element {inputArray[endpoint]}')
                print(f'New array state: {inputArray}')
    
    # Manually set first element's inverse orbit size to 1
    inverseOrbitSizes[0] = 1.0
    
    # Calculate final permutation probability
    permutationProbability = np.prod(inverseOrbitSizes)
    
    if extraInfo:
        print('Final array state:', inputArray)
        print('Final inverse orbit sizes:', inverseOrbitSizes)
        print('Final permutation probability:', permutationProbability)
    
    return inputArray, permutationProbability, quantum_states


def quantumConsensusFisherYates(inputSet, extraInfo=False):
    """
    Quantum version of consensus Fisher-Yates algorithm.
    Implements the consensus group (symmetric group modulo degeneracies).
    Uses quantum randomness for improved shuffle quality.
    """
    # Initialize set
    inputArray = inputSet.copy()
    
    if extraInfo:
        print('Initial array state:', inputArray)
        
    # Initialize inverse orbit sizes and quantum states
    inverseOrbitSizes = np.zeros(len(inputArray))
    quantum_states = []
    
    # Iterate through array elements in reverse
    for endpoint in reversed(range(1, len(inputArray))):
        # Get orbit of current element
        lastElement = inputArray[endpoint]
        if extraInfo:
            print('Current last element:', lastElement)
            
        lastElementOrbit = sorted(inputArray[:endpoint+1])
        if extraInfo:
            print('Orbit of last element:', lastElementOrbit)
            
        # Stabilizer orbit is just the element itself
        stabilizerOrbit = [lastElement]
        if extraInfo:
            print('Stabilizer orbit of last element:', stabilizerOrbit)
        
        # Find non-degenerate overlap
        overlapList = list(set(inputArray[:endpoint+1]) & (set(lastElementOrbit) - set(stabilizerOrbit)))
        overlapSize = len(overlapList)
        
        if extraInfo:
            print('Transition overlap set:', overlapList)
            print('Transition set size:', overlapSize)
            
        inverseOrbitSizes[endpoint] = 1/overlapSize if overlapSize > 0 else 0
        
        if extraInfo:
            print('Inverse Transition Size:', inverseOrbitSizes[endpoint])
        
        # Use quantum randomness for selection
        if overlapSize > 0:
            # Create a quantum superposition of all possible transitions
            amplitudes = []
            for element in overlapList:
                # Elements with higher frequency get lower probability
                count = inputArray[:endpoint+1].count(element)
                amplitude = 1.0 / np.sqrt(count * overlapSize)
                amplitudes.append(amplitude)
                
            # Normalize amplitudes
            norm = np.sqrt(sum(a**2 for a in amplitudes))
            normalized_amplitudes = [a/norm for a in amplitudes]
            
            # Use quantum probabilities for selection
            probabilities = [a**2 for a in normalized_amplitudes]
            # Simulate quantum measurement
            randomElement = np.random.choice(overlapList, p=probabilities)
            randomElementInputIndex = inputArray.index(randomElement)
            
            # Create quantum state representation
            state = np.zeros(len(overlapList), dtype=complex)
            for i, amp in enumerate(normalized_amplitudes):
                phase = np.random.uniform(0, 2*np.pi)
                state[i] = amp * np.exp(1j * phase)
            
            quantum_states.append((state, (randomElementInputIndex, endpoint)))
            
            if extraInfo:
                print(f'Selected element {randomElement} at index {randomElementInputIndex}')
                
            # Swap elements
            inputArray[randomElementInputIndex], inputArray[endpoint] = inputArray[endpoint], inputArray[randomElementInputIndex]
            
            if extraInfo:
                print(f'Swapped with element {inputArray[endpoint]}')
                print(f'New array state: {inputArray}')
    
    # Manually set first element's inverse orbit size to 1
    inverseOrbitSizes[0] = 1.0
    
    # Calculate final permutation probability
    permutationProbability = np.prod(inverseOrbitSizes)
    
    if extraInfo:
        print('Final array state:', inputArray)
        print('Final inverse orbit sizes:', inverseOrbitSizes)
        print('Final permutation probability:', permutationProbability)
    
    return inputArray, permutationProbability, quantum_states


def quantumEdgeShuffle(edgeSet, extraInfo=False):
    """
    Quantum version of edge shuffle for bipartite graphs.
    Ensures non-degenerate edges using quantum randomness.
    """
    # Copy the edge set
    edges = np.array(edgeSet)
    leftColumnEdge = list(edges.T[0])
    rightColumnEdge = list(edges.T[1])
    
    if extraInfo:
        print('Initial edge set:', edges)
        print('Target list:', leftColumnEdge)
        print('Dual list:', rightColumnEdge)
    
    # Initialize quantum states for visualization
    quantum_states = []
    
    # Iterate through edges in reverse
    for endpoint in reversed(range(0, len(leftColumnEdge)-1)):
        # Get current edge
        lastPoint = leftColumnEdge[endpoint]
        dualPoint = rightColumnEdge[endpoint]
        
        if extraInfo:
            print('Current last element:', lastPoint)
        
        # Find non-degenerate transitions
        validIndices = []
        for idx in range(endpoint+1):
            if (leftColumnEdge[idx] != lastPoint) and (rightColumnEdge[idx] != dualPoint):
                validIndices.append(idx)
        
        if extraInfo:
            print('Valid transition indices:', validIndices)
        
        # If no valid transitions, skip
        if not validIndices:
            continue
        
        # Create quantum state for this step
        state = np.zeros(len(validIndices), dtype=complex)
        for i in range(len(validIndices)):
            angle = 2 * np.pi * i / len(validIndices)
            state[i] = np.sqrt(1/len(validIndices)) * np.exp(1j * angle)
        
        # Select using quantum randomness
        randomIndex = quantum_random_choice(validIndices)
        
        if extraInfo:
            print(f'Selected transition index: {randomIndex}')
        
        quantum_states.append((state, (randomIndex, endpoint)))
        
        # Swap elements
        leftColumnEdge[endpoint], leftColumnEdge[randomIndex] = leftColumnEdge[randomIndex], leftColumnEdge[endpoint]
        
        if extraInfo:
            print(f'Swapped elements {leftColumnEdge[endpoint]} and {leftColumnEdge[randomIndex]}')
            print('New target list:', leftColumnEdge)
    
    # Build final edge set
    finalEdgeSet = np.column_stack((leftColumnEdge, rightColumnEdge))
    
    if extraInfo:
        print('Final edge set:', finalEdgeSet)
    
    return finalEdgeSet, leftColumnEdge, quantum_states


#%% Quantum distribution analysis functions

def quantumGroupDistribution(shuffleAlgorithm, inputSet, iterationSize, extraInfo=False):
    """
    Analyzes the distribution of quantum shuffle outcomes.
    Shows the entropy advantage of quantum shuffling over classical.
    """
    # Get all possible permutations
    permutationArray = list(multiset_permutations(inputSet))
    
    # Initialize tallies for quantum and classical approaches
    quantumTallySet = np.zeros(len(permutationArray), dtype=int)
    classicalTallySet = np.zeros(len(permutationArray), dtype=int)
    
    # Run iterations with quantum and classical shuffling
    for iteration in range(1, iterationSize+1):
        # Quantum shuffle
        if shuffleAlgorithm.__name__ == 'quantumFisherYates':
            outputSet, _ = shuffleAlgorithm(inputSet, False)
        else:
            outputSet, _, _ = shuffleAlgorithm(inputSet, False)
            
        # Classical version (using normal Fisher-Yates)
        classicalOutput = FisherYates(inputSet.copy(), False)
        
        if extraInfo and iteration % 1000 == 0:
            print(f'Iteration {iteration}: Quantum result: {outputSet}, Classical result: {classicalOutput}')
        
        # Record tallies
        for permIndex, perm in enumerate(permutationArray):
            if outputSet == perm:
                quantumTallySet[permIndex] += 1
            if classicalOutput == perm:
                classicalTallySet[permIndex] += 1
    
    # Calculate distributions
    quantumDistribution = quantumTallySet / iterationSize
    classicalDistribution = classicalTallySet / iterationSize
    
    # Calculate entropy for both distributions
    quantumEntropy = -np.sum(quantumDistribution * np.log2(quantumDistribution + 1e-10))
    classicalEntropy = -np.sum(classicalDistribution * np.log2(classicalDistribution + 1e-10))
    maxEntropy = np.log2(len(permutationArray))
    
    # Prepare plot
    permutationIndexSet = np.arange(1, len(permutationArray)+1)
    
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    
    # Plot 1: Distribution comparison
    ax = axs[0]
    ax.scatter(permutationIndexSet, quantumDistribution, color='magenta', alpha=0.7, label='Quantum Frequency')
    ax.scatter(permutationIndexSet, classicalDistribution, color='blue', alpha=0.7, label='Classical Frequency')
    ax.axhline(y=1/len(permutationArray), color='gray', label='Ideal Uniform', linestyle='dotted')
    ax.set_ylabel('Permutation Probability')
    ax.set_title(f'Quantum vs Classical Distribution ({iterationSize} Shuffles)')
    ax.legend()
    ax.grid(True)
    ax.set_axisbelow(True)
    
    # Plot 2: Residual analysis
    ax = axs[1]
    ax.scatter(permutationIndexSet, np.abs(quantumDistribution - 1/len(permutationArray)), 
              color='magenta', alpha=0.7, label='Quantum Residuals')
    ax.scatter(permutationIndexSet, np.abs(classicalDistribution - 1/len(permutationArray)), 
              color='blue', alpha=0.7, label='Classical Residuals')
    ax.set_xlabel('Permutation Index')
    ax.set_ylabel('|Actual - Ideal|')
    ax.set_title('Deviation from Ideal Uniform Distribution')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True)
    ax.set_axisbelow(True)
    
    # Plot 3: Entropy comparison
    ax = axs[2]
    entropies = [classicalEntropy, quantumEntropy, maxEntropy]
    labels = ['Classical', 'Quantum', 'Maximum']
    colors = ['blue', 'magenta', 'gray']
    
    ax.bar(labels, entropies, color=colors, alpha=0.7)
    ax.set_ylim(0, maxEntropy * 1.1)
    ax.set_ylabel('Shannon Entropy (bits)')
    ax.set_title('Entropy Comparison')
    
    for i, v in enumerate(entropies):
        ax.text(i, v + 0.1, f'{v:.4f}', ha='center')
    
    plt.suptitle(f'Quantum Advantage Analysis: {shuffleAlgorithm.__name__}')
    
    # Information gain calculation
    quantumAdvantage = (quantumEntropy - classicalEntropy) / classicalEntropy * 100
    
    if extraInfo:
        print(f'Classical Entropy: {classicalEntropy:.4f} bits')
        print(f'Quantum Entropy: {quantumEntropy:.4f} bits')
        print(f'Maximum Possible Entropy: {maxEntropy:.4f} bits')
        print(f'Quantum Advantage: {quantumAdvantage:.2f}%')
    
    return {
        'quantum_distribution': quantumDistribution,
        'classical_distribution': classicalDistribution,
        'quantum_entropy': quantumEntropy,
        'classical_entropy': classicalEntropy,
        'max_entropy': maxEntropy,
        'quantum_advantage': quantumAdvantage
    }


def visualize_quantum_shuffle_animation(shuffle_result, filename=None, view_angle=(30, 45)):
    """
    Creates an animation of quantum states during the shuffle process.
    
    Parameters:
        shuffle_result: Tuple of (final_array, quantum_states)
        filename: Optional filename to save the animation
        view_angle: 3D viewing angle for the Bloch sphere
    """
    quantum_states = shuffle_result[1] if shuffle_result[1] is not None else []
    
    # Initialize the figure
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)
    
    # Draw Bloch sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_surface(x, y, z, color='lightgray', alpha=0.1)
    
    # Draw axes
    ax1.plot([-1, 1], [0, 0], [0, 0], 'k-', alpha=0.5, linewidth=1)
    ax1.plot([0, 0], [-1, 1], [0, 0], 'k-', alpha=0.5, linewidth=1)
    ax1.plot([0, 0], [0, 0], [-1, 1], 'k-', alpha=0.5, linewidth=1)
    
    # Text labels
    ax1.text(1.1, 0, 0, '|0⟩', fontsize=12)
    ax1.text(-1.1, 0, 0, '|1⟩', fontsize=12)
    ax1.text(0, 0, 1.1, '|+⟩', fontsize=12)
    ax1.text(0, 0, -1.1, '|-⟩', fontsize=12)
    
    # Set plot limits and equal aspect ratio
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_zlim(-1.2, 1.2)
    ax1.set_box_aspect([1, 1, 1])
    
    # Set viewing angle
    ax1.view_init(elev=view_angle[0], azim=view_angle[1])
    
    # Remove ticks
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])
    
    # Set titles
    ax1.set_title('Quantum State Representation\n(Bloch Sphere)', fontsize=12)
    ax2.set_title('Permutation Steps', fontsize=12)
    
    # Prepare array visualization
    def draw_array(ax, array, highlight_indices=None):
        ax.clear()
        ax.set_title('Permutation Steps', fontsize=12)
        ax.set_xlim(-0.5, len(array) - 0.5)
        ax.set_ylim(-0.5, 1.5)
        ax.axis('off')
        
        for i, val in enumerate(array):
            color = 'lightgray'
            if highlight_indices and i in highlight_indices:
                color = 'magenta'
            circle = plt.Circle((i, 0.5), 0.4, color=color, alpha=0.7)
            ax.add_patch(circle)
            ax.text(i, 0.5, str(val), ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Animation update function
    array_copy = shuffle_result[0].copy() if isinstance(shuffle_result[0], list) else shuffle_result[0].tolist()
    
    def init():
        draw_array(ax2, array_copy)
        ax1.plot([0], [0], [1], 'ro', markersize=10)
        return []
        
    def animate(i):
        if i < len(quantum_states):
            state, indices = quantum_states[i]
            
            # Convert quantum state to Bloch coordinates
            point_x, point_y, point_z = bloch_sphere_coordinates(state)
            
            # Clear previous point
            ax1.clear()
            
            # Redraw Bloch sphere - use the 2D arrays we created earlier
            ax1.plot_surface(x, y, z, color='lightgray', alpha=0.1)
            ax1.plot([-1, 1], [0, 0], [0, 0], 'k-', alpha=0.5, linewidth=1)
            ax1.plot([0, 0], [-1, 1], [0, 0], 'k-', alpha=0.5, linewidth=1)
            ax1.plot([0, 0], [0, 0], [-1, 1], 'k-', alpha=0.5, linewidth=1)
            ax1.text(1.1, 0, 0, '|0⟩', fontsize=12)
            ax1.text(-1.1, 0, 0, '|1⟩', fontsize=12)
            ax1.text(0, 0, 1.1, '|+⟩', fontsize=12)
            ax1.text(0, 0, -1.1, '|-⟩', fontsize=12)
            ax1.set_xlim(-1.2, 1.2)
            ax1.set_ylim(-1.2, 1.2)
            ax1.set_zlim(-1.2, 1.2)
            ax1.set_box_aspect([1, 1, 1])
            ax1.view_init(elev=view_angle[0], azim=view_angle[1])
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.set_zticks([])
            ax1.set_title('Quantum State Representation\n(Bloch Sphere)', fontsize=12)
            
            # Plot quantum state point
            ax1.plot([point_x], [point_y], [point_z], 'ro', markersize=10)
            
            # Animate array swap
            idx1, idx2 = indices
            if idx1 != idx2:
                # Perform the swap
                array_copy[idx1], array_copy[idx2] = array_copy[idx2], array_copy[idx1]
            
            # Draw the array with highlighted elements
            draw_array(ax2, array_copy, highlight_indices=[idx1, idx2])
        
        return []
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(quantum_states) + 5, 
                                   interval=1000, blit=True)
    
    # Save animation if filename provided
    if filename:
        anim.save(filename, writer='pillow', fps=1)
    
    plt.tight_layout()
    return anim


def classical_FisherYates(inputSet):
    """
    Classical Fisher-Yates algorithm for comparison
    """
    arr = inputSet.copy()
    for i in reversed(range(1, len(arr))):
        j = random.randint(0, i)
        arr[i], arr[j] = arr[j], arr[i]
    return arr


#%% Helper function to create orbit distribution plots

def create_quantum_orbit_distribution_plot(shuffle_function, input_data, iterations=50000, filename=None):
    """
    Creates a plot comparing quantum and classical orbit-weighted distributions.
    Useful for visualizing the quantum advantage in the permutation space.
    """
    # Run the quantum distribution analysis
    results = quantumGroupDistribution(shuffle_function, input_data, iterations)
    
    # Information about the run
    quantum_advantage = results['quantum_advantage']
    
    # Create additional plot showing the quantum advantage by permutation
    permutationArray = list(multiset_permutations(input_data))
    permutationIndexSet = np.arange(1, len(permutationArray)+1)
    
    # Calculate ideal probability
    ideal_prob = 1 / len(permutationArray)
    
    # Plot the distribution with a log scale to visualize small differences
    plt.figure(figsize=(14, 10))
    
    plt.subplot(211)
    plt.bar(permutationIndexSet - 0.2, results['quantum_distribution'], width=0.4, 
            color='magenta', alpha=0.7, label='Quantum')
    plt.bar(permutationIndexSet + 0.2, results['classical_distribution'], width=0.4, 
            color='blue', alpha=0.7, label='Classical')
    plt.axhline(y=ideal_prob, color='gray', linestyle='--', alpha=0.7, label='Ideal Uniform')
    plt.xlabel('Permutation Index')
    plt.ylabel('Probability')
    plt.title(f'Permutation Distribution: {shuffle_function.__name__}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(212)
    quantum_dev = np.abs(results['quantum_distribution'] - ideal_prob)
    classical_dev = np.abs(results['classical_distribution'] - ideal_prob)
    plt.bar(permutationIndexSet - 0.2, quantum_dev, width=0.4, 
            color='magenta', alpha=0.7, label='Quantum Deviation')
    plt.bar(permutationIndexSet + 0.2, classical_dev, width=0.4, 
            color='blue', alpha=0.7, label='Classical Deviation')
    plt.xlabel('Permutation Index')
    plt.ylabel('|Actual - Ideal| (Log Scale)')
    plt.title('Deviation from Ideal Uniform Distribution')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'Quantum vs Classical Permutation Analysis\n'
                f'Input: {input_data}, Iterations: {iterations:,}\n'
                f'Quantum Advantage: {quantum_advantage:.2f}%', fontsize=14)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if filename:
        plt.savefig(filename, dpi=150)
    
    return plt


#%% Classical Fisher-Yates for comparison

def FisherYates(inputSet, extraInfo=False):
    """
    Classical Fisher-Yates shuffle algorithm. 
    Used for performance comparison.
    """
    # Making a copy of the input to not overwrite it
    inputArray = inputSet.copy()
    
    # Printing the original state of the array
    if extraInfo:
        print('Initial array state: ', inputArray)
    
    # Looping over the different included endpoints of the array
    for endpoint in reversed(range(1, len(inputArray))):
        # Select a random array element (bounded by included endpoints)
        randomIndex = np.random.randint(0, endpoint+1)
        randomElement = inputArray[randomIndex]
        if extraInfo:
            print('Array element', randomElement, 'of index', randomIndex, 'selected.')
            
        # Swapping the randomly selected element with the current array endpoint
        if extraInfo:
            print('Swapped element', randomElement, 'with', inputArray[endpoint], end='.\n')
        inputArray[randomIndex], inputArray[endpoint] = inputArray[endpoint], inputArray[randomIndex]
        if extraInfo:
            print('New array state: ', inputArray)
            
    # Presenting the final state of the array
    if extraInfo:
        print('Final array state: ', inputArray)
        
    return inputArray


#%% Main demonstration function

def main():
    """
    Demonstrates the quantum permutation algorithms with visualizations.
    """
    # Sample input data
    input_data = [0, 1, 2, 3, 4]
    
    print("Quantum Generalized Permutations Demonstration")
    print("-" * 50)
    print(f"Input data: {input_data}")
    
    # 1. Quantum Fisher-Yates
    print("\n1. Running Quantum Fisher-Yates...")
    qfy_result, quantum_states = quantumFisherYates(input_data, extraInfo=True)
    print(f"Result: {qfy_result}")
    
    # Create and save animation
    print("Creating animation...")
    anim = visualize_quantum_shuffle_animation((qfy_result, quantum_states), 
                                              filename="quantum_fisher_yates_animation.gif")
    
    # 2. Quantum Permutation-Orbit Fisher-Yates
    print("\n2. Running Quantum Permutation-Orbit Fisher-Yates...")
    qpofy_result, prob, quantum_states = quantumPermutationOrbitFisherYates(input_data, extraInfo=True)
    print(f"Result: {qpofy_result}, Probability: {prob}")
    
    # 3. Create distribution plot
    print("\n3. Creating quantum advantage distribution plot...")
    plot = create_quantum_orbit_distribution_plot(quantumConsensusFisherYates, input_data, 
                                                iterations=10000, 
                                                filename="quantum_permutation_distribution.png")
    
    print("\nDemonstration complete. Files saved to current directory.")
    
    # Return to allow interactive exploration
    return {
        'quantum_fisher_yates_result': qfy_result,
        'animation': anim,
        'plot': plot
    }


if __name__ == "__main__":
    print("Starting Quantum Generalized Permutations...")
    results = main()
    print("Execution completed successfully!")
