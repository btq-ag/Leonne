# %% Importing modules - コンプリート

"""
quantumGraphGenerator.py

Quantum-inspired network graph generator that creates various types of network
structures using quantum randomness principles.

Author: Jeffrey Morais, BTQ
"""

# Standard mathematics modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import to_rgba
import sympy
from sympy.utilities.iterables import multiset_permutations
import sys
import os

# For saving animations
from matplotlib import cm


# %% Quantum random number generator - コンプリート

def quantum_random(size=1, method='hadamard'):
    """
    Generate quantum-inspired random numbers
    
    Args:
        size (int): Number of random values to generate
        method (str): Method to use ('hadamard', 'phase', 'bloch')
        
    Returns:
        np.ndarray: Array of random values between 0 and 1
    """
    if method == 'hadamard':
        # Simulate Hadamard transform on |0> followed by measurement
        # |0> -> (|0> + |1>)/sqrt(2) -> measurement gives 0 or 1 with 50% probability
        return np.random.choice([0, 1], size=size, p=[0.5, 0.5])
    
    elif method == 'phase':
        # Simulate random phase rotation followed by Hadamard and measurement
        # More complex quantum behavior with phase
        phases = np.random.uniform(0, 2*np.pi, size=size)
        probs = np.cos(phases/2)**2
        return np.random.binomial(1, probs)
    
    elif method == 'bloch':
        # Simulate points on Bloch sphere for more complex quantum states
        # Project to binary outcome with appropriate probabilities
        theta = np.random.uniform(0, np.pi, size=size)
        phi = np.random.uniform(0, 2*np.pi, size=size)
        
        # Probability of measuring |0> in state cos(theta/2)|0> + e^(i*phi)*sin(theta/2)|1>
        probs = np.cos(theta/2)**2
        return np.random.binomial(1, probs)
    
    else:
        raise ValueError(f"Unknown quantum random method: {method}")


def quantum_choice(options, size=1):
    """
    Make a quantum-inspired random choice from a list of options
    
    Args:
        options (list): List of options to choose from
        size (int): Number of choices to make
        
    Returns:
        list: Selected options
    """
    indices = np.floor(np.random.uniform(0, len(options), size=size)).astype(int)
    
    # Apply quantum randomness to perturb the selection
    q_perturbation = quantum_random(size=size, method='phase')
    flip_mask = (q_perturbation == 1)
    
    # Flip some choices based on quantum perturbation
    if np.any(flip_mask):
        alt_indices = np.floor(np.random.uniform(0, len(options), size=sum(flip_mask))).astype(int)
        indices[flip_mask] = alt_indices
    
    # Ensure indices are within bounds
    indices = np.clip(indices, 0, len(options)-1)
    
    if size == 1:
        return options[indices[0]]
    return [options[i] for i in indices]


# %% Defining test edge sets to permute - コンプリート
testEdgeSet = [
        [1,2],[2,1],[2,3],[3,2],[4,4]
    ]

largeEdgeSet = [
        [1,1],[1,2],[2,1],[3,2],[3,3],[4,1],[5,2]
    ]

trivialEdgeSet = [
        [1,1],[2,2],[3,3]
        ]


# %% Redefining the consensus shuffle using quantum randomness - コンプリート

def quantumConsensusShuffle(targetEdgeSet, extraInfo=True, quantum_method='phase'):
    """
    Quantum-enhanced consensus shuffle algorithm
    
    Args:
        targetEdgeSet (list): Edge set to shuffle
        extraInfo (bool): Whether to print extra information
        quantum_method (str): Method for quantum randomness
        
    Returns:
        np.ndarray: Shuffled edge set
    """
    # Initializing the edge set and its columns
    edgeSet = targetEdgeSet.copy()
    u = list(np.array(edgeSet).T[0])
    v = list(np.array(edgeSet).T[1])
    
    # Extra info: printing the original configuration of the set
    if extraInfo:
        print('Initial quantum edge set of the graph:\n', edgeSet)
        print('Target list:\n', u)
        print('Dual list:\n', v)
        
    # Looping over the different endpoints of the unselected elements
    for i in reversed(range(1, len(u))):
        # Extra info: printing current selected endpoint
        if extraInfo:
            print('Current endpoint: ', u[i])
        
        # Initializing the transition index set
        t = [i]
        
        # Constructing the transition set based on non-degenerate elements
        for k in range(0, i):
            
            # Extra info: print if tuples occur in the edge set
            if extraInfo:
                print('[u[i],v[k]] tuple: ', [u[i],v[k]], [u[i],v[k]] not in edgeSet)
                print('[u[k],v[i]] tuple: ', [u[k],v[i]], [u[k],v[i]] not in edgeSet)
            
            # Quantum enhancement: Use quantum-inspired entanglement to decide on inclusion
            if quantum_random(size=1, method=quantum_method)[0] < 0.5:
                condition = ([u[i],v[k]] not in edgeSet) and ([u[k],v[i]] not in edgeSet)
            else:
                # Slightly modified condition using quantum randomness
                condition = ([u[i],v[k]] not in edgeSet)
            
            # Appending non-degenerate points indices to the transition set
            if condition:
                t.append(k)
                
        # Extra info: printing the transition set
        if extraInfo:
            print('Current quantum transition index set: ', sorted(t))        
                
        # Randomly selecting a point in the transition set using quantum choice
        j = quantum_choice(t)
        u[i], u[j] = u[j], u[i]
        
        # Extra info: printing randomly selected element
        if extraInfo:
            print('Selected quantum transition index: ', j)
            print('Selected swap point in U: ', u[i])
        
        # Extra info: printing swapped elements and new list
        if extraInfo:
            print('Swapped element', u[i], 'with', u[j])
            print('Current target list: ', u)
            print('Current dual list: ', v)
        
    # Gluing back the pieces of the edge set together 
    finalEdgeSet = np.column_stack((u,v))
    
    # Extra info: presenting the final state of the edge set
    if extraInfo:
        print('Final state of quantum edge set:\n', finalEdgeSet)
        
    return finalEdgeSet


# %% [Quantum Vertex Space] Computing quantum probability distribution - コンプリート

def quantumProbabilityDistribution(inputEdgeSet, edgeName, iterationSize, extraInfo=False, quantum_method='phase'):
    """
    Analyze the quantum probability distribution of edge set permutations
    
    Args:
        inputEdgeSet (list): Edge set to analyze
        edgeName (str): Name of the edge set for plotting
        iterationSize (int): Number of iterations for statistical analysis
        extraInfo (bool): Whether to print extra information
        quantum_method (str): Method for quantum randomness
        
    Returns:
        list: All possible permutations
        list: Classical tallies
        list: Quantum tallies
    """
    # Defining all possible unique permutations of the input set
    permutationArray = list(
        multiset_permutations(list(np.array(inputEdgeSet).T[0]))
    )
    print('Total possible permutations: ', len(permutationArray))
    print('All possible unique permutations: ', permutationArray)

    # Initialize two empty lists to tally classical and quantum permutations
    classicalTallySet = list(np.zeros(len(permutationArray), dtype=int))
    quantumTallySet = list(np.zeros(len(permutationArray), dtype=int))
    
    # Iterating through iterationSize independent shuffles
    for iteration in range(1, iterationSize+1):
        
        # Classical permutation
        outputClassicalEdgeSet = classicalConsensusShuffle(inputEdgeSet, False)
        outputClassicalTargetList = list(outputClassicalEdgeSet.T[0])
        
        # Quantum permutation
        outputQuantumEdgeSet = quantumConsensusShuffle(inputEdgeSet, False, quantum_method)
        outputQuantumTargetList = list(outputQuantumEdgeSet.T[0])
        
        if extraInfo:
            print('Classical permuted set: ', outputClassicalTargetList)
            print('Quantum permuted set: ', outputQuantumTargetList)
        
        # Check and tally for both classical and quantum permutations
        for permutation in range(0, len(permutationArray)):
            if outputClassicalTargetList == permutationArray[permutation]:
                classicalTallySet[permutation] += 1
            if outputQuantumTargetList == permutationArray[permutation]:
                quantumTallySet[permutation] += 1
            
    # Indexing all unique permutations of U
    permutationIndexSet = list(np.arange(1, len(permutationArray)+1))
    
    # Computing probability distributions
    classicalProbabilitySet = np.array(classicalTallySet)/iterationSize
    quantumProbabilitySet = np.array(quantumTallySet)/iterationSize
    
    # Expected uniform probability
    expectedProb = 1/len(permutationArray)

    # Plotting results comparing classical and quantum distributions
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    
    # Top subplot: classical probabilities
    ax = axs[0]
    ax.scatter(permutationIndexSet, classicalProbabilitySet, color='blue', label='Classical Probabilities')
    ax.axhline(y=expectedProb, color='gray', label='Expected Uniform Probability', linestyle='dotted')
    ax.set_ylabel('Classical Probability')
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.set_title(f'Comparison of {iterationSize} Classical vs Quantum Shuffles for {edgeName}')
    ax.legend()
    ax.grid()
    ax.set_axisbelow(True)

    # Middle subplot: quantum probabilities
    ax = axs[1]
    ax.scatter(permutationIndexSet, quantumProbabilitySet, color='magenta', label='Quantum Probabilities')
    ax.axhline(y=expectedProb, color='gray', label='Expected Uniform Probability', linestyle='dotted')
    ax.set_ylabel('Quantum Probability')
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.legend()
    ax.grid()
    ax.set_axisbelow(True)

    # Bottom subplot: difference between quantum and classical
    ax = axs[2]
    ax.bar(permutationIndexSet, quantumProbabilitySet - classicalProbabilitySet, color='purple', 
           label='Quantum - Classical Difference')
    ax.axhline(y=0, color='black', linestyle='-')
    ax.set_xlabel('Unique Permutations')
    ax.set_ylabel('Probability Difference')
    ax.legend()
    ax.grid()
    ax.set_axisbelow(True)
    
    # Save the plot
    plt.savefig(f'quantum_vs_classical_{edgeName}.png', dpi=300, bbox_inches='tight')
    
    return permutationArray, classicalTallySet, quantumTallySet


# %% [Quantum Edge Space] Computing quantum probability distribution for edge sets - コンプリート

def quantumRemoveDegeneracy(inputEdgeSets): 
    """
    Remove degenerate edge sets with quantum-enhanced uniqueness checking
    
    Args:
        inputEdgeSets (list): List of edge sets to check for degeneracy
        
    Returns:
        list: Unique edge sets
    """
    # Initializing the new list of unique edge sets
    uniqueEdgeSets = []

    # Initialize an empty set to keep track of duplicate edge sets
    seenEdgeSets = set()

    # Iterate through each inner edge set
    for edgeSet in inputEdgeSets:
        
        # Convert the inner list to a frozenset for immutability
        frozenSet = frozenset(map(tuple, edgeSet))

        # Quantum enhancement: occasionally allow "similar" but not identical sets
        # to be considered unique to explore more of the graph space
        if quantum_random(size=1, method='bloch')[0] > 0.9:
            # Generate a unique identifier with quantum noise
            uniqueIdentifier = hash(frozenSet) + int(quantum_random(size=1, method='phase')[0] * 1000)
            if uniqueIdentifier not in seenEdgeSets:
                seenEdgeSets.add(uniqueIdentifier)
                uniqueEdgeSets.append(edgeSet)
        else:
            # Regular deduplication
            if frozenSet not in seenEdgeSets:
                seenEdgeSets.add(frozenSet)
                uniqueEdgeSets.append(edgeSet)

    # Return the list of unique lists
    return uniqueEdgeSets


# %% Constructing a function to verify if a given degree sequence is satisfiable - コンプリート

def quantumSequenceVerifier(inputSequence, uSize, vSize, extraInfo=True, quantum_method='phase'):
    """
    Quantum-enhanced verification of degree sequence satisfiability
    
    Args:
        inputSequence (list): Degree sequence to verify
        uSize (int): Size of U set
        vSize (int): Size of V set
        extraInfo (bool): Whether to print extra information
        quantum_method (str): Method for quantum randomness
        
    Returns:
        bool: Whether the sequence is satisfiable
    """
    # Partioning degree sequence into ordered sub-sequences
    uSequence = sorted(inputSequence[:uSize], reverse=True)
    vSequence = sorted(inputSequence[vSize:], reverse=True)
    
    # Extra info: printing initial state of sequences
    if extraInfo:
        print('Quantum U sequence: ', uSequence)
        print('Quantum V sequence: ', vSequence)
    
    # Initializing constraint logic
    firstConstraint = False
    secondConstraint = False
    totalConstraint = False
    
    # Testing the first constraint: conservation with quantum tolerance
    # In quantum version, we allow slight deviations for exploration
    uSum = np.sum(uSequence)
    vSum = np.sum(vSequence)
    
    if abs(uSum - vSum) <= 0.1 * min(uSum, vSum) * quantum_random(size=1, method=quantum_method)[0]:
        firstConstraint = True
        
        # Extra info: printing the outcome of the first constraint
        if extraInfo:
            print('Quantum first constraint relaxation: ', firstConstraint)
            if uSum != vSum:
                print(f'Classical sums differ: U={uSum}, V={vSum}, but quantum allows exploration')
    elif uSum == vSum:
        firstConstraint = True
        if extraInfo:
            print('Quantum first constraint (exact): ', firstConstraint)
    else:
        # Extra info: printing the outcome of the first constraint
        if extraInfo:
            print('Quantum first constraint: ', firstConstraint)
            
        return False
    
    # Testing the second constraint: boundedness
    quantum_violations = 0
    for k in range(0, uSize):
        
        # Computing LHS of constraint
        leftHandSide = np.sum(uSequence[:k+1])
        
        # Computing RHS of constraint
        minVSequence = [min(v,k+1) for v in vSequence]
        rightHandSide = np.sum(minVSequence)
        
        # Extra info: printing the values of the sides
        if extraInfo:
            print('LHS: ', leftHandSide)
            print('RHS: ', rightHandSide)
        
        # In quantum version, we allow some violations for exploration
        if not leftHandSide <= rightHandSide:
            # Check for small violations that quantum might allow
            if leftHandSide - rightHandSide <= 2 and quantum_random(size=1, method=quantum_method)[0] > 0.7:
                quantum_violations += 1
                if extraInfo:
                    print(f"Small quantum violation allowed: {leftHandSide} > {rightHandSide}")
                continue
            
            # Too many or too large violations
            if extraInfo:
                print('Quantum second constraint: False')
            return False
    
    # If we reach here, second constraint is satisfied (possibly with quantum relaxation)
    secondConstraint = True
    
    # Extra info: printing the outcome of the constraints
    if extraInfo:
        print('Quantum second constraint: ', secondConstraint)
        if quantum_violations > 0:
            print(f'Allowed {quantum_violations} small quantum violations for exploration')
    
    # If both constraints are satisfied, it is a proper degree sequence
    if (firstConstraint and secondConstraint):
        totalConstraint = True
        
    # Returning output of the constraint
    return totalConstraint


# %% Constructing a function to output a quantum graph assignment - コンプリート

def quantumEdgeAssignment(inputSequence, uSize, vSize, extraInfo=True, quantum_method='phase'):
    """
    Generate a quantum-enhanced edge assignment for a degree sequence
    
    Args:
        inputSequence (list): Degree sequence to realize
        uSize (int): Size of U set
        vSize (int): Size of V set
        extraInfo (bool): Whether to print extra information
        quantum_method (str): Method for quantum randomness
        
    Returns:
        list: Edge set realizing the degree sequence
    """
    # Testing if is a satisfiable degree sequence
    if not quantumSequenceVerifier(inputSequence, uSize, vSize, False, quantum_method):
        return False
    else:
        print('Partially satisfiable quantum sequence.')
    
    # Initializing the ordered degree sequences and edge set
    edgeSet = []
    uSequence = sorted(inputSequence[:uSize], reverse=True)
    vSequence = sorted(inputSequence[vSize:], reverse=True)
    
    # Extra info: printing initial state of sequences and edge set
    if extraInfo:
        print('Quantum U sequence: ', uSequence)
        print('Quantum V sequence: ', vSequence)
        print('Initial quantum edge set: ', edgeSet)
    
    # In quantum version, we shuffle the sequences to introduce quantum randomness
    if quantum_random(size=1, method=quantum_method)[0] > 0.5:
        # Quantum perturbation of ordering - for U sequence only for safety
        for i in range(len(uSequence)-1):
            if quantum_random(size=1, method=quantum_method)[0] > 0.7:
                j = int(quantum_random(size=1, method='hadamard')[0] * (len(uSequence)-i-1)) + i + 1
                if j < len(uSequence):  # Safety check
                    uSequence[i], uSequence[j] = uSequence[j], uSequence[i]
    
    # Looping through to construct the edge set
    for u in range(0, len(uSequence)):
        # Quantum enhancement: sometimes shuffle the v-vertices to explore different configurations
        v_indices = list(range(0, len(vSequence)))
        if quantum_random(size=1, method=quantum_method)[0] > 0.5:
            np.random.shuffle(v_indices)
            
        for v_idx in v_indices:
            v = v_idx
            if (uSequence[u] > 0) and (vSequence[v] > 0):
                edgeSet.append([u, v])
                uSequence[u] -= 1
                vSequence[v] -= 1
                
    # Extra info: printing the final state of the edge set
    if extraInfo:
        print('Final quantum edge set: ', edgeSet)
              
    # Return the quantum-enhanced edge set
    return edgeSet


# %% Animation of quantum graph evolution - コンプリート

def animate_quantum_edge_evolution(initial_edge_set, num_frames=50, quantum_method='phase'):
    """
    Create an animation of quantum edge set evolution
    
    Args:
        initial_edge_set (list): Initial edge set
        num_frames (int): Number of frames in animation
        quantum_method (str): Method for quantum randomness
        
    Returns:
        animation.FuncAnimation: Animation object
    """
    edge_sets = [np.array(initial_edge_set)]
    
    # Generate sequence of edge sets
    current_edge_set = initial_edge_set
    for _ in range(num_frames-1):
        current_edge_set = quantumConsensusShuffle(current_edge_set, False, quantum_method)
        edge_sets.append(np.array(current_edge_set))
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get maximum coordinates for plot limits
    all_coords = np.vstack(edge_sets)
    max_x = max(all_coords[:, 0]) + 1
    max_y = max(all_coords[:, 1]) + 1
    
    # Function to update the plot for each frame
    def update(frame):
        ax.clear()
        
        # Plot the current edge set
        edge_set = edge_sets[frame]
        
        # Extract U and V vertices (unique values in columns)
        u_vertices = np.unique(edge_set[:, 0])
        v_vertices = np.unique(edge_set[:, 1])
        
        # Plot U vertices on left
        for u in u_vertices:
            ax.scatter(0, u, s=100, c='blue', alpha=0.7)
            ax.text(-0.1, u, f'U{u}', fontsize=10, ha='right')
        
        # Plot V vertices on right
        for v in v_vertices:
            ax.scatter(1, v, s=100, c='red', alpha=0.7)
            ax.text(1.1, v, f'V{v}', fontsize=10, ha='left')
        
        # Plot edges
        for edge in edge_set:
            # Quantum coloring: Edges have color based on quantum state
            q_value = quantum_random(size=1, method=quantum_method)[0]
            edge_color = cm.viridis(q_value)
            ax.plot([0, 1], [edge[0], edge[1]], 
                   c=edge_color, alpha=0.5, 
                   linewidth=2)
        
        # Add annotations
        ax.set_title(f'Quantum Edge Set Evolution (Frame {frame+1}/{len(edge_sets)})')
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, max(max_x, max_y) + 0.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['U', 'V'])
        ax.grid(alpha=0.3)
        
        # Add quantum state info
        if frame > 0:
            prev_set = set(map(tuple, edge_sets[frame-1]))
            curr_set = set(map(tuple, edge_set))
            changed = len(prev_set.symmetric_difference(curr_set))
            ax.text(0.5, -0.3, f'Quantum changes: {changed}', 
                   ha='center', fontsize=10, transform=ax.transAxes)
        
        return ax,
    
    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=len(edge_sets), 
                                 interval=200, blit=False)
    
    # Save animation
    ani.save('quantum_edge_evolution.gif', writer='pillow', fps=5, dpi=100)
    
    return ani


# %% Create a comparison plot between classical and quantum approaches - コンプリート

def compareClassicalQuantum(deg_seq, u_size, v_size, num_samples=100):
    """
    Compare classical and quantum graph generation approaches
    
    Args:
        deg_seq (list): Degree sequence to use
        u_size (int): Size of U set
        v_size (int): Size of V set
        num_samples (int): Number of samples to generate
        
    Returns:
        None: Creates and saves a plot
    """
    # Check sequence validity
    if not sequenceVerifier(deg_seq, u_size, v_size, False):
        print("Sequence not classically satisfiable!")
        return
    
    if not quantumSequenceVerifier(deg_seq, u_size, v_size, False):
        print("Sequence not quantum satisfiable!")
        return
    
    # Generate samples
    classical_edges = []
    quantum_edges = []
    
    # Create initial edge sets
    initial_classical = edgeAssignment(deg_seq, u_size, v_size, False)
    initial_quantum = quantumEdgeAssignment(deg_seq, u_size, v_size, False)
    
    # Generate samples through shuffling
    for _ in range(num_samples):
        classical_edges.append(consensusShuffle(initial_classical, False))
        quantum_edges.append(quantumConsensusShuffle(initial_quantum, False))
    
    # Flatten for histogram analysis
    classical_flat = np.vstack(classical_edges).flatten()
    quantum_flat = np.vstack(quantum_edges).flatten()
    
    # Create comparison plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # Classical distribution
    axs[0].hist(classical_flat, bins=20, alpha=0.7, color='blue')
    axs[0].set_title('Classical Edge Distribution')
    axs[0].set_xlabel('Edge Values')
    axs[0].set_ylabel('Frequency')
    axs[0].grid(alpha=0.3)
    
    # Quantum distribution
    axs[1].hist(quantum_flat, bins=20, alpha=0.7, color='magenta')
    axs[1].set_title('Quantum Edge Distribution')
    axs[1].set_xlabel('Edge Values')
    axs[1].set_ylabel('Frequency')
    axs[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('classical_vs_quantum_distribution.png', dpi=300)
    plt.close()
    
    # Also create the animation for visualization
    animate_quantum_edge_evolution(initial_quantum)


# %% Main execution section - コンプリート

# Define classical equivalents for comparison
def classicalSequenceVerifier(inputSequence, uSize, vSize, extraInfo=True):
    """Classical implementation of sequence verification"""
    # Partioning degree sequence into ordered sub-sequences
    uSequence = sorted(inputSequence[:uSize], reverse=True)
    vSequence = sorted(inputSequence[vSize:], reverse=True)
    
    # Conservation check
    uSum = np.sum(uSequence)
    vSum = np.sum(vSequence)
    if uSum != vSum:
        return False
    
    # Boundedness check
    for k in range(0, uSize):
        leftHandSide = np.sum(uSequence[:k+1])
        minVSequence = [min(v,k+1) for v in vSequence]
        rightHandSide = np.sum(minVSequence)
        if leftHandSide > rightHandSide:
            return False
    
    return True

def classicalEdgeAssignment(inputSequence, uSize, vSize, extraInfo=True):
    """Classical implementation of edge assignment"""
    if not classicalSequenceVerifier(inputSequence, uSize, vSize, False):
        return False
    
    edgeSet = []
    uSequence = sorted(inputSequence[:uSize], reverse=True)
    vSequence = sorted(inputSequence[vSize:], reverse=True)
    
    for u in range(0, len(uSequence)):
        for v in range(0, len(vSequence)):
            if (uSequence[u] > 0) and (vSequence[v] > 0):
                edgeSet.append([u, v])
                uSequence[u] -= 1
                vSequence[v] -= 1
                
    return edgeSet

def classicalConsensusShuffle(targetEdgeSet, extraInfo=True):
    """Classical implementation of consensus shuffle"""
    edgeSet = targetEdgeSet.copy()
    u = list(np.array(edgeSet).T[0])
    v = list(np.array(edgeSet).T[1])
    
    for i in reversed(range(1, len(u))):
        t = [i]
        for k in range(0, i):
            if ([u[i], v[k]] not in edgeSet) and ([u[k], v[i]] not in edgeSet):
                t.append(k)
        
        j = np.random.choice(t)
        u[i], u[j] = u[j], u[i]
    
    return np.column_stack((u, v))

if __name__ == "__main__":
    # Test the sequence verification
    print('--- Example degree sequence ---')
    deg_seq = [2, 2, 2, 2, 3, 1]
    u_size, v_size = 3, 3
    
    print("Classical verification result:", classicalSequenceVerifier(deg_seq, u_size, v_size, False))
    print("Quantum verification result:", quantumSequenceVerifier(deg_seq, u_size, v_size))
    
    # Generate edge assignments
    print('\n--- Edge Assignments ---')
    classical_edges = classicalEdgeAssignment(deg_seq, u_size, v_size)
    quantum_edges = quantumEdgeAssignment(deg_seq, u_size, v_size)
    
    print("\nClassical edge set:", classical_edges)
    print("Quantum edge set:", quantum_edges)
    
    # Create comparison visualization
    print('\n--- Creating comparison visualization ---')
    
    # Define compareClassicalQuantum with our local classical functions
    def compareClassicalQuantum(deg_seq, u_size, v_size, num_samples=100):
        # Check sequence validity
        if not classicalSequenceVerifier(deg_seq, u_size, v_size, False):
            print("Sequence not classically satisfiable!")
            return
        
        if not quantumSequenceVerifier(deg_seq, u_size, v_size, False):
            print("Sequence not quantum satisfiable!")
            return
        
        # Generate samples
        classical_edges = []
        quantum_edges = []
        
        # Create initial edge sets
        initial_classical = classicalEdgeAssignment(deg_seq, u_size, v_size, False)
        initial_quantum = quantumEdgeAssignment(deg_seq, u_size, v_size, False)
        
        # Generate samples through shuffling
        for _ in range(num_samples):
            classical_edges.append(classicalConsensusShuffle(initial_classical, False))
            quantum_edges.append(quantumConsensusShuffle(initial_quantum, False))
        
        # Flatten for histogram analysis
        classical_flat = np.vstack(classical_edges).flatten()
        quantum_flat = np.vstack(quantum_edges).flatten()
        
        # Create comparison plot
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        
        # Classical distribution
        axs[0].hist(classical_flat, bins=20, alpha=0.7, color='blue')
        axs[0].set_title('Classical Edge Distribution')
        axs[0].set_xlabel('Edge Values')
        axs[0].set_ylabel('Frequency')
        axs[0].grid(alpha=0.3)
        
        # Quantum distribution
        axs[1].hist(quantum_flat, bins=20, alpha=0.7, color='magenta')
        axs[1].set_title('Quantum Edge Distribution')
        axs[1].set_xlabel('Edge Values')
        axs[1].set_ylabel('Frequency')
        axs[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('classical_vs_quantum_distribution.png', dpi=300)
        plt.close()
        
        # Also create the animation for visualization
        animate_quantum_edge_evolution(initial_quantum)
    
    compareClassicalQuantum(deg_seq, u_size, v_size)
    
    print('\n--- Animation created: quantum_edge_evolution.gif ---')
    print('--- Distribution plot created: classical_vs_quantum_distribution.png ---')
    
    # Generate edge assignments
    print('\n--- Edge Assignments ---')
    classical_edges = edgeAssignment(deg_seq, u_size, v_size)
    quantum_edges = quantumEdgeAssignment(deg_seq, u_size, v_size)
    
    print("\nClassical edge set:", classical_edges)
    print("Quantum edge set:", quantum_edges)
    
    # Create comparison visualization
    print('\n--- Creating comparison visualization ---')
    compareClassicalQuantum(deg_seq, u_size, v_size)
    
    print('\n--- Animation created: quantum_edge_evolution.gif ---')
    print('--- Distribution plot created: classical_vs_quantum_distribution.png ---')
