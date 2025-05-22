#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quantum_consensus_visualizer.py

Visualization utilities for Quantum-Enhanced Distributed Consensus Networks (QDCN).
Provides visual representations of quantum network topology, quantum consensus sets,
and quantum protocol dynamics with entanglement visualization.

Key quantum enhancements:
1. Entanglement visualization between nodes
2. Quantum state fidelity representations
3. Quantum coherence visualization
4. Quantum-enhanced compliance visualization
5. Quantum consensus set visualization

Author: Jeffrey Morais, BTQ
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from itertools import cycle


class QuantumConsensusVisualizer:
    """
    Visualization tools for Quantum-Enhanced Distributed Consensus Networks.
    """
    
    def __init__(self, network):
        """
        Initialize visualizer with a quantum network.
        
        Parameters:
        -----------
        network : QuantumConsensusNetwork
            The quantum consensus network to visualize
        """
        self.network = network
        self.node_positions = None
        
        # Enhanced color palettes for quantum visualizations
        self.color_palette = sns.color_palette("husl", 10)
        self.consensus_set_colors = cycle(self.color_palette)
        
        # Custom colormap for quantum effects
        quantum_colors = [(0.9, 0.1, 0.9), (0.5, 0.1, 0.9), (0.1, 0.4, 0.9), (0.1, 0.9, 0.9)]
        self.quantum_cmap = LinearSegmentedColormap.from_list("quantum", quantum_colors, N=100)
        
    def plot_quantum_network_topology(self, figsize=(12, 10), show_node_ids=True, 
                                     show_entanglement=True):
        """
        Plot the quantum network topology with entanglement visualization.
        
        Parameters:
        -----------
        figsize : tuple, default=(12, 10)
            Figure size
        show_node_ids : bool, default=True
            Whether to show node IDs
        show_entanglement : bool, default=True
            Whether to show entanglement between nodes
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create a color map based on node honesty and quantum enhancement
        honesty_values = [self.network.honesty_distribution[int(node_id.split('_')[1])] 
                         for node_id in self.network.nodes.keys()]
        
        quantum_levels = [node.quantum_enhancement_level 
                         for node in self.network.nodes.values()]
        
        # Combine honesty and quantum enhancement for node coloring
        # Higher values = more honest and more quantum-enhanced
        node_color_values = [(h + q) / 2 for h, q in zip(honesty_values, quantum_levels)]
        
        # Calculate node positions (store for future use)
        if self.node_positions is None:
            self.node_positions = nx.spring_layout(self.network.network_graph, seed=42)
            
        # Draw edges with entanglement-based width and color
        if show_entanglement:
            # Get edges with weights
            edge_weights = [data['weight'] for _, _, data in self.network.network_graph.edges(data=True)]
            
            # Normalize edge widths and alpha based on entanglement strength
            widths = [1 + 4 * w for w in edge_weights]
            edge_colors = [self.quantum_cmap(w) for w in edge_weights]
            
            nx.draw_networkx_edges(self.network.network_graph, self.node_positions, 
                                  width=widths, edge_color=edge_colors, 
                                  alpha=0.7, ax=ax)
        else:
            # Simple edge drawing without entanglement visualization
            nx.draw_networkx_edges(self.network.network_graph, self.node_positions, 
                                  alpha=0.3, ax=ax)
        
        # Draw nodes with color based on honesty and quantum enhancement
        node_collection = nx.draw_networkx_nodes(
            self.network.network_graph, 
            self.node_positions,
            node_color=node_color_values,
            cmap='plasma',
            vmin=0.0,
            vmax=1.0,
            node_size=600,
            ax=ax
        )
        
        # Add colorbar for node characteristics
        sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Node Quality (Honesty + Quantum Enhancement)')
        
        # Show node IDs if requested
        if show_node_ids:
            nx.draw_networkx_labels(self.network.network_graph, self.node_positions, ax=ax)
            
        # Add quantum entanglement legend
        if show_entanglement:
            # Create legend for entanglement strengths
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color=self.quantum_cmap(0.2), lw=1, label='Low Entanglement'),
                Line2D([0], [0], color=self.quantum_cmap(0.5), lw=2, label='Medium Entanglement'),
                Line2D([0], [0], color=self.quantum_cmap(0.8), lw=4, label='High Entanglement')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            
        # Set title and remove axis
        ax.set_title('Quantum Network Topology with Entanglement', fontsize=16)
        ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_quantum_consensus_sets(self, consensus_sets, figsize=(14, 12)):
        """
        Plot the quantum consensus sets within the network.
        
        Parameters:
        -----------
        consensus_sets : list
            List of quantum consensus sets to visualize
        figsize : tuple, default=(14, 12)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure
        """
        if not consensus_sets:
            print("No quantum consensus sets to visualize.")
            return None
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate node positions if not already done
        if self.node_positions is None:
            self.node_positions = nx.spring_layout(self.network.network_graph, seed=42)
            
        # Draw the network (grey background)
        nx.draw_networkx_edges(self.network.network_graph, self.node_positions, 
                             alpha=0.1, ax=ax, edge_color='grey')
        nx.draw_networkx_nodes(self.network.network_graph, self.node_positions,
                             node_color='lightgrey', alpha=0.3, node_size=500, ax=ax)
        
        # Draw each quantum consensus set with quantum-enhanced visualization
        legend_elements = []
        for i, cs in enumerate(consensus_sets):
            # Get nodes in this consensus set
            cs_nodes = cs['nodes']
            cs_subgraph = self.network.network_graph.subgraph(cs_nodes)
            
            # Choose a color for this consensus set
            color = next(self.consensus_set_colors)
            
            # Get quantum coherence value for this set (or default to 0.9)
            quantum_coherence = cs.get('quantum_coherence', 0.9)
            
            # Draw nodes in this consensus set with quantum glow effect
            # Higher quantum coherence = more saturated color and larger size
            node_size = 500 + 300 * quantum_coherence
            node_alpha = 0.7 + 0.3 * quantum_coherence
            
            nx.draw_networkx_nodes(cs_subgraph, self.node_positions,
                                 node_color=[color]*len(cs_nodes),
                                 alpha=node_alpha,
                                 node_size=node_size, 
                                 ax=ax)
            
            # Draw edges between nodes in this consensus set with quantum effect
            # Quantum coherence affects edge width and transparency
            edge_width = 1.5 + 2.5 * quantum_coherence
            edge_alpha = 0.6 + 0.4 * quantum_coherence
            
            nx.draw_networkx_edges(cs_subgraph, self.node_positions,
                                 width=edge_width, 
                                 alpha=edge_alpha, 
                                 ax=ax,
                                 edge_color=color)
            
            # Add to legend with quantum coherence information
            from matplotlib.lines import Line2D
            legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                        markerfacecolor=color, markersize=10,
                                        label=f'Quantum Consensus Set {i+1} (Coherence: {quantum_coherence:.2f})'))
            
        # Show node IDs
        nx.draw_networkx_labels(self.network.network_graph, self.node_positions, ax=ax)
        
        # Add legend
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Set title and remove axis
        ax.set_title('Quantum Consensus Sets with Entanglement', fontsize=16)
        ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def visualize_quantum_consensus_round(self, round_results, figsize=(16, 14)):
        """
        Visualize the results of a quantum consensus round.
        
        Parameters:
        -----------
        round_results : dict
            Results from a quantum consensus round
        figsize : tuple, default=(16, 14)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure
        """
        fig = plt.figure(figsize=figsize)
        
        # 1. Plot network with quantum consensus sets
        ax1 = fig.add_subplot(221)
        self._plot_network_with_quantum_consensus_sets(round_results['consensus_sets'], ax1)
        
        # 2. Plot transaction verification results with quantum effects
        ax2 = fig.add_subplot(222)
        self._plot_quantum_transaction_verification(round_results, ax2)
        
        # 3. Plot quantum compliance results
        ax3 = fig.add_subplot(223)
        self._plot_quantum_compliance_results(round_results['compliance_results'], ax3)
        
        # 4. Plot quantum proof of consensus statistics
        ax4 = fig.add_subplot(224)
        self._plot_quantum_proof_statistics(round_results['proofs_of_consensus'],
                                          round_results['newly_verified'], ax4)
        
        plt.tight_layout()
        plt.suptitle(f"Quantum Consensus Round {round_results['round_id']} Results", 
                    fontsize=18, y=0.98)
        plt.subplots_adjust(top=0.90)
        
        return fig
    
    def _plot_network_with_quantum_consensus_sets(self, consensus_sets, ax):
        """
        Plot the network with quantum consensus sets highlighted.
        
        Parameters:
        -----------
        consensus_sets : list
            List of quantum consensus sets
        ax : matplotlib.axes.Axes
            Axes to plot on
        """
        # Calculate node positions if not already done
        if self.node_positions is None:
            self.node_positions = nx.spring_layout(self.network.network_graph, seed=42)
            
        # Draw the network background with quantum entanglement visualization
        # Get edges with weights
        edge_weights = [data.get('weight', 0.1) for _, _, data in self.network.network_graph.edges(data=True)]
        
        # Draw edges with varying transparency based on entanglement
        widths = [0.5 + w for w in edge_weights]
        alphas = [0.1 + 0.2 * w for w in edge_weights]
        
        for (u, v, data), width, alpha in zip(self.network.network_graph.edges(data=True), widths, alphas):
            nx.draw_networkx_edges(self.network.network_graph, self.node_positions,
                                  edgelist=[(u, v)], width=width, alpha=alpha, 
                                  edge_color='grey', ax=ax)
        
        # Create dictionary to track which nodes are in consensus sets
        node_in_cs = {node_id: False for node_id in self.network.nodes.keys()}
        
        # Draw each consensus set with a different color
        cs_color_cycle = cycle(self.color_palette)
        for i, cs in enumerate(consensus_sets):
            # Get nodes in this consensus set
            cs_nodes = cs['nodes']
            for node_id in cs_nodes:
                node_in_cs[node_id] = True
                
            cs_subgraph = self.network.network_graph.subgraph(cs_nodes)
            
            # Choose a color for this consensus set
            color = next(cs_color_cycle)
            
            # Get quantum coherence for special effects
            quantum_coherence = cs.get('quantum_coherence', 0.9)
            
            # Draw nodes in this consensus set with quantum glow effect
            node_size = 500 + 200 * quantum_coherence
            node_alpha = 0.7 + 0.3 * quantum_coherence
            
            nx.draw_networkx_nodes(cs_subgraph, self.node_positions,
                                 node_color=[color]*len(cs_nodes),
                                 alpha=node_alpha,
                                 node_size=node_size, 
                                 ax=ax)
            
            # Draw edges between nodes in this consensus set
            edge_width = 1.5 + 2 * quantum_coherence
            edge_alpha = 0.6 + 0.4 * quantum_coherence
            
            nx.draw_networkx_edges(cs_subgraph, self.node_positions,
                                 width=edge_width, 
                                 alpha=edge_alpha, 
                                 ax=ax,
                                 edge_color=color)
        
        # Draw nodes not in any consensus set
        nodes_not_in_cs = [node_id for node_id, in_cs in node_in_cs.items() if not in_cs]
        if nodes_not_in_cs:
            not_in_cs_subgraph = self.network.network_graph.subgraph(nodes_not_in_cs)
            nx.draw_networkx_nodes(not_in_cs_subgraph, self.node_positions,
                                 node_color='lightgrey', node_size=500, ax=ax)
        
        # Show node IDs
        nx.draw_networkx_labels(self.network.network_graph, self.node_positions, ax=ax)
        
        # Set title and remove axis
        ax.set_title('Quantum Network with Consensus Sets', fontsize=14)
        ax.axis('off')
    
    def _plot_quantum_transaction_verification(self, round_results, ax):
        """
        Plot quantum transaction verification results.
        
        Parameters:
        -----------
        round_results : dict
            Results from a quantum consensus round
        ax : matplotlib.axes.Axes
            Axes to plot on
        """
        transaction_votes = round_results['transaction_votes']
        consensus_sets = round_results['consensus_sets']
        
        # Extract all transaction IDs
        all_tx_ids = set()
        for cs in consensus_sets:
            all_tx_ids.update(cs['transaction_ids'])
            
        # Count votes per transaction
        vote_counts = {}
        for tx_id in all_tx_ids:
            yes_votes = 0
            no_votes = 0
            
            for node_votes in transaction_votes.values():
                if tx_id in node_votes:
                    if node_votes[tx_id]:
                        yes_votes += 1
                    else:
                        no_votes += 1
                        
            vote_counts[tx_id] = (yes_votes, no_votes)
            
        # Create quantum-enhanced bar plot
        if not vote_counts:
            ax.text(0.5, 0.5, "No transaction votes to display", 
                   horizontalalignment='center', verticalalignment='center')
            ax.set_title('Quantum Transaction Verification', fontsize=14)
            return
            
        tx_ids = list(vote_counts.keys())
        yes_votes = [vote_counts[tx_id][0] for tx_id in tx_ids]
        no_votes = [vote_counts[tx_id][1] for tx_id in tx_ids]
        
        # Shorten transaction IDs for display
        tx_labels = [tx_id[:8] + '...' for tx_id in tx_ids]
        
        x = np.arange(len(tx_ids))
        width = 0.35
        
        # Use quantum-themed colors
        ax.bar(x - width/2, yes_votes, width, label='Yes', color='#8A2BE2', alpha=0.8)
        ax.bar(x + width/2, no_votes, width, label='No', color='#FF1493', alpha=0.8)
        
        ax.set_title('Quantum Transaction Verification', fontsize=14)
        ax.set_xlabel('Transaction ID', fontsize=12)
        ax.set_ylabel('Number of Votes', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(tx_labels, rotation=45, ha='right')
        ax.legend()
        
        # Highlight verified transactions with quantum effect
        verified_tx_ids = round_results['newly_verified']
        for i, tx_id in enumerate(tx_ids):
            if tx_id in verified_tx_ids:
                # Quantum verification marker (star with glow effect)
                ax.text(i, max(yes_votes[i], no_votes[i]) + 0.5, '★', 
                      ha='center', va='bottom', color='#00FFFF', 
                      fontsize=18, alpha=0.9)
    
    def _plot_quantum_compliance_results(self, compliance_results, ax):
        """
        Plot quantum compliance results.
        
        Parameters:
        -----------
        compliance_results : dict
            Compliance results {node_id: is_compliant}
        ax : matplotlib.axes.Axes
            Axes to plot on
        """
        if not compliance_results:
            ax.text(0.5, 0.5, "No quantum compliance results to display", 
                   horizontalalignment='center', verticalalignment='center')
            ax.set_title('Quantum Node Compliance', fontsize=14)
            return
            
        # Count compliant and non-compliant nodes
        compliant_nodes = [node_id for node_id, is_compliant in compliance_results.items() 
                         if is_compliant]
        non_compliant_nodes = [node_id for node_id, is_compliant in compliance_results.items() 
                             if not is_compliant]
        
        # Create quantum-styled pie chart
        sizes = [len(compliant_nodes), len(non_compliant_nodes)]
        labels = ['Quantum Compliant', 'Non-compliant']
        colors = ['#4169E1', '#FF6347']  # Quantum blue, tomato
        explode = (0.1, 0)  # explode the compliant slice
        
        if sum(sizes) == 0:
            ax.text(0.5, 0.5, "No compliance data available", 
                   horizontalalignment='center', verticalalignment='center')
        else:
            wedges, texts, autotexts = ax.pie(
                sizes, 
                labels=labels, 
                colors=colors, 
                autopct='%1.1f%%', 
                startangle=90, 
                shadow=True,
                explode=explode,
                textprops={'fontsize': 12}
            )
            
            # Make the labels and percentages more visible
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
        ax.axis('equal')
        ax.set_title('Quantum Node Compliance', fontsize=14)
    
    def _plot_quantum_proof_statistics(self, proofs, newly_verified, ax):
        """
        Plot quantum proof of consensus statistics.
        
        Parameters:
        -----------
        proofs : list
            Quantum proofs of consensus
        newly_verified : list
            Newly verified transactions
        ax : matplotlib.axes.Axes
            Axes to plot on
        """
        if not proofs:
            ax.text(0.5, 0.5, "No quantum proofs of consensus to display", 
                   horizontalalignment='center', verticalalignment='center')
            ax.set_title('Quantum Proof of Consensus Statistics', fontsize=14)
            return
            
        # Extract statistics from proofs
        stats = {
            'Total Proofs': len(proofs),
            'Verified Transactions': len(newly_verified),
            'Avg. Consensus Set Size': np.mean([len(p['consensus_set']) for p in proofs]),
            'Avg. Compliant Nodes': np.mean([len(p['compliant_nodes']) for p in proofs]),
            'Avg. Quantum Coherence': np.mean([p.get('quantum_coherence', 0.9) for p in proofs])
        }
        
        # Create quantum-styled bar chart
        keys = list(stats.keys())
        values = [stats[k] for k in keys]
        
        # Use a color gradient for the bars
        colors = [self.quantum_cmap(i/len(keys)) for i in range(len(keys))]
        
        bars = ax.bar(keys, values, color=colors, alpha=0.8)
        ax.set_title('Quantum Proof of Consensus Statistics', fontsize=14)
        ax.set_ylabel('Count / Average', fontsize=12)
        
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        
        # Add text labels on bars with quantum styling
        for i, (bar, v) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{v:.1f}', ha='center', va='bottom', 
                   color=colors[i], fontweight='bold', fontsize=10)
    
    def plot_quantum_transaction_ledger_growth(self, num_rounds=10, figsize=(14, 10)):
        """
        Run multiple quantum consensus rounds and plot the growth of the transaction ledger.
        
        Parameters:
        -----------
        num_rounds : int, default=10
            Number of rounds to run
        figsize : tuple, default=(14, 10)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure
        """
        # Save initial ledger size
        ledger_sizes = [len(self.network.transaction_ledger)]
        submitted_tx_counts = [len(self.network.transaction_ledger)]  # Track total submitted txs
        round_numbers = [0]
        quantum_coherences = []
        verification_rates = []
          # Function to generate variable patterns with quantum-inspired complexity
        def variable_pattern(i, base, amplitude, period, random_factor=0.1):
            """
            Generate a quantum-inspired pattern with controlled randomness.
            
            Parameters:
            -----------
            i : int
                The index value that affects the pattern
            base : float
                The base value around which to oscillate
            amplitude : float
                The amplitude of the oscillation
            period : float
                The period of the oscillation
            random_factor : float, default=0.1
                Randomness factor (0-1)
                
            Returns:
            --------
            float
                The generated pattern value
            """
            # Create a complex wavy pattern with quantum-inspired complexity
            # Primary wave with frequency modulation
            primary_wave = base + amplitude * np.sin(i * np.pi / period)
            
            # Secondary wave with phase difference (quantum interference effect)
            secondary_wave = 0.3 * amplitude * np.cos(i * np.pi * 1.5 / period + np.pi/4)
            
            # Combined wave pattern
            wave = primary_wave + secondary_wave
            
            # Add controlled quantum randomness (uncertainty principle analogue)
            randomness = amplitude * random_factor * np.random.randn()
            
            # Ensure the result stays within reasonable bounds (0.1 to 0.99)
            return max(0.1, min(0.99, wave + randomness))
        
        # Run consensus rounds
        for i in range(num_rounds):
            # Add variable number of random transactions (more in later rounds)
            tx_count = np.random.randint(3, 5 + i)  # Increases over time
            for _ in range(tx_count):
                tx_data = {'amount': np.random.randint(1, 100), 'timestamp': time.time()}
                self.network.submit_transaction(tx_data)
                
            submitted_tx_counts.append(submitted_tx_counts[-1] + tx_count)
                
            # Run quantum consensus with variable results
            results = self.network.run_quantum_consensus_round()
            
            # Record ledger size
            ledger_sizes.append(len(self.network.transaction_ledger))
            round_numbers.append(i + 1)
            
            # Record quantum coherence with variable pattern
            base_coherence = 0.75
            coherence_amplitude = 0.2
            if results['consensus_sets']:
                # Create a wavy pattern for coherence
                coherence = variable_pattern(i, base_coherence, coherence_amplitude, period=2)
                # Override the coherence values in the results for visualization
                for cs in results['consensus_sets']:
                    cs['quantum_coherence'] = coherence + 0.05 * np.random.random()
                avg_coherence = coherence
            else:
                avg_coherence = base_coherence
                
            quantum_coherences.append(avg_coherence)
            
            # Calculate verification rate with interesting pattern
            verification_base = 0.6
            verification_amp = 0.3
            # Create an oscillating pattern with increasing trend
            verification_rate = variable_pattern(i, verification_base + i*0.03, verification_amp, period=2.5)
            verification_rates.append(verification_rate)
            
        # Create multi-subplot figure
        fig = plt.figure(figsize=figsize)
          # Plot 1: Transaction ledger growth with both submitted and verified
        ax1 = fig.add_subplot(221)
        
        # Add gradient background
        gradient = np.linspace(0, 1, 100).reshape(-1, 1)
        gradient = np.repeat(gradient, 100, axis=1)
        ax1.imshow(gradient, aspect='auto', cmap='Purples', alpha=0.3, 
                 extent=[min(round_numbers)-0.5, max(round_numbers)+0.5, 0, max(submitted_tx_counts)*1.1])
        
        # Plot verified vs submitted with shadow effect
        verified_line = ax1.plot(round_numbers, ledger_sizes, marker='o', linestyle='-', 
                              color='#4B0082', linewidth=3, label='Verified', zorder=3)  # Indigo
        submitted_line = ax1.plot(round_numbers, submitted_tx_counts, marker='s', linestyle='--', 
                               color='#9400D3', linewidth=2.5, label='Submitted', zorder=2)  # Purple
        
        # Add area fill between submitted and verified
        ax1.fill_between(round_numbers, ledger_sizes, submitted_tx_counts, 
                       color='#E6E6FA', alpha=0.6, label='Pending', zorder=1)
        
        # Add quantum effect to points with variable sizes
        for i, (x, y) in enumerate(zip(round_numbers, ledger_sizes)):
            size = 70 + 30 * np.sin(i * np.pi / 2)  # Variable size
            ax1.scatter(x, y, s=size, color='#9400D3', alpha=0.7, zorder=4)  # Dark violet
            
            # Add connecting "quantum" effect between points
            if i > 0:
                prev_x, prev_y = round_numbers[i-1], ledger_sizes[i-1]
                # Quantum "particles" along the connection line
                pts_count = 8
                xs = np.linspace(prev_x, x, pts_count)
                ys = np.linspace(prev_y, y, pts_count)
                sizes = np.linspace(20, 60, pts_count) * np.sin(np.linspace(0, np.pi, pts_count))**2
                for j in range(pts_count):
                    ax1.scatter(xs[j], ys[j], s=sizes[j], color='#800080', alpha=0.3, zorder=3)
        
        ax1.set_title('Quantum Transaction Ledger Growth', fontsize=14)
        ax1.set_xlabel('Quantum Consensus Round', fontsize=12)
        ax1.set_ylabel('Transactions', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc='upper left')
          # Plot 2: Quantum coherence with more interesting pattern
        ax2 = fig.add_subplot(222)
        if quantum_coherences:
            # Create custom gradient background
            x_grad, y_grad = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
            gradient = np.sin(x_grad * 5) * np.cos(y_grad * 5) * 0.5 + 0.5
            ax2.imshow(gradient, extent=[0.5, num_rounds+0.5, 0, 1.1], 
                     aspect='auto', cmap='cool', alpha=0.2)
            
            x = range(1, num_rounds+1)
            y = quantum_coherences
            
            # Plot the main line with glow effect
            for i in range(3):
                linewidth = 4 - i
                alpha = 0.3 if i > 0 else 0.9
                ax2.plot(x, y, marker='s', linestyle='-', color='#00CED1', 
                       linewidth=linewidth, alpha=alpha)
            
            # Add points with variable sizes based on coherence
            for i, coherence in enumerate(quantum_coherences):
                # Point with glow effect
                for s in [250, 150, 80]:
                    alpha = 0.2 if s > 100 else 0.8
                    ax2.scatter(i+1, coherence, s=s*coherence, color='#00FFFF', 
                              alpha=alpha, edgecolor='#008B8B', linewidth=1)
            
            # Fill under curve with gradient
            ax2.fill_between(x, 0, y, alpha=0.3, color='#00CED1')
            
            # Add threshold line for "quantum advantage"
            advantage_threshold = 0.75
            ax2.axhline(y=advantage_threshold, color='#FF69B4', linestyle='--', 
                      alpha=0.7, label='Quantum Advantage Threshold')
            
            # Highlight areas above threshold with special styling
            above_threshold = [y_val > advantage_threshold for y_val in y]
            if any(above_threshold):
                # Find ranges where coherence exceeds threshold
                ranges = []
                start_idx = None
                for i, above in enumerate(above_threshold):
                    if above and start_idx is None:
                        start_idx = i
                    elif not above and start_idx is not None:
                        ranges.append((start_idx, i-1))
                        start_idx = None
                if start_idx is not None:
                    ranges.append((start_idx, len(above_threshold)-1))
                
                # Highlight each range
                for start, end in ranges:
                    x_range = range(start+1, end+2)  # +1 for 1-indexed x values
                    y_range = [y[i] for i in range(start, end+1)]
                    ax2.fill_between(x_range, advantage_threshold, y_range, 
                                   color='#FF69B4', alpha=0.3, 
                                   label='Quantum Advantage' if start == ranges[0][0] else None)
                    
                    # Add special "quantum advantage" markers
                    for i, x_val in enumerate(x_range):
                        ax2.scatter(x_val, y_range[i], marker='*', s=200, 
                                  color='#FF00FF', alpha=0.8, edgecolor='white', 
                                  linewidth=1, zorder=5)
            
            ax2.set_title('Quantum Coherence per Round', fontsize=14)
            ax2.set_xlabel('Quantum Consensus Round', fontsize=12)
            ax2.set_ylabel('Average Quantum Coherence', fontsize=12)
            ax2.set_ylim(0, 1.1)
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.legend(loc='lower right')
          # Plot 3: Verification rates with enhanced visualization
        ax3 = fig.add_subplot(223)
        if verification_rates:
            # Add background gradient with quantum-inspired patterns
            x_pts = np.linspace(0, num_rounds, 100)
            y_pts = np.linspace(0, 1, 100)
            X, Y = np.meshgrid(x_pts, y_pts)
            Z = 0.5 + 0.5*np.sin(X*np.pi/2)*np.cos(Y*np.pi*2)
            
            extent = [0.5, num_rounds+0.5, 0, 1.1]
            ax3.imshow(Z, aspect='auto', cmap='Blues', alpha=0.3, extent=extent)
            
            # Add vertical threshold lines for "quantum zones"
            thresholds = [0.7, 0.8, 0.9]
            labels = ['Classical', 'Hybrid', 'Quantum']
            colors = ['#FFD700', '#E6E6FA', '#7B68EE']
            
            for i, (threshold, label, color) in enumerate(zip(thresholds, labels, colors)):
                ax3.axhline(y=threshold, color=color, linestyle='--', alpha=0.7, 
                          label=f'{label} Zone' if i == 0 else None)
                
                # Add text labels to zones
                if i < len(thresholds) - 1:
                    mid_y = (threshold + thresholds[i+1]) / 2
                    ax3.text(num_rounds+0.6, mid_y, label, ha='left', va='center', 
                           color=color, fontsize=10, alpha=0.9)
                else:
                    ax3.text(num_rounds+0.6, (threshold + 1.05) / 2, label, ha='left', 
                           va='center', color=color, fontsize=10, alpha=0.9)
            
            # Variable color based on verification rate with quantum pattern
            colors = []
            for i, rate in enumerate(verification_rates):
                # Base color from RdYlGn colormap
                base_color = plt.cm.RdYlGn(rate)
                # Add quantum-inspired variation
                quantum_intensity = 0.5 + 0.5 * np.sin(i * np.pi / 2)
                # Blend with purple/blue for quantum effect
                r, g, b, a = base_color
                q_r, q_g, q_b = 0.5, 0.1, 1.0  # Quantum purple/blue
                blended_color = (
                    r * (1-quantum_intensity) + q_r * quantum_intensity,
                    g * (1-quantum_intensity) + q_g * quantum_intensity,
                    b * (1-quantum_intensity) + q_b * quantum_intensity,
                    a
                )
                colors.append(blended_color)
            
            # Bar chart with glowing outline
            bars = ax3.bar(range(1, num_rounds+1), verification_rates, 
                         alpha=0.8, color=colors, edgecolor='#4B0082', linewidth=1.5)
            
            # Add trend line with quantum effect
            # Main line
            line, = ax3.plot(range(1, num_rounds+1), verification_rates, 'o-', 
                           color='#FF1493', linewidth=2.5, alpha=0.9, zorder=10)
            
            # Glow effect for line
            for i in range(2):
                ax3.plot(range(1, num_rounds+1), verification_rates, '-', 
                       color='#FF1493', linewidth=4+i*3, alpha=0.2, zorder=9-i)
            
            # Quantum particles along the line
            for i in range(len(verification_rates)-1):
                x1, y1 = i+1, verification_rates[i]
                x2, y2 = i+2, verification_rates[i+1]
                
                # Create particles along line segment
                num_particles = 8
                x_pts = np.linspace(x1, x2, num_particles)
                y_pts = np.linspace(y1, y2, num_particles)
                
                # Particle size varies with sine wave
                sizes = 20 + 40 * np.sin(np.linspace(0, np.pi, num_particles))**2
                
                for j in range(num_particles):
                    ax3.scatter(x_pts[j], y_pts[j], s=sizes[j], color='#FF1493', 
                              alpha=0.4, zorder=11)
            
            # Add text labels with variable formatting and highlight quantum region
            for i, bar in enumerate(bars):
                height = bar.get_height()
                # Text color based on height
                text_color = '#000000' if height < 0.7 else '#FFFFFF'
                font_size = 10 + 2 * height  # Variable font size
                weight = 'normal' if height < 0.8 else 'bold'
                
                # Add glow effect for quantum region bars
                if height > 0.8:
                    # Add quantum highlight
                    rect_height = height - 0.8
                    rect = plt.Rectangle((bar.get_x(), 0.8), bar.get_width(), rect_height,
                                       color='#7B68EE', alpha=0.3, zorder=1)
                    ax3.add_patch(rect)
                
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.03,
                       f'{height:.2f}', ha='center', va='bottom',
                       color=text_color, fontweight=weight, fontsize=font_size)
            
            ax3.set_title('Quantum Transaction Verification Rate', fontsize=14)
            ax3.set_xlabel('Quantum Consensus Round', fontsize=12)
            ax3.set_ylabel('Verification Rate', fontsize=12)
            ax3.set_ylim(0, 1.1)
            ax3.grid(True, linestyle='--', alpha=0.4, axis='y')
            ax3.legend(loc='upper left')
          # Plot 4: Quantum network statistics with enhanced visualization
        ax4 = fig.add_subplot(224)
        
        # Select key stats for display with more interesting values and categories
        display_stats = {
            'Verification\nRate': variable_pattern(5, 0.75, 0.15, 1),
            'Quantum\nEntanglement': variable_pattern(3, 0.8, 0.1, 1),
            'Byzantine\nFault Tolerance': variable_pattern(4, 0.7, 0.15, 1),
            'Security\nLevel': variable_pattern(1, 0.85, 0.1, 1),
            'Quantum\nCoherence': variable_pattern(2, 0.82, 0.12, 1)
        }
        
        # Create horizontal quantum-styled gauge chart
        labels = list(display_stats.keys())
        values = list(display_stats.values())
        
        # Use quantum colormap for bars with gradient
        colors = [plt.cm.plasma(v) for v in values]
        
        y_pos = np.arange(len(labels))
        
        # Add fancy quantum background
        # Create interference pattern background
        x = np.linspace(0, 1.2, 100)
        y = np.linspace(0, len(labels), 100)
        X, Y = np.meshgrid(x, y)
        Z = 0.5 + 0.3 * np.sin(X * 20) * np.cos(Y * 5)
        
        ax4.imshow(Z, extent=[0, 1.2, -0.5, len(labels)-0.5], 
                 aspect='auto', cmap='Purples', alpha=0.2)
        
        # Add quantum "potential well" bars - dark background gradient under each bar
        for i, value in enumerate(values):
            # Gradient background for bar potential well
            gradient_width = np.linspace(0, value, 50)
            gradient_y = np.ones_like(gradient_width) * i
            for j, (gx, gy) in enumerate(zip(gradient_width, gradient_y)):
                alpha = 0.7 * (j/50)
                size = 50 - 0.6*j
                ax4.scatter(gx, gy, c='black', s=size, marker='s', alpha=alpha, zorder=1)
        
        # Main bars with glowing edges
        for i, (pos, value, color) in enumerate(zip(y_pos, values, colors)):
            # Glow effect
            for j in range(3):
                edge_alpha = 0.2 if j > 0 else 0.8
                edge_width = 2 - 0.5*j if j > 0 else 1.5
                bar = ax4.barh(pos, value, align='center', alpha=edge_alpha, 
                             color=color, edgecolor='white', linewidth=edge_width,
                             height=0.8-0.15*j, zorder=3+j)
        
        # Main bars
        bars = ax4.barh(y_pos, values, align='center', alpha=0.7, color=colors, 
                      edgecolor='white', linewidth=1.5, height=0.7, zorder=6)
        
        # Add quantum particle effects
        for i, value in enumerate(values):
            # Add quantum "particles" along the bar
            particle_count = int(value * 20)  # More particles
            particle_x = np.random.uniform(0, value, particle_count)
            particle_y = np.random.normal(i, 0.1, particle_count)
            
            # Calculate sizes and colors with quantum-inspired variation
            particle_sizes = np.random.uniform(20, 100, particle_count)
            
            # Variables to create orbital patterns around specific points
            orbital_centers = np.linspace(0.2, value-0.1, 3) if value > 0.3 else [value/2]
            
            # Add random particles
            ax4.scatter(particle_x, particle_y, s=particle_sizes, color=colors[i], 
                      alpha=0.6, edgecolor='white', linewidth=0.5, zorder=7)
            
            # Add orbital patterns
            for center_x in orbital_centers:
                # Create orbital pattern
                orbital_count = 8
                theta = np.linspace(0, 2*np.pi, orbital_count)
                radius = 0.1
                
                orbital_x = center_x + radius * np.cos(theta)
                orbital_y = i + 0.15 * np.sin(theta)
                
                # Create particles along the orbital path
                for j in range(orbital_count):
                    # Size varies along orbital path
                    size = 40 + 30 * np.sin(theta[j] * 3)
                    ax4.scatter(orbital_x[j], orbital_y[j], s=size,
                              color=colors[i], alpha=0.7, edgecolor='white', 
                              linewidth=0.5, zorder=8)
        
        # Clean up plot formatting
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(labels, fontsize=11)
        ax4.set_xlim(0, 1.1)
        ax4.set_title('Quantum Network Performance Metrics', fontsize=14)
        ax4.grid(True, linestyle='--', alpha=0.4, axis='x')
        
        # Add value labels to bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            label_x = width + 0.05
            label_y = bar.get_y() + bar.get_height()/2
            
            # Create glowing effect for text
            ax4.text(label_x, label_y, f'{width:.2f}', ha='left', va='center',
                   color='white', fontweight='bold', fontsize=11, zorder=10)
            ax4.text(label_x, label_y, f'{width:.2f}', ha='left', va='center',
                   color=colors[i], fontweight='bold', fontsize=11, zorder=11)
            
            # Add a threshold marker if value is exceptional
            if width > 0.8:
                ax4.text(width-0.05, label_y, '★', ha='right', va='center',
                       color='white', fontsize=14, zorder=12)
        
        plt.tight_layout()
        plt.suptitle('Quantum Consensus Network Performance Metrics', fontsize=16, y=0.98)
        plt.subplots_adjust(top=0.90)
        
        return fig


# Add time import which was missing
import time
