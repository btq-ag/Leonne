#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
consensus_visualizer.py

Visualization utilities for Distributed Consensus Networks (DCN).
Provides visual representations of network topology, consensus sets,
and protocol dynamics.

Author: Jeffrey Morais, BTQ
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from matplotlib.animation import FuncAnimation
from itertools import cycle


class ConsensusVisualizer:
    """
    Visualization tools for Distributed Consensus Networks.
    """
    
    def __init__(self, network):
        """
        Initialize visualizer with a network.
        
        Parameters:
        -----------
        network : ConsensusNetwork
            The consensus network to visualize
        """
        self.network = network
        self.node_positions = None
        self.color_palette = sns.color_palette("husl", 10)
        self.consensus_set_colors = cycle(self.color_palette)
        
    def plot_network_topology(self, figsize=(10, 8), show_node_ids=True):
        """
        Plot the network topology.
        
        Parameters:
        -----------
        figsize : tuple, default=(10, 8)
            Figure size
        show_node_ids : bool, default=True
            Whether to show node IDs
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create a color map based on node honesty
        honesty_cmap = plt.cm.RdYlGn  # Red to Yellow to Green colormap
        honesty_values = [self.network.honesty_distribution[int(node_id.split('_')[1])] 
                         for node_id in self.network.nodes.keys()]
        
        # Calculate node positions (store for future use)
        if self.node_positions is None:
            self.node_positions = nx.spring_layout(self.network.network_graph, seed=42)
            
        # Draw the network
        nx.draw_networkx_edges(self.network.network_graph, self.node_positions, 
                              alpha=0.3, ax=ax)
        
        node_collection = nx.draw_networkx_nodes(
            self.network.network_graph, 
            self.node_positions,
            node_color=honesty_values,
            cmap=honesty_cmap,
            vmin=0.0,
            vmax=1.0,
            node_size=500,
            ax=ax
        )
        
        # Add colorbar for honesty
        sm = plt.cm.ScalarMappable(cmap=honesty_cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Honesty Probability')
        
        # Show node IDs if requested
        if show_node_ids:
            nx.draw_networkx_labels(self.network.network_graph, self.node_positions, ax=ax)
            
        # Set title and remove axis
        ax.set_title('Network Topology')
        ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_consensus_sets(self, consensus_sets, figsize=(12, 10)):
        """
        Plot the consensus sets within the network.
        
        Parameters:
        -----------
        consensus_sets : list
            List of consensus sets to visualize
        figsize : tuple, default=(12, 10)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure
        """
        if not consensus_sets:
            print("No consensus sets to visualize.")
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
        
        # Draw each consensus set with a different color
        legend_elements = []
        for i, cs in enumerate(consensus_sets):
            # Get nodes in this consensus set
            cs_nodes = cs['nodes']
            cs_subgraph = self.network.network_graph.subgraph(cs_nodes)
            
            # Choose a color for this consensus set
            color = next(self.consensus_set_colors)
            
            # Draw nodes in this consensus set
            nx.draw_networkx_nodes(cs_subgraph, self.node_positions,
                                 node_color=[color]*len(cs_nodes),
                                 node_size=600, ax=ax)
            
            # Draw edges between nodes in this consensus set
            nx.draw_networkx_edges(cs_subgraph, self.node_positions,
                                 width=2, alpha=0.7, ax=ax,
                                 edge_color=color)
            
            # Add to legend
            from matplotlib.lines import Line2D
            legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                        markerfacecolor=color, markersize=10,
                                        label=f'Consensus Set {i+1}'))
            
        # Show node IDs
        nx.draw_networkx_labels(self.network.network_graph, self.node_positions, ax=ax)
        
        # Add legend
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Set title and remove axis
        ax.set_title('Consensus Sets')
        ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def visualize_consensus_round(self, round_results, figsize=(15, 12)):
        """
        Visualize the results of a consensus round.
        
        Parameters:
        -----------
        round_results : dict
            Results from a consensus round
        figsize : tuple, default=(15, 12)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure
        """
        fig = plt.figure(figsize=figsize)
        
        # 1. Plot network with consensus sets
        ax1 = fig.add_subplot(221)
        self._plot_network_with_consensus_sets(round_results['consensus_sets'], ax1)
        
        # 2. Plot transaction verification results
        ax2 = fig.add_subplot(222)
        self._plot_transaction_verification(round_results, ax2)
        
        # 3. Plot compliance results
        ax3 = fig.add_subplot(223)
        self._plot_compliance_results(round_results['compliance_results'], ax3)
        
        # 4. Plot proof of consensus statistics
        ax4 = fig.add_subplot(224)
        self._plot_proof_statistics(round_results['proofs_of_consensus'], 
                                  round_results['newly_verified'], ax4)
        
        plt.tight_layout()
        return fig
    
    def _plot_network_with_consensus_sets(self, consensus_sets, ax):
        """
        Plot the network with consensus sets highlighted.
        
        Parameters:
        -----------
        consensus_sets : list
            List of consensus sets
        ax : matplotlib.axes.Axes
            Axes to plot on
        """
        # Calculate node positions if not already done
        if self.node_positions is None:
            self.node_positions = nx.spring_layout(self.network.network_graph, seed=42)
            
        # Draw the network (grey background)
        nx.draw_networkx_edges(self.network.network_graph, self.node_positions, 
                              alpha=0.1, ax=ax, edge_color='grey')
        
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
            
            # Draw nodes in this consensus set
            nx.draw_networkx_nodes(cs_subgraph, self.node_positions,
                                 node_color=[color]*len(cs_nodes),
                                 node_size=600, ax=ax)
            
            # Draw edges between nodes in this consensus set
            nx.draw_networkx_edges(cs_subgraph, self.node_positions,
                                 width=2, alpha=0.7, ax=ax,
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
        ax.set_title('Network with Consensus Sets')
        ax.axis('off')
    
    def _plot_transaction_verification(self, round_results, ax):
        """
        Plot transaction verification results.
        
        Parameters:
        -----------
        round_results : dict
            Results from a consensus round
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
            
        # Create bar plot
        if not vote_counts:
            ax.text(0.5, 0.5, "No transaction votes to display", 
                   horizontalalignment='center', verticalalignment='center')
            ax.set_title('Transaction Verification')
            return
            
        tx_ids = list(vote_counts.keys())
        yes_votes = [vote_counts[tx_id][0] for tx_id in tx_ids]
        no_votes = [vote_counts[tx_id][1] for tx_id in tx_ids]
        
        # Shorten transaction IDs for display
        tx_labels = [tx_id[:8] + '...' for tx_id in tx_ids]
        
        x = np.arange(len(tx_ids))
        width = 0.35
        
        ax.bar(x - width/2, yes_votes, width, label='Yes', color='green', alpha=0.7)
        ax.bar(x + width/2, no_votes, width, label='No', color='red', alpha=0.7)
        
        ax.set_title('Transaction Verification')
        ax.set_xlabel('Transaction ID')
        ax.set_ylabel('Number of Votes')
        ax.set_xticks(x)
        ax.set_xticklabels(tx_labels, rotation=45, ha='right')
        ax.legend()
        
        # Highlight verified transactions
        verified_tx_ids = round_results['newly_verified']
        for i, tx_id in enumerate(tx_ids):
            if tx_id in verified_tx_ids:
                ax.text(i, max(yes_votes[i], no_votes[i]) + 0.5, '✓', 
                      ha='center', va='bottom', color='green', fontsize=12)
    
    def _plot_compliance_results(self, compliance_results, ax):
        """
        Plot compliance results.
        
        Parameters:
        -----------
        compliance_results : dict
            Compliance results {node_id: is_compliant}
        ax : matplotlib.axes.Axes
            Axes to plot on
        """
        if not compliance_results:
            ax.text(0.5, 0.5, "No compliance results to display", 
                   horizontalalignment='center', verticalalignment='center')
            ax.set_title('Node Compliance')
            return
            
        # Count compliant and non-compliant nodes
        compliant_nodes = [node_id for node_id, is_compliant in compliance_results.items() 
                         if is_compliant]
        non_compliant_nodes = [node_id for node_id, is_compliant in compliance_results.items() 
                             if not is_compliant]
        
        # Create pie chart
        sizes = [len(compliant_nodes), len(non_compliant_nodes)]
        labels = ['Compliant', 'Non-compliant']
        colors = ['lightgreen', 'lightcoral']
        
        if sum(sizes) == 0:
            ax.text(0.5, 0.5, "No compliance data available", 
                   horizontalalignment='center', verticalalignment='center')
        else:
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                 startangle=90, shadow=True)
            
        ax.axis('equal')
        ax.set_title('Node Compliance')
    
    def _plot_proof_statistics(self, proofs, newly_verified, ax):
        """
        Plot proof of consensus statistics.
        
        Parameters:
        -----------
        proofs : list
            Proofs of consensus
        newly_verified : list
            Newly verified transactions
        ax : matplotlib.axes.Axes
            Axes to plot on
        """
        if not proofs:
            ax.text(0.5, 0.5, "No proofs of consensus to display", 
                   horizontalalignment='center', verticalalignment='center')
            ax.set_title('Proof of Consensus Statistics')
            return
            
        # Extract statistics from proofs
        stats = {
            'Total Proofs': len(proofs),
            'Verified Transactions': len(newly_verified),
            'Avg. Consensus Set Size': np.mean([len(p['consensus_set']) for p in proofs]),
            'Avg. Compliant Nodes': np.mean([len(p['compliant_nodes']) for p in proofs])
        }
        
        # Create bar chart
        keys = list(stats.keys())
        values = [stats[k] for k in keys]
        
        ax.bar(keys, values, color='lightblue')
        ax.set_title('Proof of Consensus Statistics')
        ax.set_ylabel('Count / Average')
        
        # Rotate x-axis labels if needed
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        
        # Add text labels on bars
        for i, v in enumerate(values):
            ax.text(i, v + 0.1, f'{v:.1f}', ha='center')
    
    def animate_consensus_rounds(self, num_rounds=5, interval=2000):
        """
        Create an animation of multiple consensus rounds.
        
        Parameters:
        -----------
        num_rounds : int, default=5
            Number of rounds to animate
        interval : int, default=2000
            Interval between frames in milliseconds
            
        Returns:
        --------
        matplotlib.animation.FuncAnimation
            Animation
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Calculate node positions if not already done
        if self.node_positions is None:
            self.node_positions = nx.spring_layout(self.network.network_graph, seed=42)
        
        # Initialize with empty plot
        nx.draw_networkx_edges(self.network.network_graph, self.node_positions, 
                              alpha=0.1, ax=ax, edge_color='grey')
        nx.draw_networkx_nodes(self.network.network_graph, self.node_positions,
                             node_color='lightgrey', alpha=0.3, node_size=500, ax=ax)
        nx.draw_networkx_labels(self.network.network_graph, self.node_positions, ax=ax)
        
        round_results = []
        
        def init():
            ax.clear()
            ax.set_title('Network with Consensus Sets - Round 0')
            ax.axis('off')
            nx.draw_networkx_edges(self.network.network_graph, self.node_positions, 
                                  alpha=0.1, ax=ax, edge_color='grey')
            nx.draw_networkx_nodes(self.network.network_graph, self.node_positions,
                                 node_color='lightgrey', alpha=0.3, node_size=500, ax=ax)
            nx.draw_networkx_labels(self.network.network_graph, self.node_positions, ax=ax)
            return ax,
        
        def update(frame):
            ax.clear()
            
            # Run a new consensus round if needed
            if frame >= len(round_results):
                result = self.network.run_consensus_round()
                round_results.append(result)
            
            # Plot the consensus sets for this round
            self._plot_network_with_consensus_sets(round_results[frame]['consensus_sets'], ax)
            ax.set_title(f'Network with Consensus Sets - Round {frame+1}')
            return ax,
        
        # Create animation
        anim = FuncAnimation(fig, update, frames=num_rounds, init_func=init,
                           interval=interval, blit=True)
        
        plt.tight_layout()
        return anim
    
    def plot_transaction_ledger_growth(self, num_rounds=10, figsize=(10, 6)):
        """
        Run multiple consensus rounds and plot the growth of the transaction ledger.
        
        Parameters:
        -----------
        num_rounds : int, default=10
            Number of rounds to run
        figsize : tuple, default=(10, 6)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure
        """
        # Save initial ledger size
        ledger_sizes = [len(self.network.transaction_ledger)]
        round_numbers = [0]
        
        # Run consensus rounds
        for i in range(num_rounds):
            # Add some random transactions
            for _ in range(np.random.randint(3, 8)):
                tx_data = {'amount': np.random.randint(1, 100), 'timestamp': time.time()}
                self.network.submit_transaction(tx_data)
                
            # Run consensus
            self.network.run_consensus_round()
            
            # Record ledger size
            ledger_sizes.append(len(self.network.transaction_ledger))
            round_numbers.append(i + 1)
            
        # Plot results
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(round_numbers, ledger_sizes, marker='o', linestyle='-', color='blue')
        
        ax.set_title('Transaction Ledger Growth')
        ax.set_xlabel('Consensus Round')
        ax.set_ylabel('Number of Verified Transactions')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        return fig
