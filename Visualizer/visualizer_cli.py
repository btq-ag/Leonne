#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualizer_cli.py

A command line interface for visualizing and running classical algorithms.
This script provides an interactive way to explore and run the various
implementations in the Classical Algorithms directory.

Author: Jeffrey Morais, BTQ
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import argparse
import importlib

# Add current directory to system path
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))
# Add parent directory to system path
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

# Import the ClassicalAlgorithmsHub
from classical_hub import ClassicalAlgorithmsHub

# Set up matplotlib for interactive mode
plt.style.use('dark_background')

class VisualizerCLI:
    """
    Command Line Interface for visualizing classical algorithms.
    Provides interactive functionality to explore and run algorithms.
    """
    
    def __init__(self):
        """Initialize the CLI with the Classical Algorithms Hub"""
        self.hub = ClassicalAlgorithmsHub()
        self.current_category = None
        self.current_module = None
        self.current_algorithm = None
        
    def display_welcome(self):
        """Display welcome message and basic instructions"""
        print("\n" + "="*80)
        print("  BTQ CLASSICAL ALGORITHMS VISUALIZER")
        print("="*80)
        print("\nThis tool allows you to explore and run various classical algorithms")
        print("from the Classical Algorithms modules. You can visualize network structures,")
        print("run simulations, and explore the different implementations.")
        print("\nType 'help' for a list of commands or 'exit' to quit.\n")
    
    def display_help(self):
        """Display help information with available commands"""
        print("\nAvailable Commands:")
        print("  categories             - List all algorithm categories")
        print("  modules [category]     - List modules in a category")
        print("  algos [category] [module] - List algorithms in a category/module")
        print("  select category module algorithm - Select an algorithm")
        print("  info [algorithm]       - Show information about an algorithm")
        print("  run [args...]          - Run the selected algorithm with arguments")
        print("  examples               - Show example commands for common tasks")
        print("  visualize [type]       - Run a predefined visualization")
        print("  clear                  - Clear the screen")
        print("  help                   - Show this help message")
        print("  exit                   - Exit the program")
    
    def display_examples(self):
        """Display example commands for common tasks"""
        print("\nExample Commands:")
        print("  1. Run a trust partitioner:")
        print("     > select Trust Partitioning trustPartitioner networkPartitioner")
        print("     > run [[0,1,2],[3,4],[5,6,7]] np.random.rand(8,8) {i:0.3+0.05*i for i in range(8)} {i:0.5 for i in range(3)}")
        print()
        print("  2. Create a blockchain visualization:")
        print("     > select Blockchain Simulation blockchainVisualizer create_blockchain_network_visualization")
        print("     > run 5 10 30")
        print()
        print("  3. Visualize a consensus network:")
        print("     > visualize consensus_network")
        print()
        print("  4. Visualize a topological partitioning:")
        print("     > visualize topological_partitioning")
    
    def list_categories(self):
        """List all available algorithm categories"""
        categories = self.hub.list_categories()
        print("\nAvailable Categories:")
        for category in categories:
            print(f"  - {category}")
    
    def list_modules(self, category=None):
        """List all available modules in a category"""
        if not category and not self.current_category:
            print("Please specify a category or select one first.")
            return
            
        category = category or self.current_category
        modules = self.hub.list_modules(category)
        
        if not modules:
            print(f"No modules found in category '{category}'.")
            return
            
        print(f"\nModules in '{category}':")
        for module in modules:
            print(f"  - {module}")
    
    def list_algorithms(self, category=None, module=None):
        """List all available algorithms in a category/module"""
        category = category or self.current_category
        module = module or self.current_module
        
        if not category:
            print("Please specify a category or select one first.")
            return
            
        algorithms = self.hub.list_algorithms(category, module)
        
        if not algorithms:
            msg = f"No algorithms found in category '{category}'"
            if module:
                msg += f", module '{module}'"
            print(msg + ".")
            return
            
        print(f"\nAlgorithms in '{category}'", end="")
        if module:
            print(f", module '{module}'", end="")
        print(":")
        
        for algo in algorithms:
            print(f"  - {algo}")
    
    def select_algorithm(self, category, module, algorithm_name):
        """Select an algorithm for further operations"""
        # Check if the category exists
        if category not in self.hub.list_categories():
            print(f"Category '{category}' not found.")
            return False
            
        # Check if the module exists in the category
        modules = self.hub.list_modules(category)
        if module not in modules:
            print(f"Module '{module}' not found in category '{category}'.")
            return False
            
        # Find the algorithm
        algo_key = f"{category}.{module}.{algorithm_name}"
        algorithm = self.hub.get_algorithm(algo_key)
        
        if not algorithm:
            print(f"Algorithm '{algorithm_name}' not found in {category}/{module}.")
            return False
            
        # Set the current selections
        self.current_category = category
        self.current_module = module
        self.current_algorithm = algo_key
        
        print(f"Selected algorithm: {algo_key}")
        return True
    
    def display_algorithm_info(self, algorithm_key=None):
        """Display information about an algorithm"""
        if not algorithm_key and not self.current_algorithm:
            print("Please specify an algorithm or select one first.")
            return
            
        algorithm_key = algorithm_key or self.current_algorithm
        info = self.hub.get_algorithm_info(algorithm_key)
        
        if not info:
            print(f"Algorithm '{algorithm_key}' not found.")
            return
            
        print(f"\nAlgorithm: {info['name']}")
        print(f"Full name: {info['full_name']}")
        print(f"Type: {info['type']}")
        print(f"Signature: {info['signature']}")
        print("\nDocumentation:")
        print(info['docstring'])
    
    def run_algorithm(self, *args):
        """Run the selected algorithm with provided arguments"""
        if not self.current_algorithm:
            print("No algorithm selected. Please select an algorithm first.")
            return
            
        # Get the algorithm
        algorithm = self.hub.get_algorithm(self.current_algorithm)
        
        try:
            # Convert string arguments to Python objects
            processed_args = []
            for arg in args:
                try:
                    # Try to evaluate the argument as a Python expression
                    processed_args.append(eval(arg))
                except:
                    # If that fails, treat it as a string
                    processed_args.append(arg)
            
            # Run the algorithm
            result = algorithm(*processed_args)
            print(f"\nAlgorithm executed successfully.")
            return result
            
        except Exception as e:
            print(f"Error executing algorithm: {e}")
            return None
    
    def run_visualization(self, visualization_type):
        """Run a predefined visualization"""
        if visualization_type == "consensus_network":
            self._visualize_consensus_network()
        elif visualization_type == "blockchain":
            self._visualize_blockchain()
        elif visualization_type == "topological_partitioning":
            self._visualize_topological_partitioning()
        elif visualization_type == "network_generator":
            self._visualize_network_generator()
        else:
            print(f"Visualization type '{visualization_type}' not recognized.")
            print("Available types: consensus_network, blockchain, topological_partitioning, network_generator")
    
    def _visualize_consensus_network(self):
        """Visualize a consensus network simulation"""
        # First, select the ConsensusNetwork class
        self.select_algorithm("Consensus Iterations", "consensus_network", "ConsensusNetwork")
        
        print("\nCreating a consensus network simulation...")
        
        # Create a consensus network with 15 nodes
        network = self.run_algorithm("15", "None", "0.3")
        
        # Run a few rounds of consensus
        for i in range(3):
            network.run_consensus_round()
            
        # Get node positions for visualization
        pos = nx.spring_layout(network.network_graph)
        
        # Create a figure
        plt.figure(figsize=(10, 8))
        
        # Draw the network
        nx.draw_networkx_edges(network.network_graph, pos, alpha=0.3)
        
        # Color nodes based on their state
        node_colors = []
        for node_id in network.network_graph.nodes():
            node = network.nodes[node_id]
            if node.state == "consensus":
                node_colors.append('green')
            elif node.state == "candidate":
                node_colors.append('blue')
            else:
                node_colors.append('red')
                
        nx.draw_networkx_nodes(network.network_graph, pos, node_color=node_colors)
        nx.draw_networkx_labels(network.network_graph, pos)
        
        plt.title("Consensus Network Simulation")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def _visualize_blockchain(self):
        """Visualize a blockchain network"""
        # Select the blockchain visualizer
        self.select_algorithm("Blockchain Simulation", "blockchainVisualizer", "create_blockchain_network_visualization")
        
        print("\nCreating a blockchain visualization...")
        print("This will open a separate window with an animated visualization.")
        
        # Run the visualization with default parameters
        self.run_algorithm("4", "8", "20")
    
    def _visualize_topological_partitioning(self):
        """Visualize topological partitioning"""
        # Select the topological partitioning function
        self.select_algorithm("Trust Partitioning", "topologicalPartitioner", "visualize_topological_partitioning")
        
        print("\nCreating a topological partitioning visualization...")
        print("This will open a separate window with an animated visualization.")
        
        # Run the visualization
        self.run_algorithm("'small_world'", "30", "20")
    
    def _visualize_network_generator(self):
        """Visualize network generation and shuffling"""
        # Select the consensus shuffle function
        self.select_algorithm("Network Graph Generator", "graphGenerator", "consensusShuffle")
        
        print("\nDemonstrating a consensus shuffle on a network edge set...")
        
        # Create a sample edge set
        edge_set = [[1,2], [2,3], [3,4], [4,5], [5,1], [2,5], [3,1]]
        
        # Run the shuffle
        result = self.run_algorithm(str(edge_set), "False")
        
        # Create a directed graph from the original edge set
        G_original = nx.DiGraph()
        for edge in edge_set:
            G_original.add_edge(edge[0], edge[1])
            
        # Create a directed graph from the shuffled edge set
        G_shuffled = nx.DiGraph()
        for edge in result:
            G_shuffled.add_edge(edge[0], edge[1])
        
        # Plot both graphs
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot original graph
        pos_original = nx.spring_layout(G_original, seed=42)
        nx.draw(G_original, pos_original, ax=ax1, with_labels=True, 
                node_color='skyblue', node_size=500, font_size=10, font_weight='bold',
                edge_color='gray', width=2, arrowsize=15)
        ax1.set_title("Original Graph")
        
        # Plot shuffled graph
        pos_shuffled = nx.spring_layout(G_shuffled, seed=42)
        nx.draw(G_shuffled, pos_shuffled, ax=ax2, with_labels=True, 
                node_color='lightgreen', node_size=500, font_size=10, font_weight='bold',
                edge_color='gray', width=2, arrowsize=15)
        ax2.set_title("Shuffled Graph")
        
        plt.tight_layout()
        plt.show()
    
    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def run_cli(self):
        """Run the command line interface"""
        self.display_welcome()
        
        while True:
            # Display the current selection in the prompt
            prompt = "> "
            if self.current_category:
                prompt = f"[{self.current_category}"
                if self.current_module:
                    prompt += f"/{self.current_module}"
                prompt += "] > "
            
            # Get user input
            try:
                user_input = input(prompt).strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting...")
                break
                
            # Handle empty input
            if not user_input:
                continue
                
            # Parse the command
            parts = user_input.split()
            command = parts[0].lower()
            args = parts[1:]
            
            # Execute the command
            if command == 'exit' or command == 'quit':
                print("Exiting...")
                break
                
            elif command == 'help':
                self.display_help()
                
            elif command == 'examples':
                self.display_examples()
                
            elif command == 'categories':
                self.list_categories()
                
            elif command == 'modules':
                category = args[0] if args else None
                self.list_modules(category)
                
            elif command == 'algos':
                category = args[0] if len(args) > 0 else None
                module = args[1] if len(args) > 1 else None
                self.list_algorithms(category, module)
                
            elif command == 'select':
                if len(args) < 3:
                    print("Usage: select <category> <module> <algorithm>")
                else:
                    self.select_algorithm(args[0], args[1], args[2])
                    
            elif command == 'info':
                algorithm = args[0] if args else None
                self.display_algorithm_info(algorithm)
                
            elif command == 'run':
                self.run_algorithm(*args)
                
            elif command == 'visualize':
                if not args:
                    print("Usage: visualize <type>")
                    print("Available types: consensus_network, blockchain, topological_partitioning, network_generator")
                else:
                    self.run_visualization(args[0])
                    
            elif command == 'clear':
                self.clear_screen()
                
            else:
                print(f"Unknown command: {command}")
                print("Type 'help' for a list of commands.")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='BTQ Classical Algorithms Visualizer')
    parser.add_argument('--run', metavar='ALGORITHM', help='Run a specific algorithm directly')
    parser.add_argument('--list', action='store_true', help='List all available algorithms')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    if args.list:
        # List all algorithms and exit
        hub = ClassicalAlgorithmsHub()
        print("Available Algorithms:")
        for algo in hub.list_algorithms():
            print(f"  - {algo}")
        sys.exit(0)
        
    if args.run:
        # Run a specific algorithm and exit
        hub = ClassicalAlgorithmsHub()
        algorithm = hub.get_algorithm(args.run)
        if algorithm:
            print(f"Running algorithm: {args.run}")
            algorithm()
        else:
            print(f"Algorithm '{args.run}' not found.")
        sys.exit(0)
    
    # Run the interactive CLI
    cli = VisualizerCLI()
    cli.run_cli()
