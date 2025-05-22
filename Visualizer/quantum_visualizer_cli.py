#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quantum_visualizer_cli.py

A command line interface for visualizing and running quantum algorithms.
This script provides an interactive way to explore and run the various
implementations in the Quantum Algorithms directory.

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
import ast

# Add current directory to system path
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))
# Add parent directory to system path
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

# Import the QuantumAlgorithmsHub
from quantum_hub import QuantumAlgorithmsHub

# Set up matplotlib for interactive mode
plt.style.use('dark_background')

class QuantumVisualizerCLI:
    """
    Command Line Interface for visualizing quantum algorithms.
    Provides interactive functionality to explore and run algorithms.
    """
    
    def __init__(self):
        """Initialize the CLI with the Quantum Algorithms Hub"""
        self.hub = QuantumAlgorithmsHub()
        self.current_category = None
        self.current_module = None
        self.current_algorithm = None
        
    def display_welcome(self):
        """Display welcome message and basic instructions"""
        print("\n" + "="*80)
        print("  BTQ QUANTUM ALGORITHMS VISUALIZER")
        print("="*80)
        print("\nThis tool allows you to explore and run various quantum algorithms")
        print("from the Quantum Algorithms modules. You can visualize quantum network structures,")
        print("run quantum-enhanced simulations, and explore the different implementations.")
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
        print("  1. Run a quantum trust partitioner:")
        print("     > select Trust Partitioning quantum_trust_partitioner quantumNetworkPartitioner")
        print("     > run [[0,1,2],[3,4],[5]] {0:0.3,1:0.4,2:0.3,3:0.6,4:0.5,5:0.2} 0.5 16")
        print()
        print("  2. Create a quantum blockchain visualization:")
        print("     > select Blockchain Simulation quantumBlockchainVisualizer create_quantum_blockchain_visualization")
        print("     > run 5 12 30 0.6 True")
        print()
        print("  3. Visualize a quantum consensus network:")
        print("     > visualize quantum_consensus_network")
        print()
        print("  4. Visualize quantum generalized permutations:")
        print("     > visualize quantum_permutations")
        print()
        print("  5. Generate quantum vs. classical entropy comparison:")
        print("     > select Blockchain Simulation quantumBlockchainVisualizer create_quantum_vs_classical_entropy_plot")
        print("     > run")
    
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
            
        cat = category or self.current_category
        modules = self.hub.list_modules(cat)
        
        print(f"\nModules in {cat}:")
        for module in modules:
            print(f"  - {module}")
    
    def list_algorithms(self, category=None, module=None):
        """List all available algorithms in a category/module"""
        cat = category or self.current_category
        mod = module or self.current_module
        
        if not cat:
            print("Please specify a category or select one first.")
            return
            
        if not mod:
            print("Please specify a module or select one first.")
            return
            
        algos = self.hub.list_algorithms(cat, mod)
        
        print(f"\nAlgorithms in {cat}.{mod}:")
        for algo in algos:
            name = algo.split('.')[-1]
            print(f"  - {name}")
    
    def select_algorithm(self, category, module, algorithm):
        """Select a specific algorithm to work with"""
        # Check if category exists
        if category not in self.hub.list_categories():
            print(f"Category '{category}' not found.")
            return
            
        # Check if module exists in category
        modules = self.hub.list_modules(category)
        if module not in modules:
            print(f"Module '{module}' not found in category '{category}'.")
            return
            
        # Find algorithm in list
        full_key = None
        for algo_key in self.hub.list_algorithms(category, module):
            if algo_key.split('.')[-1] == algorithm:
                full_key = algo_key
                break
                
        if not full_key:
            print(f"Algorithm '{algorithm}' not found in {category}.{module}.")
            return
            
        # Set current selections
        self.current_category = category
        self.current_module = module
        self.current_algorithm = full_key
        
        print(f"Selected algorithm: {full_key}")
        
        # Show algorithm info
        self.show_algorithm_info(full_key)
    
    def show_algorithm_info(self, algorithm_key=None):
        """Show information about an algorithm"""
        key = algorithm_key or self.current_algorithm
        
        if not key:
            print("No algorithm selected. Please select an algorithm first.")
            return
            
        info = self.hub.get_algorithm_info(key)
        if not info:
            print(f"Algorithm '{key}' not found.")
            return
            
        print(f"\n{info['name']}:")
        print(f"  Type: {info['type']}")
        if info['signature']:
            print(f"  Signature: {info['name']}{info['signature']}")
        print("\nDescription:")
        print(info['docstring'])
    
    def run_algorithm(self, *args):
        """Run the selected algorithm with arguments"""
        if not self.current_algorithm:
            print("No algorithm selected. Please select an algorithm first.")
            return
            
        # Get the algorithm
        algo = self.hub.get_algorithm(self.current_algorithm)
        if not algo:
            print(f"Algorithm '{self.current_algorithm}' not found.")
            return
            
        # Parse the arguments
        parsed_args = []
        for arg in args:
            try:
                # Try to parse the argument as a Python expression
                parsed = ast.literal_eval(arg)
                parsed_args.append(parsed)
            except (SyntaxError, ValueError):
                # If parsing fails, use the argument as a string
                parsed_args.append(arg)
                
        try:
            # Run the algorithm
            print(f"Running {self.current_algorithm}...")
            result = algo(*parsed_args)
            
            # If the result is a matplotlib figure, show it
            if plt.get_fignums():
                print("Displaying visualization result...")
                plt.show()
            else:
                print("Result:", result)
                
        except Exception as e:
            print(f"Error running algorithm: {e}")
    
    def run_visualization(self, vis_type):
        """Run a predefined visualization"""
        try:
            if vis_type == "quantum_consensus_network":
                from quantum_hub import run_quantum_consensus_network_simulation
                network = run_quantum_consensus_network_simulation(num_nodes=15)
                network.visualize_network()
                plt.show()
                
            elif vis_type == "quantum_blockchain":
                from quantum_hub import run_quantum_blockchain_visualization
                run_quantum_blockchain_visualization(n_blocks=5, n_nodes_per_block=12, n_frames=30)
                plt.show()
                
            elif vis_type == "quantum_permutations":
                from quantum_hub import run_quantum_generalized_permutations
                run_quantum_generalized_permutations(elements=list(range(10)))
                plt.show()
                
            elif vis_type == "quantum_communities":
                from quantum_hub import run_quantum_community_optimizer
                run_quantum_community_optimizer(n_nodes=30, n_communities=4)
                plt.show()
                
            elif vis_type == "quantum_graph":
                from quantum_hub import run_quantum_graph_generator
                run_quantum_graph_generator(n_nodes=25, n_frames=40)
                plt.show()
                
            else:
                print(f"Visualization type '{vis_type}' not found.")
                print("Available types: quantum_consensus_network, quantum_blockchain, quantum_permutations, quantum_communities, quantum_graph")
        
        except Exception as e:
            print(f"Error running visualization: {e}")
    
    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def run_cli(self, script=None):
        """Run the CLI interactively or from a script"""
        self.display_welcome()
        
        if script:
            # Run commands from a script file
            with open(script, 'r') as f:
                commands = f.readlines()
                
            for cmd in commands:
                cmd = cmd.strip()
                if not cmd or cmd.startswith('#'):
                    continue
                    
                print(f"> {cmd}")
                self.process_command(cmd)
                
        else:
            # Interactive mode
            while True:
                try:
                    cmd = input("> ").strip()
                    if not cmd:
                        continue
                        
                    if cmd.lower() == "exit":
                        break
                        
                    self.process_command(cmd)
                    
                except KeyboardInterrupt:
                    print("\nExiting...")
                    break
                    
                except Exception as e:
                    print(f"Error: {e}")
    
    def process_command(self, cmd):
        """Process a command string"""
        parts = cmd.split()
        command = parts[0].lower()
        args = parts[1:]
        
        if command == "help":
            self.display_help()
            
        elif command == "examples":
            self.display_examples()
            
        elif command == "categories":
            self.list_categories()
            
        elif command == "modules":
            category = args[0] if args else None
            self.list_modules(category)
            
        elif command == "algos":
            category = args[0] if len(args) > 0 else None
            module = args[1] if len(args) > 1 else None
            self.list_algorithms(category, module)
            
        elif command == "select":
            if len(args) < 3:
                print("Usage: select category module algorithm")
                return
                
            category, module, algorithm = args[0], args[1], args[2]
            self.select_algorithm(category, module, algorithm)
            
        elif command == "info":
            algorithm = args[0] if args else None
            if algorithm:
                # Find the algorithm by name
                for algo_key in self.hub.list_algorithms():
                    if algo_key.split('.')[-1] == algorithm:
                        self.show_algorithm_info(algo_key)
                        break
                else:
                    print(f"Algorithm '{algorithm}' not found.")
            else:
                self.show_algorithm_info()
                
        elif command == "run":
            self.run_algorithm(*args)
            
        elif command == "visualize":
            if not args:
                print("Please specify a visualization type.")
                return
                
            self.run_visualization(args[0])
            
        elif command == "clear":
            self.clear_screen()
            
        else:
            print(f"Unknown command: {command}")
            print("Type 'help' for a list of commands.")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="BTQ Quantum Algorithms Visualizer CLI")
    parser.add_argument('--script', type=str, help='Path to a script file with commands')
    args = parser.parse_args()
    
    # Create and run the CLI
    cli = QuantumVisualizerCLI()
    cli.run_cli(args.script)
