#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
classical_hub.py

A centralized hub that provides an interface to access and run all the 
classical algorithm implementations from the Classical Algorithms folder.
This module serves as an entry point for using the various algorithms 
without having to work directly with individual implementation files.

Author: Jeffrey Morais, BTQ
"""

import os
import sys
import importlib.util
from pathlib import Path
import inspect

# Add current directory to system path
current_dir = Path(__file__).parent.absolute()
# Add parent directory to system path to allow imports from other modules
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

# Classical Algorithms subdirectories
CLASSICAL_MODULES = [
    "Network Graph Generator",
    "Consensus Iterations",
    "Trust Partitioning",
    "Community Optimization",
    "Generalized Permutations",
    "Blockchain Simulation"
]

class ClassicalAlgorithmsHub:
    """
    A hub for accessing and running Classical Algorithm implementations.
    Provides methods to list available algorithms, access specific 
    implementations, and run algorithm functions with parameters.
    """
    
    def __init__(self):
        """
        Initialize the Classical Algorithms Hub by loading all available
        modules and algorithms from the Classical Algorithms directory.
        """
        self.base_path = Path(__file__).parent.parent / "Classical Algorithms"
        self.modules = {}
        self.algorithms = {}
        
        # Load all modules and algorithms
        self._load_modules()
        
    def _load_modules(self):
        """
        Load all Python modules from the Classical Algorithms directory
        and its subdirectories, making them available for use.
        """
        print("Loading Classical Algorithm modules...")
        
        for module_dir in CLASSICAL_MODULES:
            module_path = self.base_path / module_dir
            if not module_path.exists():
                print(f"Warning: Module directory {module_dir} not found.")
                continue
                
            # Load all Python files in the module directory
            py_files = list(module_path.glob("*.py"))
            for py_file in py_files:
                if py_file.name.startswith("__") or py_file.name.startswith("test_"):
                    continue
                    
                # Import the module
                module_name = py_file.stem
                full_path = str(py_file)
                
                try:
                    spec = importlib.util.spec_from_file_location(module_name, full_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Store the module
                    category = module_dir
                    if category not in self.modules:
                        self.modules[category] = {}
                    
                    self.modules[category][module_name] = module
                    
                    # Extract callable functions from the module
                    self._extract_algorithms(category, module_name, module)
                    
                except Exception as e:
                    print(f"Error loading module {module_name}: {e}")
        
        print(f"Loaded {sum(len(mods) for mods in self.modules.values())} modules with {len(self.algorithms)} algorithms")
        
    def _extract_algorithms(self, category, module_name, module):
        """
        Extract callable functions and classes from a module and store them as algorithms.
        
        Parameters:
        -----------
        category : str
            The category (directory) the module belongs to
        module_name : str
            The name of the module
        module : module
            The imported Python module
        """
        # Get all callables from the module
        for name, obj in inspect.getmembers(module):
            # Skip private members and imported objects
            if name.startswith("_"):
                continue
                
            try:
                # Only include functions and classes defined in this module
                # In some cases obj.__module__ might not match exactly, so we check if it contains the module_name
                if (inspect.isfunction(obj) or inspect.isclass(obj)) and (
                    obj.__module__ == module.__name__ or module_name in obj.__module__
                ):
                    # Store the algorithm
                    algo_key = f"{category}.{module_name}.{name}"
                    self.algorithms[algo_key] = obj
            except (AttributeError, TypeError) as e:
                # Skip objects that cause errors when checking attributes
                continue
    
    def list_categories(self):
        """
        Get a list of all available algorithm categories.
        
        Returns:
        --------
        list
            List of category names
        """
        return list(self.modules.keys())
    
    def list_modules(self, category=None):
        """
        Get a list of all modules, optionally filtered by category.
        
        Parameters:
        -----------
        category : str, optional
            The category to filter by
            
        Returns:
        --------
        dict or list
            Dictionary of {category: [modules]} or list of modules if category is specified
        """
        if category:
            if category in self.modules:
                return list(self.modules[category].keys())
            return []
        
        # Return all modules by category
        return {category: list(modules.keys()) for category, modules in self.modules.items()}
    
    def list_algorithms(self, category=None, module=None):
        """
        Get a list of all available algorithms, optionally filtered by category and/or module.
        
        Parameters:
        -----------
        category : str, optional
            The category to filter by
        module : str, optional
            The module to filter by
            
        Returns:
        --------
        list
            List of algorithm names
        """
        results = []
        
        for algo_key in self.algorithms:
            parts = algo_key.split('.')
            algo_category, algo_module = parts[0], parts[1]
            
            if (category is None or algo_category == category) and \
               (module is None or algo_module == module):
                results.append(algo_key)
        
        return results
    
    def get_algorithm(self, algorithm_key):
        """
        Get a specific algorithm by its key.
        
        Parameters:
        -----------
        algorithm_key : str
            The full key of the algorithm in format "category.module.function"
            
        Returns:
        --------
        callable or None
            The algorithm function or class, or None if not found
        """
        return self.algorithms.get(algorithm_key)
    
    def get_algorithm_info(self, algorithm_key):
        """
        Get information about a specific algorithm.
        
        Parameters:
        -----------
        algorithm_key : str
            The full key of the algorithm
            
        Returns:
        --------
        dict
            Dictionary with algorithm information
        """
        algo = self.get_algorithm(algorithm_key)
        if not algo:
            return None
            
        # Extract information from docstring
        doc = inspect.getdoc(algo) or "No documentation available"
        signature = str(inspect.signature(algo)) if inspect.isfunction(algo) else ""
        
        # Get source code if available
        try:
            source = inspect.getsource(algo)
        except:
            source = "Source code not available"
        
        return {
            "name": algorithm_key.split('.')[-1],
            "full_name": algorithm_key,
            "docstring": doc,
            "signature": signature,
            "source": source,
            "type": "function" if inspect.isfunction(algo) else "class",
        }
    
    def run_algorithm(self, algorithm_key, *args, **kwargs):
        """
        Run a specific algorithm with the provided arguments.
        
        Parameters:
        -----------
        algorithm_key : str
            The full key of the algorithm
        *args, **kwargs
            Arguments to pass to the algorithm
            
        Returns:
        --------
        any
            The result of the algorithm
        """
        algo = self.get_algorithm(algorithm_key)
        if not algo:
            raise ValueError(f"Algorithm '{algorithm_key}' not found")
            
        # Run the algorithm
        return algo(*args, **kwargs)


# Helper functions for easy access to algorithms

def list_all_algorithms():
    """List all available classical algorithms"""
    hub = ClassicalAlgorithmsHub()
    return hub.list_algorithms()

def get_algorithm_by_name(full_name):
    """Get an algorithm by its full name (category.module.function)"""
    hub = ClassicalAlgorithmsHub()
    return hub.get_algorithm(full_name)

def run_trust_partitioner(networks, trust_matrix, node_security, network_security, **kwargs):
    """Run the trust partitioner algorithm"""
    hub = ClassicalAlgorithmsHub()
    partitioner = hub.get_algorithm("Trust Partitioning.trustPartitioner.networkPartitioner")
    return partitioner(networks, trust_matrix, node_security, network_security, **kwargs)

def run_blockchain_visualization(n_blocks=5, n_nodes_per_block=10, n_frames=40, **kwargs):
    """Run the blockchain visualization algorithm"""
    hub = ClassicalAlgorithmsHub()
    visualizer = hub.get_algorithm("Blockchain Simulation.blockchainVisualizer.create_blockchain_network_visualization")
    return visualizer(n_blocks, n_nodes_per_block, n_frames, **kwargs)

def run_consensus_network_simulation(num_nodes=10, honesty_distribution=None, network_connectivity=0.8, **kwargs):
    """Run a consensus network simulation"""
    hub = ClassicalAlgorithmsHub()
    network_class = hub.get_algorithm("Consensus Iterations.consensus_network.ConsensusNetwork")
    network = network_class(num_nodes, honesty_distribution, network_connectivity)
    return network

def run_topological_partitioning(graph, **kwargs):
    """Run the topological partitioning algorithm"""
    hub = ClassicalAlgorithmsHub()
    partitioner = hub.get_algorithm("Trust Partitioning.topologicalPartitioner.perform_topological_partitioning")
    return partitioner(graph, **kwargs)


if __name__ == "__main__":
    # Example usage
    hub = ClassicalAlgorithmsHub()
    
    print("\nAvailable Categories:")
    for category in hub.list_categories():
        print(f"- {category}")
    
    print("\nAvailable Modules in Trust Partitioning:")
    for module in hub.list_modules("Trust Partitioning"):
        print(f"- {module}")
    
    print("\nSample of Available Algorithms:")
    all_algos = hub.list_algorithms()
    for algo in all_algos[:5]:  # Show first 5 only
        print(f"- {algo}")
    
    print(f"\nTotal Algorithms: {len(all_algos)}")
