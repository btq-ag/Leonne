#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quantum_hub_test.py

A simple test script to demonstrate that the quantum hub can list
available algorithms even if it can't load all of them due to import issues.

Author: Jeffrey Morais, BTQ
"""

import os
import sys
from pathlib import Path

# Add current directory to system path
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

# Import the QuantumAlgorithmsHub
from quantum_hub import QuantumAlgorithmsHub

def test_quantum_hub():
    """Test the quantum hub by listing available categories and algorithms"""
    hub = QuantumAlgorithmsHub()
    
    print("\n==== Testing Quantum Algorithms Hub ====")
    
    print("\nAvailable Categories:")
    categories = hub.list_categories()
    for category in categories:
        print(f"- {category}")
    
    print("\nTotal Number of Categories:", len(categories))
    
    all_modules = {}
    for category in categories:
        modules = hub.list_modules(category)
        if modules:
            all_modules[category] = modules
            
    print("\nSuccessfully Loaded Modules by Category:")
    for category, modules in all_modules.items():
        print(f"- {category}: {len(modules)} modules")
        for module in modules:
            print(f"  * {module}")
    
    print("\nTotal Number of Loaded Modules:", sum(len(modules) for modules in all_modules.values()))
    
    print("\nTotal Number of Loaded Algorithms:", len(hub.algorithms))
    
    if hub.algorithms:
        print("\nSample of Available Algorithms:")
        for algo in list(hub.algorithms.keys())[:10]:  # Show first 10 only
            print(f"- {algo}")
    
    print("\n==== Test Completed ====")

if __name__ == "__main__":
    test_quantum_hub()
