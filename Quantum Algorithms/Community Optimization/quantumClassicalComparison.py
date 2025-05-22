#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quantumClassicalComparison.py

This script creates side-by-side visualizations comparing classical and quantum
network communities, highlighting the differences and advantages of quantum approaches.
It directly loads visualizations from both approaches and creates combined comparative images.

Author: Jeffrey Morais, BTQ
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import os
from PIL import Image, ImageDraw, ImageFont
import imageio
from tqdm import tqdm

# Create output directory if it doesn't exist
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(output_dir, exist_ok=True)

# Define paths for classical and quantum visualizations
classical_dir = os.path.join("c:", os.sep, "Users", "hunkb", "OneDrive", "Desktop", "btq", "OS", "Moon", "Classical Algorithms", "Community Optimization", "network_animations")
quantum_dir = output_dir

def create_comparison_image(classical_name, quantum_name, comparison_name, title):
    """
    Creates a side-by-side comparison of classical and quantum network visualizations
    
    Args:
        classical_name: Filename of the classical visualization
        quantum_name: Filename of the quantum visualization
        comparison_name: Filename for the output comparison
        title: Title for the comparison image
    """
    try:
        # Open classical and quantum images
        classical_path = os.path.join(classical_dir, classical_name)
        quantum_path = os.path.join(quantum_dir, quantum_name)
        
        if not os.path.exists(classical_path):
            print(f"Warning: Classical image not found at {classical_path}")
            return
        
        if not os.path.exists(quantum_path):
            print(f"Warning: Quantum image not found at {quantum_path}")
            return
        
        classical_img = Image.open(classical_path)
        quantum_img = Image.open(quantum_path)
        
        # Resize if needed to ensure same dimensions
        width = max(classical_img.width, quantum_img.width)
        height = max(classical_img.height, quantum_img.height)
        
        if classical_img.width != width or classical_img.height != height:
            classical_img = classical_img.resize((width, height), Image.LANCZOS)
        
        if quantum_img.width != width or quantum_img.height != height:
            quantum_img = quantum_img.resize((width, height), Image.LANCZOS)
        
        # Create a new image with both side by side
        comparison_img = Image.new('RGB', (2 * width + 20, height + 40), color='white')
        
        # Add title
        draw = ImageDraw.Draw(comparison_img)
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except:
            # Fallback if Arial not available
            font = ImageFont.load_default()
        
        draw.text((width, 10), title, fill="black", font=font, anchor="mt")
        
        # Add "Classical" and "Quantum" labels
        draw.text((width // 2, 30), "Classical Approach", fill="black", font=font, anchor="mt")
        draw.text((width * 3 // 2 + 20, 30), "Quantum-Enhanced Approach", fill="black", font=font, anchor="mt")
        
        # Paste the two images
        comparison_img.paste(classical_img, (0, 40))
        comparison_img.paste(quantum_img, (width + 20, 40))
        
        # Save the comparison
        comparison_path = os.path.join(output_dir, comparison_name)
        comparison_img.save(comparison_path)
        print(f"Comparison image saved to {comparison_path}")
        
    except Exception as e:
        print(f"Error creating comparison image: {e}")

def create_comparison_animation(classical_name, quantum_name, comparison_name, title):
    """
    Creates a side-by-side animation comparing classical and quantum network visualizations
    
    Args:
        classical_name: Filename of the classical animation
        quantum_name: Filename of the quantum animation
        comparison_name: Filename for the output comparison
        title: Title for the comparison animation
    """
    try:
        # Open classical and quantum animations
        classical_path = os.path.join(classical_dir, classical_name)
        quantum_path = os.path.join(quantum_dir, quantum_name)
        
        if not os.path.exists(classical_path):
            print(f"Warning: Classical animation not found at {classical_path}")
            return
        
        if not os.path.exists(quantum_path):
            print(f"Warning: Quantum animation not found at {quantum_path}")
            return
        
        # Read both GIFs
        classical_frames = []
        quantum_frames = []
        
        # Load classical frames
        classical_gif = Image.open(classical_path)
        try:
            while True:
                classical_frames.append(classical_gif.copy())
                classical_gif.seek(classical_gif.tell() + 1)
        except EOFError:
            pass
        
        # Load quantum frames
        quantum_gif = Image.open(quantum_path)
        try:
            while True:
                quantum_frames.append(quantum_gif.copy())
                quantum_gif.seek(quantum_gif.tell() + 1)
        except EOFError:
            pass
        
        # Determine the number of frames to use (minimum of both)
        n_frames = min(len(classical_frames), len(quantum_frames))
        
        # Resize if needed to ensure same dimensions
        width = max(classical_frames[0].width, quantum_frames[0].width)
        height = max(classical_frames[0].height, quantum_frames[0].height)
        
        # Create comparison frames
        comparison_frames = []
        for i in tqdm(range(n_frames), desc=f"Creating {comparison_name}"):
            # Get frames and resize if needed
            classical_frame = classical_frames[i]
            quantum_frame = quantum_frames[i]
            
            if classical_frame.width != width or classical_frame.height != height:
                classical_frame = classical_frame.resize((width, height), Image.LANCZOS)
            
            if quantum_frame.width != width or quantum_frame.height != height:
                quantum_frame = quantum_frame.resize((width, height), Image.LANCZOS)
            
            # Create a new image with both side by side
            comparison_frame = Image.new('RGB', (2 * width + 20, height + 40), color='white')
            
            # Add title
            draw = ImageDraw.Draw(comparison_frame)
            try:
                font = ImageFont.truetype("arial.ttf", 18)
            except:
                # Fallback if Arial not available
                font = ImageFont.load_default()
            
            draw.text((width, 10), title, fill="black", font=font, anchor="mt")
            
            # Add "Classical" and "Quantum" labels
            draw.text((width // 2, 30), "Classical Approach", fill="black", font=font, anchor="mt")
            draw.text((width * 3 // 2 + 20, 30), "Quantum-Enhanced Approach", fill="black", font=font, anchor="mt")
            
            # Paste the two frames
            comparison_frame.paste(classical_frame, (0, 40))
            comparison_frame.paste(quantum_frame, (width + 20, 40))
            
            # Convert to RGB (required for some PIL operations)
            comparison_frame = comparison_frame.convert("RGB")
            
            # Add to list of frames
            comparison_frames.append(comparison_frame)
        
        # Save the comparison animation
        comparison_path = os.path.join(output_dir, comparison_name)
        comparison_frames[0].save(
            comparison_path,
            save_all=True,
            append_images=comparison_frames[1:],
            optimize=False,
            duration=200,  # Duration between frames in milliseconds
            loop=0  # Loop forever
        )
        
        print(f"Comparison animation saved to {comparison_path}")
        
    except Exception as e:
        print(f"Error creating comparison animation: {e}")

def create_all_comparisons():
    """Create all comparison images and animations"""
    print("Creating quantum vs classical network comparison visualizations...")
    
    # Create side-by-side comparison animations
    create_comparison_animation(
        "random_network.gif", 
        "quantum_random_network.gif", 
        "comparison_random_network.gif",
        "Random Network: Classical vs Quantum"
    )
    
    create_comparison_animation(
        "community_network.gif", 
        "quantum_community_network.gif", 
        "comparison_community_network.gif",
        "Community Structure: Classical vs Quantum"
    )
    
    create_comparison_animation(
        "small_world_network.gif", 
        "quantum_small_world_network.gif", 
        "comparison_small_world_network.gif",
        "Small-World Network: Classical vs Quantum"
    )
    
    create_comparison_animation(
        "hub_network.gif", 
        "quantum_hub_network.gif", 
        "comparison_hub_network.gif",
        "Hub Network: Classical vs Quantum"
    )
    
    create_comparison_animation(
        "spatial_network.gif", 
        "quantum_spatial_network.gif", 
        "comparison_spatial_network.gif",
        "Spatial Network: Classical vs Quantum"
    )
    
    # Create comparison for statistics
    create_comparison_animation(
        "network_statistics_evolution.gif", 
        "quantum_network_statistics_evolution.gif", 
        "comparison_network_statistics.gif",
        "Network Statistics Evolution: Classical vs Quantum"
    )
    
    print("All comparison visualizations completed!")

if __name__ == "__main__":
    create_all_comparisons()
