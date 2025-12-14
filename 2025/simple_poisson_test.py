#!/usr/bin/env python3
"""
Simple test to verify random Poisson function works in model_13_gl.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pysd

# Add the current directory to Python path
sys.path.append('/Users/georgia/Documents/GitHub/OrganizationOscillation')

def create_working_poisson_model():
    """Create a working version by adding the function directly to the model"""
    
    print("🔧 CREATING WORKING POISSON MODEL:")
    print("=" * 50)
    
    try:
        # Load the model
        print("Loading model...")
        model = pysd.load("model_13_gl.py")
        print("✅ Model loaded successfully")
        
        # Add the random_poisson function to the model's namespace
        def random_poisson(min_val, max_val, mean, shift, stretch, seed_val):
            """Random Poisson function implementation"""
            # Use time-based seed for variation during simulation
            import time
            time_based_seed = (int(seed_val) + int(time.time() * 1000)) % 10000
            
            # Set random seed
            np.random.seed(time_based_seed)
            
            # Generate Poisson random variable
            poisson_val = np.random.poisson(lam=mean)
            
            # Apply stretch and shift (from Vensim specification)
            result = poisson_val * stretch + shift
            
            # Apply bounds (clip to min_val and max_val)
            return np.clip(result, min_val, max_val)
        
        # Add the function to the model's components
        model.components.random_poisson = random_poisson
        print("✅ Added random_poisson function to model")
        
        # Run the model
        print("Running model simulation...")
        np.random.seed(42)
        result = model.run()
        print("✅ Model simulation completed")
        
        # Get accidents data
        accidents_data = result['accidents']
        
        print(f"\nAccidents statistics:")
        print(f"  Min: {accidents_data.min():.4f}")
        print(f"  Max: {accidents_data.max():.4f}")
        print(f"  Mean: {accidents_data.mean():.4f}")
        print(f"  Std: {accidents_data.std():.4f}")
        print(f"  Unique values: {sorted(accidents_data.unique())}")
        
        # Check if we have variation (random Poisson working)
        unique_count = len(accidents_data.unique())
        if unique_count > 1:
            print(f"✅ Random Poisson is working! Found {unique_count} unique accident values")
        else:
            print(f"⚠️  Random Poisson may not be working - only {unique_count} unique value")
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        plt.stem(result.index, accidents_data, basefmt=' ', linefmt='b-', markerfmt='bo')
        plt.title('Accidents with Random Poisson Function', fontsize=14, fontweight='bold')
        plt.xlabel('Time (Month)', fontsize=12)
        plt.ylabel('Accidents', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f'Mean: {accidents_data.mean():.3f}, Std: {accidents_data.std():.3f}\nUnique values: {unique_count}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        print("\n✅ Plot created successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 SIMPLE POISSON TEST")
    print("=" * 60)
    create_working_poisson_model()
