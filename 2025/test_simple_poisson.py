#!/usr/bin/env python3
"""
Simple test to verify random Poisson function works in model_13_gl.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pysd

def test_model_and_plot_accidents():
    """Test the model and plot accidents to verify random Poisson is working"""
    
    print("🧪 TESTING MODEL WITH RANDOM POISSON:")
    print("=" * 50)
    
    try:
        # Load the model
        print("Loading model...")
        model = pysd.load("model_13_gl.py")
        print("✅ Model loaded successfully")
        
        # Run the model with a fixed seed for reproducibility
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
        print(f"❌ Error testing model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 SIMPLE RANDOM POISSON TEST")
    print("=" * 60)
    test_model_and_plot_accidents()
