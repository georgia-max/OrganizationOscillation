#!/usr/bin/env python3
"""
Test script to verify random Poisson function is working in model_13_gl.py
This script loads the model and creates a simple plot of accidents over time.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pysd

# Add the current directory to Python path to import fix_pysd_poisson
sys.path.append('/Users/georgia/Documents/GitHub/OrganizationOscillation')

def setup_random_poisson_functionspace():
    """Set up PySD functionspace with random_poisson from fix_pysd_poisson.py"""
    
    print("🔧 SETTING UP PYSD FUNCTIONSPACE:")
    print("=" * 50)
    
    try:
        # Import the fix_pysd_poisson module
        from fix_pysd_poisson import add_random_poisson_to_pysd
        
        print("✅ Successfully imported fix_pysd_poisson.py")
        
        # Use the function from fix_pysd_poisson.py
        add_random_poisson_to_pysd()
        
        # Verify it was added
        from pysd.builders.python.python_expressions_builder import functionspace
        
        if 'random_poisson' in functionspace:
            print("✅ random_poisson successfully added to PySD functionspace")
            print(f"Expression: {functionspace['random_poisson'][0]}")
            print(f"Modules: {functionspace['random_poisson'][1]}")
            
            # Test the functionspace version
            print("\nTesting functionspace random_poisson:")
            test_params = {
                'min_val': 0,
                'max_val': 5, 
                'mean_val': 1.0 * 0.0625,
                'shift_val': 0,
                'stretch_val': 1,
                'seed_val': 42
            }
            
            # Test with different seeds
            results = []
            for seed in range(10):
                np.random.seed(seed)
                # Create the expression from functionspace
                expression = functionspace['random_poisson'][0]
                # Replace placeholders
                size = (1,)
                expr = expression % {
                    '0': test_params['min_val'],
                    '1': test_params['max_val'], 
                    '2': test_params['mean_val'],
                    '3': test_params['shift_val'],
                    '4': test_params['stretch_val'],
                    'size': size
                }
                result = eval(expr)[0]  # Extract single value
                results.append(result)
                print(f"  Seed {seed}: {result}")
            
            unique_results = len(set(results))
            print(f"\nFunctionspace unique results: {unique_results} out of 10 tests")
            
            if unique_results > 1:
                print("✅ Functionspace random_poisson is working!")
            else:
                print("⚠️  Functionspace random_poisson has low variation (expected with low lambda)")
            
            return True
        else:
            print("❌ random_poisson NOT found in functionspace")
            return False
            
    except ImportError as e:
        print(f"❌ Could not import fix_pysd_poisson.py: {e}")
        print("Make sure fix_pysd_poisson.py is in the project directory")
        return False
    except Exception as e:
        print(f"❌ Error setting up functionspace: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_and_plot_accidents():
    """Test the model and plot accidents to verify random Poisson is working"""
    
    print("\n🎯 TESTING MODEL WITH RANDOM POISSON:")
    print("=" * 50)
    
    try:
        # Load the model (functionspace should already be set up)
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

def main():
    """Main function to run the test"""
    
    print("🧪 TESTING RANDOM POISSON FUNCTION IN MODEL_13_GL.PY")
    print("=" * 60)
    
    # CRITICAL: Set up the functionspace FIRST, before any model loading
    functionspace_success = setup_random_poisson_functionspace()
    
    if functionspace_success:
        print("\n🎯 PySD functionspace is ready - testing full model...")
        test_model_and_plot_accidents()
    else:
        print("\n❌ Functionspace setup failed - cannot test model!")
        return False
    
    print("\n🎉 Test completed!")
    return True

if __name__ == "__main__":
    main()
