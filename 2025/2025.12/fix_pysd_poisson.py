#!/usr/bin/env python3
"""
Fix PySD to support RANDOM POISSON function by adding it to the functionspace.
This script patches PySD to add support for random_poisson function.
"""

import pysd
import numpy as np
from scipy import stats

def add_random_poisson_to_pysd():
    """Add random_poisson function to PySD's functionspace"""
    
    # Import the functionspace
    from pysd.builders.python.python_expressions_builder import functionspace
    
    # Define the random_poisson function following PySD's pattern
    # Vensim RANDOM POISSON(min, max, mean, shift, stretch, seed)
    # PySD expects: random_poisson(min, max, mean, shift, stretch, seed)
    
    random_poisson_expression = """
    np.clip(
        np.random.poisson(lam=%(2)s, size=%(size)s) * %(4)s + %(3)s,
        %(0)s, %(1)s
    )
    """.strip()
    
    # Add to functionspace with proper modules
    functionspace['random_poisson'] = (
        random_poisson_expression,
        (('numpy',),)
    )
    
    print("✅ Added random_poisson to PySD functionspace")
    print(f"Expression: {random_poisson_expression}")
    print(f"Modules: {functionspace['random_poisson'][1]}")

def test_random_poisson_function():
    """Test the random_poisson function"""
    
    print("\n🧪 TESTING RANDOM POISSON FUNCTION:")
    print("=" * 50)
    
    # Test parameters (from your Vensim model)
    min_val = 0
    max_val = 5
    mean = 1.0 * 0.0625  # accident_rate * TIME_STEP
    shift = 0
    stretch = 1
    size = (1,)  # Single value
    
    # Create the expression manually
    expression = f"""
    np.clip(
        np.random.poisson(lam={mean}, size={size}) * {stretch} + {shift},
        {min_val}, {max_val}
    )
    """.strip()
    
    print(f"Test parameters:")
    print(f"  min: {min_val}")
    print(f"  max: {max_val}")
    print(f"  mean: {mean}")
    print(f"  shift: {shift}")
    print(f"  stretch: {stretch}")
    print(f"  size: {size}")
    
    print(f"\nTesting with different seeds:")
    for seed in range(10):
        np.random.seed(seed)
        result = eval(expression)
        print(f"  Seed {seed}: {result}")
    
    print(f"\nTesting with higher mean (more variation):")
    mean_high = 1.0  # Higher mean
    expression_high = f"""
    np.clip(
        np.random.poisson(lam={mean_high}, size={size}) * {stretch} + {shift},
        {min_val}, {max_val}
    )
    """.strip()
    
    for seed in range(10):
        np.random.seed(seed)
        result = eval(expression_high)
        print(f"  Seed {seed}: {result}")

def create_corrected_model():
    """Create a corrected model using the proper PySD approach"""
    
    print(f"\n🔧 CREATING CORRECTED MODEL:")
    print("=" * 50)
    
    # First, add random_poisson to PySD
    add_random_poisson_to_pysd()
    
    # Now try to translate the model again
    try:
        print("Translating model with random_poisson support...")
        model = pysd.read_vensim('2025/model_13.mdl')
        print("✅ Model translated successfully!")
        
        # Try to run it
        print("Testing model run...")
        result = model.run()
        print("✅ Model runs successfully!")
        
        # Check accidents
        if 'accidents' in result.columns:
            accidents_data = result['accidents']
            print(f"\nAccidents statistics:")
            print(f"  Min: {accidents_data.min()}")
            print(f"  Max: {accidents_data.max()}")
            print(f"  Mean: {accidents_data.mean():.4f}")
            print(f"  Std: {accidents_data.std():.4f}")
            print(f"  Unique values: {sorted(accidents_data.unique())}")
        
        return model, result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    print("🔧 FIXING PYSD RANDOM POISSON SUPPORT")
    print("=" * 60)
    
    # Test the function first
    test_random_poisson_function()
    
    # Try to create corrected model
    model, result = create_corrected_model()
    
    if model is not None:
        print(f"\n✅ SUCCESS!")
        print("PySD now supports RANDOM POISSON function!")
        print("The model should now work correctly with proper accident variation.")
    else:
        print(f"\n❌ FAILED!")
        print("Could not fix the PySD random_poisson support.")
