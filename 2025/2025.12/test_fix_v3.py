#!/usr/bin/env python3
"""
Test script to regenerate model from .mdl after fixing functionspace
"""

import sys
import os
import pysd

# Import the fix
sys.path.append('.')
from fix_pysd_poisson import add_random_poisson_to_pysd

# Apply the fix BEFORE loading the model
print("🔧 Setting up PySD functionspace...")
add_random_poisson_to_pysd()

# Verify the functionspace was updated
try:
    from pysd.builders.python.python_expressions_builder import functionspace
    if 'random_poisson' in functionspace:
        print("✅ random_poisson function registered in PySD functionspace")
    else:
        print("❌ random_poisson function NOT found in functionspace")
        sys.exit(1)
except Exception as e:
    print(f"❌ Could not access functionspace: {e}")
    sys.exit(1)

# Remove the old .py file to force regeneration
if os.path.exists("model_13.py"):
    print("🗑️  Removing old model_13.py to force regeneration...")
    os.remove("model_13.py")

print("📊 Converting .mdl to .py with fixed functionspace...")
try:
    # Use read_vensim to force conversion from .mdl
    model = pysd.read_vensim("model_13.mdl")
    print("✅ Model converted successfully!")

    # Try to run it
    print("🚀 Running model...")
    result = model.run()
    print("✅ Model ran successfully!")

    print(f"Result shape: {result.shape}")
    print(f"Columns: {list(result.columns)}")

    # Check if accidents data looks reasonable
    if 'accidents' in result.columns:
        accidents_data = result['accidents']
        print(f"\n🎲 Accidents statistics:")
        print(f"  Min: {accidents_data.min()}")
        print(f"  Max: {accidents_data.max()}")
        print(f"  Mean: {accidents_data.mean():.4f}")
        print(f"  Std: {accidents_data.std():.4f}")
        print(f"  Non-zero values: {len(accidents_data[accidents_data != 0])}")

        # Show first few non-zero values
        non_zero = accidents_data[accidents_data != 0]
        if len(non_zero) > 0:
            print(f"  Sample non-zero values: {list(non_zero.head(10).values)}")
    else:
        print("⚠️  No 'accidents' column found. Available columns:")
        for col in result.columns:
            print(f"    - {col}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()