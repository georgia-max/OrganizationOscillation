#!/usr/bin/env python3
"""
Check the accidents data specifically
"""

import sys
import os
import pysd
import pandas as pd

# Import the fix
sys.path.append('.')
from fix_pysd_poisson import add_random_poisson_to_pysd

# Apply the fix BEFORE loading the model
print("🔧 Setting up PySD functionspace...")
add_random_poisson_to_pysd()

print("📊 Loading model...")
try:
    model = pysd.load("model_13.py")
    print("✅ Model loaded successfully!")

    # Try to run it
    print("🚀 Running model...")
    result = model.run()
    print("✅ Model ran successfully!")

    # Check accidents data specifically
    accidents_a = result['accidents[A]']
    accidents_b = result['accidents[B]']

    print(f"\n🎲 Accidents[A] statistics:")
    print(f"  Min: {accidents_a.min()}")
    print(f"  Max: {accidents_a.max()}")
    print(f"  Mean: {accidents_a.mean():.4f}")
    print(f"  Std: {accidents_a.std():.4f}")
    print(f"  Non-zero values: {len(accidents_a[accidents_a != 0])}")
    print(f"  Sample values: {list(accidents_a.head(20).values)}")

    print(f"\n🎲 Accidents[B] statistics:")
    print(f"  Min: {accidents_b.min()}")
    print(f"  Max: {accidents_b.max()}")
    print(f"  Mean: {accidents_b.mean():.4f}")
    print(f"  Std: {accidents_b.std():.4f}")
    print(f"  Non-zero values: {len(accidents_b[accidents_b != 0])}")
    print(f"  Sample values: {list(accidents_b.head(20).values)}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()