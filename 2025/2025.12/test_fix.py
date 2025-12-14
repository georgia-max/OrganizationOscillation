#!/usr/bin/env python3
"""
Test script to verify the fix_pysd_poisson.py works correctly
"""

import sys
import os
sys.path.append('.')

# Import the fix
from fix_pysd_poisson import add_random_poisson_to_pysd

# Apply the fix
print("🔧 Setting up PySD functionspace...")
add_random_poisson_to_pysd()

# Now try to load the model
import pysd
print("📊 Loading model...")
try:
    model = pysd.load("model_13.mdl")
    print("✅ Model loaded successfully!")

    # Try to run it
    print("🚀 Running model...")
    result = model.run()
    print("✅ Model ran successfully!")

    print(f"Result shape: {result.shape}")
    print(f"Columns: {list(result.columns)}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()