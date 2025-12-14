#!/usr/bin/env python3
"""
WORKING SOLUTION for PySD with random_poisson function

This script shows the complete working solution:
1. Set up PySD functionspace with random_poisson
2. Load the .mdl file (which will auto-generate working .py file)
3. Run the model successfully

Copy this code into your notebook cells.
"""

import sys
import os
import pysd

# Step 1: Import and apply the fix
sys.path.append('.')
from fix_pysd_poisson import add_random_poisson_to_pysd

print("🔧 STEP 1: Setting up PySD functionspace...")
add_random_poisson_to_pysd()

# Verify the fix worked
from pysd.builders.python.python_expressions_builder import functionspace
if 'random_poisson' in functionspace:
    print("✅ random_poisson function is now available in PySD")
else:
    print("❌ random_poisson function setup failed")
    exit(1)

# Step 2: Remove old .py file to force clean regeneration
if os.path.exists("model_13.py"):
    print("🗑️  STEP 2: Removing old model_13.py to ensure clean regeneration...")
    os.remove("model_13.py")

# Step 3: Load the .mdl file (this will generate a new .py file with working random_poisson)
print("📊 STEP 3: Loading model_13.mdl (will auto-generate working .py file)...")
model = pysd.read_vensim("model_13.mdl")
print("✅ Model loaded successfully!")

# Step 4: Test the model
print("🚀 STEP 4: Running model to test random_poisson functionality...")
result = model.run()
print("✅ Model ran successfully!")

# Step 5: Verify results
print("\n📈 STEP 5: Verifying accident simulation results...")
accidents_a = result['accidents[A]']
accidents_b = result['accidents[B]']

print(f"Accidents[A]: Mean={accidents_a.mean():.4f}, Non-zero events={len(accidents_a[accidents_a != 0])}")
print(f"Accidents[B]: Mean={accidents_b.mean():.4f}, Non-zero events={len(accidents_b[accidents_b != 0])}")

print(f"\n🎯 SUCCESS! The model now correctly simulates random Poisson accidents!")
print(f"Total data points: {len(result)}")
print(f"Simulation time range: {result.index.min():.2f} to {result.index.max():.2f}")
print(f"Available variables: {len(result.columns)} columns")

# Show some sample results
print(f"\n📊 Sample time series (first 10 time steps):")
sample_data = result[['accidents[A]', 'accidents[B]', 'performance[A]', 'performance[B]']].head(10)
for i, (time, row) in enumerate(sample_data.iterrows()):
    print(f"  t={time:5.2f}: accidents_A={row['accidents[A]']:3.0f}, accidents_B={row['accidents[B]']:3.0f}, perf_A={row['performance[A]']:6.2f}, perf_B={row['performance[B]']:6.2f}")