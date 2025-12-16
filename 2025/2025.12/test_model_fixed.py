#!/usr/bin/env python3
"""
Fixed test script for model_13.py with PySD functionspace random_poisson
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pysd

def setup_pysd_functionspace():
    """Set up PySD functionspace with random_poisson function"""

    print("🔧 SETTING UP PYSD FUNCTIONSPACE WITH RANDOM POISSON:")
    print("=" * 70)

    # Import the functionspace
    from pysd.builders.python.python_expressions_builder import functionspace

    # Define the random_poisson function following PySD's pattern
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

    return functionspace

def test_model_with_pysd_poisson():
    """Test model with the PySD functionspace random_poisson"""

    print(f"\n🧪 TESTING MODEL WITH PYSD FUNCTIONSPACE:")
    print("=" * 60)

    try:
        # Load the model
        print("Loading model...")
        model = pysd.load("model_13.py")
        print("✅ Model loaded successfully")

        # Run simulation
        print("Running simulation...")
        result = model.run()
        print("✅ Simulation completed successfully")

        # Check available columns
        print(f"\n📋 AVAILABLE COLUMNS:")
        print(f"Total columns: {len(result.columns)}")
        print(f"Sample columns: {list(result.columns)[:10]}")

        # Look for accident-related columns
        accident_cols = [col for col in result.columns if 'accident' in col.lower()]
        print(f"\nAccident-related columns: {accident_cols}")

        # Check accidents with subscripts (PySD creates columns like 'accidents[A]', 'accidents[B]')
        if 'accidents[A]' in result.columns and 'accidents[B]' in result.columns:
            accidents_a = result['accidents[A]']
            accidents_b = result['accidents[B]']
            # Combine both goals for total accidents
            accidents_total = accidents_a + accidents_b

            print(f"\n📊 ACCIDENTS ANALYSIS:")
            print("-" * 40)
            print(f"Accidents A - Min: {accidents_a.min()}, Max: {accidents_a.max()}, Mean: {accidents_a.mean():.4f}")
            print(f"Accidents B - Min: {accidents_b.min()}, Max: {accidents_b.max()}, Mean: {accidents_b.mean():.4f}")
            print(f"Total - Min: {accidents_total.min()}, Max: {accidents_total.max()}, Mean: {accidents_total.mean():.4f}")
            print(f"Total Std: {accidents_total.std():.4f}")
            print(f"Total unique values: {sorted(accidents_total.unique())}")

            # Count non-zero accidents
            non_zero = np.sum(accidents_total != 0)
            total = len(accidents_total)
            print(f"Non-zero accidents: {non_zero}/{total} ({non_zero/total*100:.2f}%)")

            # Show first 20 accident values
            print(f"\nFirst 20 total accident values:")
            print(accidents_total.head(20).tolist())

            return True, accidents_total, result

        elif 'accidents' in result.columns:
            accidents_data = result['accidents']
            print(f"\n📊 ACCIDENTS ANALYSIS:")
            print("-" * 40)
            print(f"Min: {accidents_data.min()}")
            print(f"Max: {accidents_data.max()}")
            print(f"Mean: {accidents_data.mean():.4f}")
            print(f"Std: {accidents_data.std():.4f}")
            print(f"Unique values: {sorted(accidents_data.unique())}")

            # Count non-zero accidents
            non_zero = np.sum(accidents_data != 0)
            total = len(accidents_data)
            print(f"Non-zero accidents: {non_zero}/{total} ({non_zero/total*100:.2f}%)")

            # Show first 20 accident values
            print(f"\nFirst 20 accident values:")
            print(accidents_data.head(20).tolist())

            return True, accidents_data, result

        else:
            print("❌ No accident columns found in results")
            print(f"Available columns: {list(result.columns)}")
            return False, None, None

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def visualize_results(success, accidents_data, result):
    """Create visualization if test was successful"""

    if not success:
        print("❌ Cannot create visualization - model test failed")
        return

    # Create comprehensive visualization
    plt.figure(figsize=(15, 10))

    # Plot 1: Accidents over time
    plt.subplot(2, 3, 1)
    accidents_data.plot(title='Accidents Over Time (PySD Functionspace)', color='blue')
    plt.xlabel('Time (Month)')
    plt.ylabel('Accidents')
    plt.grid(True)

    # Plot 2: Accident shock level
    plt.subplot(2, 3, 2)
    shock_cols = [col for col in result.columns if 'shock level' in col.lower()]
    if shock_cols:
        result[shock_cols[0]].plot(title='Accident Shock Level', color='red')
    plt.xlabel('Time (Month)')
    plt.ylabel('Shock Level')
    plt.grid(True)

    # Plot 3: Performance impact
    plt.subplot(2, 3, 3)
    perf_cols = [col for col in result.columns if 'performance[' in col.lower()]
    for col in perf_cols[:2]:  # Show first 2 performance columns
        result[col].plot(label=col, alpha=0.8)
    plt.title('Performance Impact')
    plt.xlabel('Time (Month)')
    plt.ylabel('Performance')
    plt.legend()
    plt.grid(True)

    # Plot 4: Resources impact
    plt.subplot(2, 3, 4)
    resource_cols = [col for col in result.columns if 'resources[' in col.lower()]
    for col in resource_cols[:2]:  # Show first 2 resource columns
        result[col].plot(label=col, alpha=0.8)
    plt.title('Resources Impact')
    plt.xlabel('Time (Month)')
    plt.ylabel('Resources')
    plt.legend()
    plt.grid(True)

    # Plot 5: Aspiration dynamics
    plt.subplot(2, 3, 5)
    asp_cols = [col for col in result.columns if 'aspiration[' in col.lower()]
    for col in asp_cols[:2]:  # Show first 2 aspiration columns
        result[col].plot(label=col, alpha=0.8)
    plt.title('Aspiration Dynamics')
    plt.xlabel('Time (Month)')
    plt.ylabel('Aspiration')
    plt.legend()
    plt.grid(True)

    # Plot 6: Combined performance
    plt.subplot(2, 3, 6)
    combined_cols = [col for col in result.columns if 'combined performance' in col.lower()]
    if combined_cols:
        result[combined_cols[0]].plot(title='Combined Performance', color='green')
    plt.xlabel('Time (Month)')
    plt.ylabel('Combined Performance')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print("📊 Model visualization completed!")

if __name__ == "__main__":
    # Set up PySD functionspace
    setup_pysd_functionspace()

    # Test the model
    success, accidents_data, result = test_model_with_pysd_poisson()

    # Create visualization
    visualize_results(success, accidents_data, result)

    print("\n🎉 Test completed!")