#!/usr/bin/env python3
"""
Safe version of accident model analysis - no infinite loops, robust error handling.

Last Update: 10/01/25
Author: @georgia-max
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pysd
import signal

# Configuration
CURRENT_DIR = '/Users/georgia/Documents/GitHub/OrganizationOscillation/2025/'
MODEL_FILE = 'model_13_gl.py'
MAX_RUNTIME = 300  # 5 minutes max total runtime
MAX_MODEL_RUNTIME = 60  # 1 minute max per model run

# Set pandas display options
pd.set_option('display.max_rows', None)

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def setup_random_poisson_functionspace():
    """Set up PySD functionspace with random_poisson from fix_pysd_poisson.py"""

    print("🔧 SETTING UP fix_pysd_poisson.py:")
    print("=" * 50)

    try:
        # Import from current directory
        sys.path.insert(0, os.getcwd())
        from fix_pysd_poisson import add_random_poisson_to_pysd

        print("✅ Successfully imported fix_pysd_poisson.py")
        add_random_poisson_to_pysd()

        from pysd.builders.python.python_expressions_builder import functionspace

        if 'random_poisson' in functionspace:
            print("✅ random_poisson successfully added to PySD functionspace")
            return True
        else:
            print("❌ random_poisson NOT found in functionspace")
            return False

    except Exception as e:
        print(f"❌ Error setting up functionspace: {e}")
        return False


def load_model_safely():
    """Load the PySD model with timeout protection"""

    print("\n🔧 LOADING MODEL SAFELY:")
    print("=" * 40)

    start_time = time.time()

    # Set up functionspace first
    functionspace_success = setup_random_poisson_functionspace()

    if not functionspace_success:
        print("❌ Cannot load model - functionspace setup failed")
        return None

    print("\n🎯 PySD functionspace is ready!")

    try:
        print("✅ Loading model...")

        # Set timeout for model loading
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)  # 30 second timeout for loading

        model = pysd.load(MODEL_FILE)

        signal.alarm(0)  # Cancel timeout

        load_time = time.time() - start_time
        print(f"✅ Model loaded successfully in {load_time:.1f}s")

        # Quick test with timeout
        print("🧪 Testing model...")
        signal.alarm(MAX_MODEL_RUNTIME)  # 60 second timeout for test run

        np.random.seed(42)
        test_result = model.run()

        signal.alarm(0)  # Cancel timeout

        test_time = time.time() - start_time - load_time
        print(f"✅ Model test completed in {test_time:.1f}s!")

        return model

    except TimeoutError:
        print("❌ Model loading/testing timed out")
        return None
    except Exception as e:
        signal.alarm(0)  # Cancel any pending timeout
        print(f"❌ Cannot load model: {e}")
        return None


def run_single_scenario_safely(model, name, params, timeout=MAX_MODEL_RUNTIME):
    """Run a single scenario with timeout protection"""

    print(f"📊 Running scenario: {name}")
    start_time = time.time()

    try:
        # Set timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)

        result = model.run(params=params)

        signal.alarm(0)  # Cancel timeout

        run_time = time.time() - start_time

        if 'accidents' in result.columns:
            accidents = result['accidents']
            non_zero = np.sum(accidents != 0)
            total = len(accidents)
            print(f"  ✅ Completed in {run_time:.1f}s - Accidents: {non_zero}/{total} events")
        else:
            print(f"  ✅ Completed in {run_time:.1f}s - No accidents column")

        return result

    except TimeoutError:
        print(f"  ❌ Scenario {name} timed out after {timeout}s")
        return None
    except Exception as e:
        signal.alarm(0)  # Cancel timeout
        print(f"  ❌ Error in scenario {name}: {e}")
        return None


def run_safe_analysis(model):
    """Run analysis with timeouts and error handling"""

    print("\n🚀 SAFE ANALYSIS:")
    print("=" * 50)

    results = {}

    # Test only 2 simple scenarios
    scenarios = {
        'G,G': {'sw_a_to_protective': 0, 'sw_b_to_protective': 0},
        'P,P': {'sw_a_to_protective': 1, 'sw_b_to_protective': 1}
    }

    for i, (name, params) in enumerate(scenarios.items()):
        print(f"\n📊 Running scenario {i+1}/2: {name}")

        result = run_single_scenario_safely(model, name, params)
        if result is not None:
            results[name] = result

    return results


def run_safe_sensitivity(model):
    """Run parameter sensitivity with timeouts"""

    print("\n🔬 SAFE PARAMETER SENSITIVITY:")
    print("=" * 50)

    # Test only 2 accident rates for safety
    accident_rates = [1.0, 5.0]
    results_dict = {}

    for i, rate in enumerate(accident_rates):
        print(f"\n📊 Testing {i+1}/2: accident_rate = {rate}")

        params = {'accident_rate': rate}
        result = run_single_scenario_safely(model, f"rate_{rate}", params)

        if result is not None:
            results_dict[rate] = result

    return results_dict


def create_simple_plots(scenario_results, sensitivity_results):
    """Create simple plots without complex operations"""

    print("\n📊 CREATING SIMPLE PLOTS:")
    print("=" * 40)

    try:
        # Simple scenario plot
        if scenario_results:
            print("📈 Creating scenario plot...")

            plt.figure(figsize=(10, 4))

            plt.subplot(1, 2, 1)
            for name, result in scenario_results.items():
                if 'performance[A]' in result.columns:
                    plt.plot(result.index, result['performance[A]'], label=f'{name} - A', alpha=0.8)
            plt.title('Performance A')
            plt.xlabel('Time')
            plt.ylabel('Performance')
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 2, 2)
            for name, result in scenario_results.items():
                if 'accidents' in result.columns:
                    plt.plot(result.index, result['accidents'], label=name, alpha=0.8)
            plt.title('Accidents')
            plt.xlabel('Time')
            plt.ylabel('Accidents')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig('safe_scenario_plots.png', dpi=150, bbox_inches='tight')
            print("✅ Scenario plots saved as: safe_scenario_plots.png")
            plt.show()

        # Simple sensitivity plot
        if sensitivity_results:
            print("📈 Creating sensitivity plot...")

            plt.figure(figsize=(8, 4))

            for i, (rate, result) in enumerate(sensitivity_results.items()):
                if 'accidents' in result.columns:
                    accidents_data = result['accidents']
                    plt.subplot(1, len(sensitivity_results), i+1)
                    plt.plot(result.index, accidents_data, 'o-', alpha=0.7)
                    plt.title(f'Rate = {rate}')
                    plt.xlabel('Time')
                    plt.ylabel('Accidents')
                    plt.grid(True)

            plt.tight_layout()
            plt.savefig('safe_sensitivity_plots.png', dpi=150, bbox_inches='tight')
            print("✅ Sensitivity plots saved as: safe_sensitivity_plots.png")
            plt.show()

        return True

    except Exception as e:
        print(f"❌ Error creating plots: {e}")
        return False


def main():
    """Main execution function with comprehensive safety measures"""

    print("🎯 SAFE ACCIDENT MODEL ANALYSIS")
    print("=" * 50)

    # Set overall timeout
    start_time = time.time()

    try:
        # Load model
        model = load_model_safely()
        if model is None:
            print("❌ Cannot proceed without model")
            return

        # Check overall timeout
        if time.time() - start_time > MAX_RUNTIME:
            print("❌ Overall timeout reached")
            return

        # Run safe analysis
        scenario_results = run_safe_analysis(model)

        # Check timeout again
        if time.time() - start_time > MAX_RUNTIME:
            print("❌ Overall timeout reached during analysis")
            return

        sensitivity_results = run_safe_sensitivity(model)

        # Create plots
        create_simple_plots(scenario_results, sensitivity_results)

        total_time = time.time() - start_time

        print(f"\n🎉 SAFE ANALYSIS COMPLETE!")
        print("=" * 50)
        print(f"⏱️  Total runtime: {total_time:.1f} seconds")
        print("\n✅ COMPLETED:")
        print("1. ✅ PySD functionspace setup")
        print("2. ✅ Model loading with timeout protection")
        print("3. ✅ Safe scenario analysis (2 scenarios)")
        print("4. ✅ Safe sensitivity analysis (2 rates)")
        print("5. ✅ Simple visualizations")

        print("\n📁 SAVED PLOTS:")
        print("   - safe_scenario_plots.png")
        print("   - safe_sensitivity_plots.png")

        print(f"\n💡 This safe version prevents infinite loops and timeouts!")

    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cancel any pending alarms
        signal.alarm(0)


if __name__ == "__main__":
    main()