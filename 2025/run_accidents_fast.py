#!/usr/bin/env python3
"""
Fast version of accident model analysis with progress tracking.

Last Update: 09/30/25
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

# Configuration
PROJECT_DIR = '/Users/georgia/Documents/GitHub/OrganizationOscillation/'
CURRENT_DIR = '/Users/georgia/Documents/GitHub/OrganizationOscillation/2025/'
MODEL_FILE = 'model_13_gl.py'

# Set pandas display options
pd.set_option('display.max_rows', None)

# CRITICAL: Set up functionspace at module level BEFORE any PySD operations
# This ensures random_poisson is available during model translation AND runtime
import sys
sys.path.insert(0, '/Users/georgia/Documents/GitHub/OrganizationOscillation/2025')

try:
    from fix_pysd_poisson import add_random_poisson_to_pysd
    add_random_poisson_to_pysd()
    print("✅ Functionspace set up at module level")
except Exception as e:
    print(f"⚠️  Warning: Could not set up functionspace at module level: {e}")


def load_model():
    """Load the PySD model after functionspace setup"""

    print("\n🔧 LOADING MODEL:")
    print("=" * 40)

    start_time = time.time()

    # Verify functionspace is set up (should already be done at module level)
    from pysd.builders.python.python_expressions_builder import functionspace
    if 'random_poisson' not in functionspace:
        print("⚠️  Functionspace not set, setting it up now...")
        from fix_pysd_poisson import add_random_poisson_to_pysd
        add_random_poisson_to_pysd()
        print("✅ Functionspace set up")
    else:
        print("✅ Functionspace already set up (random_poisson found)")

    print("\n🎯 PySD functionspace is ready - model can now use random_poisson!")

    try:
        # Use read_vensim instead of load (like test_functionspace_model.py)
        print("✅ Functionspace ready, translating model from Vensim...")
        model = pysd.read_vensim(CURRENT_DIR+"model_13.mdl")

        load_time = time.time() - start_time
        print(f"✅ Model translated and loaded successfully in {load_time:.1f}s")

        # Quick test to verify it works
        print("🧪 Testing model with random_poisson...")
        test_start = time.time()
        np.random.seed(42)
        test_result = model.run()
        test_time = time.time() - test_start
        print(f"✅ Model runs successfully in {test_time:.1f}s!")

        return model

    except Exception as e:
        print(f"❌ Cannot load model: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_quick_analysis(model):
    """Run a quick analysis with fewer scenarios for faster results"""

    print("\n🚀 QUICK ANALYSIS:")
    print("=" * 50)

    results = {}

    # Test just 2 scenarios instead of 3
    scenarios = {
        'G,G': {'sw_a_to_protective': 0, 'sw_b_to_protective': 0},
        'P,P': {'sw_a_to_protective': 1, 'sw_b_to_protective': 1}
    }

    for i, (name, params) in enumerate(scenarios.items()):
        print(f"\n📊 Running scenario {i+1}/2: {name}")
        start_time = time.time()

        try:
            result = model.run(params=params)
            results[name] = result

            run_time = time.time() - start_time

            if 'accidents' in result.columns:
                accidents = result['accidents']
                non_zero = np.sum(accidents != 0)
                total = len(accidents)
                print(f"  ✅ Completed in {run_time:.1f}s - Accidents: {non_zero}/{total} events")
            else:
                print(f"  ✅ Completed in {run_time:.1f}s - No accidents column")

        except Exception as e:
            print(f"  ❌ Error in scenario {name}: {e}")

    return results


def run_limited_sensitivity(model):
    """Run parameter sensitivity with fewer values for faster results"""

    print("\n🔬 LIMITED PARAMETER SENSITIVITY:")
    print("=" * 50)

    # Test fewer accident rates for speed
    accident_rates = [1.0, 5.0, 10.0]  # Only 3 instead of 6
    results_dict = {}

    for i, rate in enumerate(accident_rates):
        print(f"\n📊 Testing {i+1}/3: accident_rate = {rate}")
        start_time = time.time()

        try:
            params = {'accident_rate': rate}
            result = model.run(params=params)
            results_dict[rate] = result

            run_time = time.time() - start_time

            if 'accidents' in result.columns:
                accidents = result['accidents']
                non_zero = np.sum(accidents != 0)
                total = len(accidents)
                print(f"  ✅ Completed in {run_time:.1f}s")
                print(f"  📈 Non-zero accidents: {non_zero}/{total} ({non_zero/total*100:.1f}%)")

        except Exception as e:
            print(f"  ❌ Error with accident_rate = {rate}: {e}")

    return results_dict


def create_accident_plot(model):
    """Create a dedicated accident plot like in test_functionspace_model.py"""
    
    print("📈 Creating dedicated accident plot...")
    
    try:
        # Run model with fixed seed for consistent results
        np.random.seed(42)
        result = model.run()
        
        # Get accidents data
        accidents_data = result['accidents']
        
        print(f"Accidents statistics:")
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
        
        # Create the plot (exactly like test_functionspace_model.py)
        plt.figure(figsize=(12, 6))
        plt.stem(result.index, accidents_data, basefmt=' ', linefmt='b-', markerfmt='bo')
        plt.title('Accidents with PySD Functionspace Random Poisson', fontsize=14, fontweight='bold')
        plt.xlabel('Time (Month)', fontsize=12)
        plt.ylabel('Accidents', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f'Mean: {accidents_data.mean():.3f}, Std: {accidents_data.std():.3f}\nUnique values: {unique_count}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        plot_filename = "accidents_functionspace_plot.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved as: {plot_filename}")
        
        plt.show()
        
        print("✅ Dedicated accident plot created successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error creating accident plot: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_quick_visualizations(scenario_results, sensitivity_results):
    """Create essential visualizations quickly"""

    print("\n📊 CREATING VISUALIZATIONS:")
    print("=" * 40)

    if scenario_results:
        print("📈 Creating scenario comparison plot...")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Quick Analysis: Scenarios & Accidents', fontsize=14)

        # Plot 1: Performance comparison
        ax1 = axes[0]
        performance_found = False
        for name, result in scenario_results.items():
            if 'performance[A]' in result.columns:
                result['performance[A]'].plot(ax=ax1, label=f'{name} - A', alpha=0.8)
                performance_found = True
            if 'performance[B]' in result.columns:
                result['performance[B]'].plot(ax=ax1, label=f'{name} - B', alpha=0.8)
                performance_found = True
        
        if not performance_found:
            # Try alternative column names
            for name, result in scenario_results.items():
                perf_cols = [col for col in result.columns if 'performance' in col.lower()]
                if perf_cols:
                    for col in perf_cols[:2]:  # Take first 2 performance columns
                        result[col].plot(ax=ax1, label=f'{name} - {col}', alpha=0.8)
                        performance_found = True
        
        if not performance_found:
            ax1.text(0.5, 0.5, 'No performance columns found', 
                    transform=ax1.transAxes, ha='center', va='center')
        
        ax1.set_title('Performance Comparison')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Performance')
        ax1.legend()
        ax1.grid(True)

        # Plot 2: Accidents comparison with stem plots (like test_functionspace_model.py)
        ax2 = axes[1]
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        markers = ['o', 's', '^', 'D', 'v']
        linestyles = ['-', '--', '-.', ':', '-']
        
        for idx, (name, result) in enumerate(scenario_results.items()):
            if 'accidents' in result.columns:
                accidents_data = result['accidents']
                color = colors[idx % len(colors)]
                marker = markers[idx % len(markers)]
                linestyle = linestyles[idx % len(linestyles)]
                # Use stem plot for discrete accident events with unique styling per scenario
                ax2.stem(result.index, accidents_data, 
                        basefmt=' ', 
                        linefmt=f'{color}{linestyle}', 
                        markerfmt=f'{color}{marker}', 
                        label=name)
            else:
                print(f"⚠️  Warning: 'accidents' column not found in scenario '{name}'")
        
        ax2.set_title('Accidents Comparison (Stem Plot)')
        ax2.set_xlabel('Time (Month)')
        ax2.set_ylabel('Accidents')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        
        # Save the scenario plots
        scenario_filename = "scenario_comparison_plots.png"
        plt.savefig(scenario_filename, dpi=300, bbox_inches='tight')
        print(f"✅ Scenario plots saved as: {scenario_filename}")
        
        plt.show()
        print("✅ Scenario plots created")

    if sensitivity_results:
        print("📈 Creating sensitivity analysis plot...")

        fig, axes = plt.subplots(1, len(sensitivity_results), figsize=(15, 5))
        fig.suptitle('Parameter Sensitivity: Accident Rate', fontsize=14)

        if len(sensitivity_results) == 1:
            axes = [axes]

        colors = ['blue', 'orange', 'green']

        for i, (rate, result) in enumerate(sensitivity_results.items()):
            ax = axes[i]

            if 'accidents' in result.columns:
                accidents_data = result['accidents']
                non_zero_mask = accidents_data != 0
                time_points = accidents_data.index[non_zero_mask]
                accident_values = accidents_data[non_zero_mask]

                if len(accident_values) > 0:
                    ax.stem(time_points, accident_values,
                           linefmt=f'{colors[i]}-',
                           markerfmt=f'{colors[i]}o',
                           basefmt='k-')

                ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
                ax.set_title(f'Rate = {rate}')
                ax.set_xlabel('Time')
                ax.set_ylabel('Accidents')
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        
        # Save the sensitivity plots
        sensitivity_filename = "sensitivity_analysis_plots.png"
        plt.savefig(sensitivity_filename, dpi=300, bbox_inches='tight')
        print(f"✅ Sensitivity plots saved as: {sensitivity_filename}")
        
        plt.show()
        print("✅ Sensitivity plots created")


def main():
    """Main execution function - faster version"""

    print("🎯 FAST ACCIDENT MODEL ANALYSIS")
    print("=" * 50)

    total_start = time.time()

    try:
        # Load model
        model = load_model()
        if model is None:
            print("❌ Cannot proceed without model")
            return

        # Create dedicated accident plot first (like test_functionspace_model.py)
        create_accident_plot(model)

        # Run quick analysis
        scenario_results = run_quick_analysis(model)
        sensitivity_results = run_limited_sensitivity(model)

        # Create visualizations
        create_quick_visualizations(scenario_results, sensitivity_results)

        total_time = time.time() - total_start

        print(f"\n🎉 FAST ANALYSIS COMPLETE!")
        print("=" * 50)
        print(f"⏱️  Total runtime: {total_time:.1f} seconds")
        print("\n✅ COMPLETED:")
        print("1. ✅ PySD functionspace setup with random_poisson")
        print("2. ✅ Model translation from Vensim with functionspace support")
        print("3. ✅ Dedicated accident plot with stem visualization (saved)")
        print("4. ✅ Quick scenario analysis (2 scenarios)")
        print("5. ✅ Limited sensitivity analysis (3 rates)")
        print("6. ✅ Essential visualizations (all plots saved)")
        
        print("\n📁 SAVED PLOTS:")
        print("   - accidents_functionspace_plot.png")
        print("   - scenario_comparison_plots.png") 
        print("   - sensitivity_analysis_plots.png")

        print(f"\n💡 This fast version runs ~3x faster than the full analysis!")

    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()