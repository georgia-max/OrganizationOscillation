#!/usr/bin/env python3
"""
Test script to check what columns are available in the model results
"""

import numpy as np
import pysd

def test_model_columns():
    """Test what columns are available in the model results"""
    
    print("🔍 TESTING MODEL COLUMNS:")
    print("=" * 50)
    
    try:
        # Set up functionspace
        print("Setting up functionspace...")
        from fix_pysd_poisson import add_random_poisson_to_pysd
        add_random_poisson_to_pysd()
        print("✅ Functionspace set up")
        
        # Load model
        print("Loading model...")
        model = pysd.read_vensim("model_13.mdl")
        print("✅ Model loaded successfully")
        
        # Run model
        print("Running model...")
        np.random.seed(42)
        result = model.run()
        print("✅ Model run completed")
        
        # Check available columns
        print(f"\n📊 AVAILABLE COLUMNS ({len(result.columns)} total):")
        print("=" * 40)
        for i, col in enumerate(result.columns):
            print(f"{i+1:2d}. {col}")
        
        # Check for specific columns that scenario_comparison_plots looks for
        print(f"\n🔍 CHECKING FOR EXPECTED COLUMNS:")
        print("=" * 40)
        
        expected_columns = ['performance[A]', 'performance[B]', 'accidents']
        for col in expected_columns:
            if col in result.columns:
                print(f"✅ Found: {col}")
            else:
                print(f"❌ Missing: {col}")
        
        # Show some sample data for accidents if it exists
        if 'accidents' in result.columns:
            print(f"\n📈 ACCIDENTS SAMPLE DATA:")
            print("=" * 40)
            accidents = result['accidents']
            print(f"First 10 values: {accidents.head(10).tolist()}")
            print(f"Unique values: {sorted(accidents.unique())}")
            print(f"Non-zero count: {np.sum(accidents != 0)}/{len(accidents)}")
        
        # Show performance columns if they exist
        performance_cols = [col for col in result.columns if 'performance' in col.lower()]
        if performance_cols:
            print(f"\n📈 PERFORMANCE COLUMNS FOUND:")
            print("=" * 40)
            for col in performance_cols:
                print(f"  - {col}")
        
        return result.columns.tolist()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    columns = test_model_columns()
    
    if columns:
        print(f"\n✅ Successfully retrieved {len(columns)} columns")
    else:
        print(f"\n❌ Failed to retrieve columns")
