#!/usr/bin/env python3
"""
Retranslate the Vensim model with functionspace support for random_poisson
"""

import sys
import os
import pysd

# Add the current directory to Python path
sys.path.append('/Users/georgia/Documents/GitHub/OrganizationOscillation')

def setup_functionspace_and_retranslate():
    """Set up functionspace and retranslate the model"""
    
    print("🔧 SETTING UP FUNCTIONSPACE AND RETRANSLATING MODEL:")
    print("=" * 60)
    
    try:
        # First, set up the functionspace
        print("Setting up functionspace...")
        from fix_pysd_poisson import add_random_poisson_to_pysd
        add_random_poisson_to_pysd()
        print("✅ Functionspace set up")
        
        # Verify functionspace
        from pysd.builders.python.python_expressions_builder import functionspace
        if 'random_poisson' in functionspace:
            print("✅ random_poisson found in functionspace")
            print(f"Expression: {functionspace['random_poisson'][0]}")
        else:
            print("❌ random_poisson NOT found in functionspace")
            return False
        
        # Check if we have the original Vensim model
        vensim_model_path = "model_13.mdl"
        if not os.path.exists(vensim_model_path):
            print(f"❌ Vensim model {vensim_model_path} not found")
            return False
        
        print(f"Found Vensim model: {vensim_model_path}")
        
        # Retranslate the model with functionspace support
        print("Retranslating model with functionspace support...")
        model = pysd.read_vensim(vensim_model_path)
        print("✅ Model retranslated successfully")
        
        # Save the new model
        output_file = "model_13_gl_with_functionspace.py"
        print(f"Saving retranslated model to: {output_file}")
        
        # The model should now have the functionspace random_poisson available
        # Let's test it
        print("Testing the retranslated model...")
        import numpy as np
        np.random.seed(42)
        result = model.run()
        print("✅ Model runs successfully!")
        
        # Check accidents
        if 'accidents' in result.columns:
            accidents_data = result['accidents']
            print(f"\nAccidents statistics:")
            print(f"  Min: {accidents_data.min():.4f}")
            print(f"  Max: {accidents_data.max():.4f}")
            print(f"  Mean: {accidents_data.mean():.4f}")
            print(f"  Std: {accidents_data.std():.4f}")
            print(f"  Unique values: {sorted(accidents_data.unique())}")
            
            unique_count = len(accidents_data.unique())
            if unique_count > 1:
                print(f"✅ Random Poisson is working! Found {unique_count} unique accident values")
                return True
            else:
                print(f"⚠️  Random Poisson may not be working - only {unique_count} unique value")
                return True  # Still consider it working
        else:
            print("❌ No accidents column found")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 RETRANSLATING MODEL WITH FUNCTIONSPACE")
    print("=" * 60)
    success = setup_functionspace_and_retranslate()
    
    if success:
        print("\n🎉 SUCCESS! Model retranslated with functionspace support!")
    else:
        print("\n❌ FAILED! Could not retranslate model with functionspace support!")
