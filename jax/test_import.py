#!/usr/bin/env python
"""Test script to verify imports work correctly"""
import os
import sys
import importlib.util
import traceback

# Add current directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

print(f"Script directory: {script_dir}")
print(f"Python path: {sys.path[:3]}")

# First check if JAX/Flax are available
print("\n" + "="*60)
print("Checking JAX/Flax dependencies...")
try:
    import jax
    import jax.numpy as jnp
    from flax import linen as nn
    print("✓ JAX and Flax are available")
    print(f"  JAX version: {jax.__version__ if hasattr(jax, '__version__') else 'unknown'}")
except ImportError as e:
    print(f"✗ JAX/Flax not available: {e}")
    print("  This will cause the module to fail to define classes")
    sys.exit(1)

# Test jax_models import
print("\n" + "="*60)
models_path = os.path.join(script_dir, "jax_models.py")
print(f"Loading jax_models from: {models_path}")
print(f"File exists: {os.path.exists(models_path)}")

if os.path.exists(models_path):
    try:
        # Clear any cached version
        if 'jax_models' in sys.modules:
            del sys.modules['jax_models']
        
        spec = importlib.util.spec_from_file_location("jax_models", models_path)
        module = importlib.util.module_from_spec(spec)
        
        # Execute with error handling
        print("\nExecuting module...")
        spec.loader.exec_module(module)
        
        print(f"✓ Module loaded successfully")
        print(f"\nModule attributes: {[x for x in dir(module) if not x.startswith('_')]}")
        print(f"Has JIPNetFullFlax: {hasattr(module, 'JIPNetFullFlax')}")
        
        if hasattr(module, 'JIPNetFullFlax'):
            print("\n✓ JIPNetFullFlax found!")
            cls = module.JIPNetFullFlax
            print(f"  Class: {cls}")
            print(f"  MRO: {cls.__mro__}")
            print(f"  Module: {cls.__module__}")
        else:
            print("\n✗ JIPNetFullFlax NOT found!")
            print("  This suggests a runtime error during class definition")
            print("\n  Trying to identify the issue...")
            
            # Try to execute the file directly to see errors
            print("\n  Executing file directly to catch errors...")
            try:
                with open(models_path, 'r') as f:
                    code = f.read()
                exec(code, {'__name__': '__main__', '__file__': models_path})
                print("  Direct execution completed")
                print(f"  Classes in namespace: {[k for k in globals().keys() if 'JIPNet' in k or 'Flax' in k]}")
            except Exception as e:
                print(f"  ✗ Error during direct execution:")
                traceback.print_exc()
                
    except Exception as e:
        print(f"✗ Error loading module:")
        traceback.print_exc()
else:
    print(f"✗ File not found!")
