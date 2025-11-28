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

if not os.path.exists(models_path):
    print(f"✗ File not found!")
    sys.exit(1)

# Read the file first to check content
print("\nChecking file content...")
with open(models_path, 'r') as f:
    content = f.read()
    print(f"  File size: {len(content)} bytes")
    print(f"  Contains 'class JIPNetFullFlax': {'class JIPNetFullFlax' in content}")
    print(f"  Contains 'class ConvBlock': {'class ConvBlock' in content}")
    print(f"  Number of 'class ' occurrences: {content.count('class ')}")

# Try multiple import methods
print("\n" + "="*60)
print("Method 1: Using importlib...")
try:
    # Clear any cached version
    if 'jax_models' in sys.modules:
        del sys.modules['jax_models']
    
    spec = importlib.util.spec_from_file_location("jax_models", models_path)
    module = importlib.util.module_from_spec(spec)
    
    # Execute with error handling
    print("  Executing module...")
    spec.loader.exec_module(module)
    
    print(f"  ✓ Module loaded")
    attrs = [x for x in dir(module) if not x.startswith('_')]
    print(f"  Attributes: {attrs}")
    print(f"  Has JIPNetFullFlax: {hasattr(module, 'JIPNetFullFlax')}")
    
    if hasattr(module, 'JIPNetFullFlax'):
        cls = module.JIPNetFullFlax
        print(f"  ✓ JIPNetFullFlax found: {cls}")
        print(f"    MRO: {cls.__mro__}")
        print(f"    Module: {cls.__module__}")
    else:
        print("  ✗ JIPNetFullFlax NOT found in module")
        
except Exception as e:
    print(f"  ✗ Error:")
    traceback.print_exc()

# Method 2: Direct execution
print("\n" + "="*60)
print("Method 2: Direct execution...")
try:
    # Create a namespace
    namespace = {
        '__name__': '__main__',
        '__file__': models_path,
    }
    
    # Execute in namespace
    exec(compile(content, models_path, 'exec'), namespace)
    
    print(f"  ✓ Execution completed")
    classes = [k for k in namespace.keys() if 'JIPNet' in k or 'Flax' in k or 'Block' in k or 'Encoder' in k]
    print(f"  Classes in namespace: {classes}")
    
    if 'JIPNetFullFlax' in namespace:
        cls = namespace['JIPNetFullFlax']
        print(f"  ✓ JIPNetFullFlax found: {cls}")
    else:
        print("  ✗ JIPNetFullFlax NOT found in namespace")
        
except Exception as e:
    print(f"  ✗ Error during execution:")
    traceback.print_exc()

# Method 3: Try importing as a regular module
print("\n" + "="*60)
print("Method 3: Regular import (if jax is in path)...")
try:
    # Make sure jax directory is in path
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    
    # Clear cache
    if 'jax_models' in sys.modules:
        del sys.modules['jax_models']
    
    import jax_models
    print(f"  ✓ Imported as module")
    print(f"  Attributes: {[x for x in dir(jax_models) if not x.startswith('_')]}")
    print(f"  Has JIPNetFullFlax: {hasattr(jax_models, 'JIPNetFullFlax')}")
    
    if hasattr(jax_models, 'JIPNetFullFlax'):
        cls = jax_models.JIPNetFullFlax
        print(f"  ✓ JIPNetFullFlax found: {cls}")
        
except Exception as e:
    print(f"  ✗ Error:")
    traceback.print_exc()

print("\n" + "="*60)
print("Summary: If all methods fail, there may be a runtime error")
print("during class definition that's being silently caught.")
