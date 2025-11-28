#!/usr/bin/env python
"""Test script to verify imports work correctly"""
import os
import sys
import importlib.util

# Add current directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

print(f"Script directory: {script_dir}")
print(f"Python path: {sys.path[:3]}")

# Test jax_models import
models_path = os.path.join(script_dir, "jax_models.py")
print(f"\nLoading jax_models from: {models_path}")
print(f"File exists: {os.path.exists(models_path)}")

if os.path.exists(models_path):
    try:
        spec = importlib.util.spec_from_file_location("jax_models", models_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        print(f"Module loaded successfully")
        print(f"Module attributes: {[x for x in dir(module) if not x.startswith('_')]}")
        print(f"Has JIPNetFullFlax: {hasattr(module, 'JIPNetFullFlax')}")
        
        if hasattr(module, 'JIPNetFullFlax'):
            print("✓ JIPNetFullFlax found!")
            cls = module.JIPNetFullFlax
            print(f"  Class: {cls}")
            print(f"  MRO: {cls.__mro__}")
        else:
            print("✗ JIPNetFullFlax NOT found!")
            print("  This suggests a runtime error during class definition")
    except Exception as e:
        print(f"✗ Error loading module:")
        import traceback
        traceback.print_exc()
else:
    print(f"✗ File not found!")

