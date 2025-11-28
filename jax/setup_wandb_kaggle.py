#!/usr/bin/env python
"""
Setup script for wandb on Kaggle.
Run this before training to set up wandb API key.

Usage:
    from kaggle_secrets import UserSecretsClient
    import os
    user_secrets = UserSecretsClient()
    os.environ['WANDB_API_KEY'] = user_secrets.get_secret("wandb_api_key")
    
    # Then run your training script - it will use the env var automatically
"""

import os

def setup_wandb_from_kaggle_secrets():
    """Set up wandb API key from Kaggle secrets."""
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        api_key = user_secrets.get_secret("wandb_api_key")
        
        if api_key:
            os.environ['WANDB_API_KEY'] = api_key
            print("✓ Wandb API key set from Kaggle secrets")
            return True
        else:
            print("✗ Wandb API key not found in Kaggle secrets")
            return False
    except ImportError:
        print("✗ kaggle_secrets not available (not running on Kaggle?)")
        return False
    except Exception as e:
        print(f"✗ Error getting wandb API key: {e}")
        return False

if __name__ == "__main__":
    setup_wandb_from_kaggle_secrets()

