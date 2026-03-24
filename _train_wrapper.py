"""
Compatibility wrapper to patch gymnasium as gym for diffusion_policy
"""
import sys
import os
import gymnasium as gym

# Change to correct working directory  
os.chdir('/bigtemp/rhm4nj/safe_diffusion/diffusion_policy')
sys.path.insert(0, '/bigtemp/rhm4nj/safe_diffusion/diffusion_policy')

# Inject gymnasium as gym in sys.modules
sys.modules['gym'] = gym

# Import and run training
import train
if __name__ == '__main__':
    train.main()
