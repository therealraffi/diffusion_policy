#!/usr/bin/env python
"""
Environment setup for diffusion_policy training.
Handles gymnasium -> gym compatibility for older code.
"""

import sys

# Try to use gym if available, otherwise alias gymnasium
try:
    import gym
    # gym is available, use it directly
except ImportError:
    try:
        import gymnasium
        sys.modules['gym'] = gymnasium
    except ImportError:
        pass
