#!/usr/bin/env python3
"""Quick test to show dashboard command output"""
import sys
sys.path.insert(0, '/home/joker/LlamaForge')

# Import the function
from llamaforge_interactive import launch_dashboard

print("\n" + "="*80)
print("  SIMULATING: User types 'dashboard' in interactive CLI")
print("="*80)

# This is what executes when user types "dashboard"
launch_dashboard()
