#!/usr/bin/env python3
"""
Script to download OSF benchmark data.
Usage: python3 download_osf_data.py
Reference: gridvoting_jax.benchmarks.osf_comparison.ensure_osf_data_cached
"""
import sys
import os

# Add src to path to import gridvoting_jax
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

try:
    from gridvoting_jax.benchmarks.osf_comparison import ensure_osf_data_cached
    print(f"Downloading OSF data to: {os.environ.get('GV_OSF_CACHE_DIR', 'default')}")
    cache_path = ensure_osf_data_cached()
    print(f"Successfully downloaded to: {cache_path}")
except Exception as e:
    print(f"Error downloading OSF data: {e}")
    sys.exit(1)
