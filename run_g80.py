
import sys
import os

# Ensure we can import the package
sys.path.append('/workspace/src')

try:
    from gridvoting_jax.benchmarks.osf_comparison import run_comparison_report
except ImportError:
    # Try importing as module if package structure allows
    import gridvoting_jax
    from gridvoting_jax.benchmarks.osf_comparison import run_comparison_report

print("========================================")
print("Running g=80 Verification (Float64)")
print("========================================")

# Run only g=80 configs
configs = [(80, False), (80, True)]

run_comparison_report(configs=configs)
