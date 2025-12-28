
import sys
import gridvoting_jax as gv

print(f"JAX version: {gv.__version__}")
try:
    print("Attempting to access gv.benchmarks...")
    print(dir(gv))
    print(f"benchmarks found: {hasattr(gv, 'benchmarks')}")
    
    if hasattr(gv, 'benchmarks'):
        print("Attempting to call performance()...")
        gv.benchmarks.performance(dict=True)
        print("Success!")
    else:
        print("FAIL: benchmarks not found in gv")

except Exception as e:
    print(f"FAIL: {e}")
    import traceback
    traceback.print_exc()
