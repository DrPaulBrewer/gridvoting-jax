import sys
import test_package

print("=== Import successful ===")
print(f"Module location: {test_package.__file__}")
print(f"Module version: {test_package.__version__}")
print(f"\nPython path entries:")
for i, path in enumerate(sys.path[:10]):
    print(f"  {i}: {path}")
