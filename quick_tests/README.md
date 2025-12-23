# Quick Tests

This directory contains standalone test scripts for development, debugging, and validation.
These are NOT part of the official pytest suite but provide useful diagnostic and verification capabilities.

**Note**: Each test file contains detailed documentation in its header comment. See individual files for complete information.

## Security Note

This directory is tracked in git. Do NOT add files that contain:
- SSH commands or credentials
- API keys or tokens
- Sensitive data or secrets

## Usage

```bash
cd quick_tests
python test_name.py
```

## Test Index

### Configuration & Diagnostics
- `test_auto_config_verify.py` - Device configuration verification
- `test_cpu_config.py` - CPU-only configuration
- `test_cpu_config_forced.py` - Force CPU mode
- `test_cpu_config_user.py` - User-specified CPU config
- `test_cpu_diagnostics.py` - CPU diagnostics and metrics
- `test_cpu_utilization.py` - CPU utilization monitoring

### Large Grid Tests (g=100)
- `test_g100_cpu.py` - g=100 validation on CPU
- `test_g100_solver_comparison.py` - Solver comparison at g=100

### GPU Tests
- `test_gmres_gpu_float32.py` - GMRES on GPU (float32)
- `test_gmres_l4_gpu.py` - L4 GPU specific tests
- `test_power_method_float64_gpu.py` - Power method (float64) on GPU

### Grid Upscaling Tests
- `test_grid_upscaling_large.py` - Large grid upscaling (g=80, g=100)

### OSF Validation Tests
- `test_osf_validation.py` - General OSF validation
- `test_osf_g80_g100_cpu.py` - OSF validation for g=80, g=100

### Precision & Solver Tests
- `test_precision.py` - Float32 vs float64 precision
- `test_power_method_diagnostic.py` - Power method diagnostics
- `test_solver_comparison_cpu.py` - CPU solver comparison

## Adding New Quick Tests

When adding new quick tests:

1. **Naming**: Use `test_<feature>_<variant>.py` format
2. **Header comment**: Add comprehensive documentation (see existing files)
3. **Update README**: Add entry to appropriate section in this index
4. **Security**: Ensure no secrets, credentials, or sensitive paths
