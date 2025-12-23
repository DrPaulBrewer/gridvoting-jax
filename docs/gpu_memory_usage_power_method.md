# GPU Memory Usage - Power Method Test

## Test: g=100 Lazy Power Method (Float32)

**GPU**: NVIDIA GeForce GTX 1080 Ti (11264 MiB VRAM)

### Memory Usage Snapshot (During Test)
- **Test process memory**: 8530 MiB (~75% of total VRAM)
- **GPU utilization**: 58%
- **Temperature**: 67°C
- **Power draw**: 122W / 250W

### Observations

1. **High memory usage**: Power method uses ~8.5GB for g=100 (40,401 points)
   - This is close to the 11GB limit of GTX 1080 Ti
   - Float64 would require ~17GB → would not fit on this GPU

2. **Moderate GPU utilization**: 58% suggests:
   - Memory bandwidth bound (not compute bound)
   - Batched operations are efficient but limited by memory transfers
   - Some idle time between iterations

3. **Comparison with GMRES**:
   - GMRES with grid upscaling likely uses less memory (concentrated initial guess)
   - Power method needs full uniform distribution → more memory

### Implications

**For GTX 1080 Ti (11GB)**:
- ✅ g=100 float32: Fits (~8.5GB used)
- ❌ g=100 float64: Would not fit (~17GB needed)
- ❌ g=120+ float32: Likely would not fit

**For larger grids or float64**:
- Need GPU with more VRAM (A100 40GB, H100 80GB)
- Or use CPU with large RAM (64-96GB)

### Test Status
- Test running successfully on GPU
- No memory errors
- Converging normally
