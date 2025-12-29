"""
Analyze the pattern for interpolation matrix creation.

For a fine grid with spacing (xstep, ystep) and coarse grid with spacing (2*xstep, 2*ystep),
both with the same boundaries, there's a mathematical relationship between indices.

Key insight: Grid indices are based on (row, col) in the 2D grid.
- Fine grid: g points in each dimension
- Coarse grid: (g+1)//2 points in each dimension (approximately g/2)

For a grid with x in [x0, x1] with step xstep:
- Number of x points: (x1 - x0) / xstep + 1
- For fine grid: n_x_fine = (x1 - x0) / xstep + 1
- For coarse grid: n_x_coarse = (x1 - x0) / (2*xstep) + 1 = (n_x_fine + 1) / 2

Grid index calculation (from Grid class):
- col = (x - x0) / xstep
- row = (y1 - y) / ystep  # Note: y is inverted
- index = row * n_cols + col

For fine grid point at (row_f, col_f):
- If row_f is even AND col_f is even: Maps to coarse grid at (row_f//2, col_f//2)
- If row_f is odd OR col_f is odd: Needs interpolation

Pattern for interpolation:
- (even, even) -> copy from (row_f//2, col_f//2)
- (odd, even) -> average of (row_f//2, col_f) and (row_f//2+1, col_f) [up-down neighbors]
- (even, odd) -> average of (row_f, col_f//2) and (row_f, col_f//2+1) [left-right neighbors]
- (odd, odd) -> average of 4 neighbors at (row_f//2, col_f//2), (row_f//2+1, col_f//2), 
                                          (row_f//2, col_f//2+1), (row_f//2+1, col_f//2+1)

This eliminates ALL coordinate lookups!
"""

import jax.numpy as jnp

def create_interpolation_matrix_pattern(n_rows_fine, n_cols_fine):
    """
    Create interpolation matrix using index pattern (no coordinate lookups).
    
    Args:
        n_rows_fine: Number of rows in fine grid
        n_cols_fine: Number of columns in fine grid
    
    Returns:
        rows, cols, data: Lists for sparse matrix construction
    """
    n_rows_coarse = (n_rows_fine + 1) // 2
    n_cols_coarse = (n_cols_fine + 1) // 2
    
    rows = []
    cols = []
    data = []
    
    for row_f in range(n_rows_fine):
        for col_f in range(n_cols_fine):
            # Fine grid index (1D)
            idx_f = row_f * n_cols_fine + col_f
            
            row_even = (row_f % 2 == 0)
            col_even = (col_f % 2 == 0)
            
            if row_even and col_even:
                # Direct copy from coarse grid
                row_c = row_f // 2
                col_c = col_f // 2
                idx_c = row_c * n_cols_coarse + col_c
                rows.append(idx_f)
                cols.append(idx_c)
                data.append(1.0)
                
            elif row_even and not col_even:
                # Left-right interpolation
                row_c = row_f // 2
                col_c_left = col_f // 2
                col_c_right = col_c_left + 1
                
                # Check bounds
                if col_c_right < n_cols_coarse:
                    # Average of left and right
                    idx_c_left = row_c * n_cols_coarse + col_c_left
                    idx_c_right = row_c * n_cols_coarse + col_c_right
                    rows.extend([idx_f, idx_f])
                    cols.extend([idx_c_left, idx_c_right])
                    data.extend([0.5, 0.5])
                else:
                    # Only left neighbor (boundary)
                    idx_c_left = row_c * n_cols_coarse + col_c_left
                    rows.append(idx_f)
                    cols.append(idx_c_left)
                    data.append(1.0)
                    
            elif not row_even and col_even:
                # Up-down interpolation
                col_c = col_f // 2
                row_c_up = row_f // 2
                row_c_down = row_c_up + 1
                
                # Check bounds
                if row_c_down < n_rows_coarse:
                    # Average of up and down
                    idx_c_up = row_c_up * n_cols_coarse + col_c
                    idx_c_down = row_c_down * n_cols_coarse + col_c
                    rows.extend([idx_f, idx_f])
                    cols.extend([idx_c_up, idx_c_down])
                    data.extend([0.5, 0.5])
                else:
                    # Only up neighbor (boundary)
                    idx_c_up = row_c_up * n_cols_coarse + col_c
                    rows.append(idx_f)
                    cols.append(idx_c_up)
                    data.append(1.0)
                    
            else:  # not row_even and not col_even
                # 4-neighbor interpolation
                row_c_up = row_f // 2
                row_c_down = row_c_up + 1
                col_c_left = col_f // 2
                col_c_right = col_c_left + 1
                
                # Collect valid neighbors
                neighbors = []
                if row_c_up < n_rows_coarse and col_c_left < n_cols_coarse:
                    neighbors.append(row_c_up * n_cols_coarse + col_c_left)
                if row_c_up < n_rows_coarse and col_c_right < n_cols_coarse:
                    neighbors.append(row_c_up * n_cols_coarse + col_c_right)
                if row_c_down < n_rows_coarse and col_c_left < n_cols_coarse:
                    neighbors.append(row_c_down * n_cols_coarse + col_c_left)
                if row_c_down < n_rows_coarse and col_c_right < n_cols_coarse:
                    neighbors.append(row_c_down * n_cols_coarse + col_c_right)
                
                # Average neighbors
                weight = 1.0 / len(neighbors)
                for idx_c in neighbors:
                    rows.append(idx_f)
                    cols.append(idx_c)
                    data.append(weight)
    
    return rows, cols, data

# Test the pattern
print("Testing pattern-based matrix creation")
print("="*80)

# Example: fine grid 5x5, coarse grid 3x3
n_rows_fine = 5
n_cols_fine = 5
n_rows_coarse = (n_rows_fine + 1) // 2  # = 3
n_cols_coarse = (n_cols_fine + 1) // 2  # = 3

print(f"Fine grid: {n_rows_fine}x{n_cols_fine} = {n_rows_fine * n_cols_fine} points")
print(f"Coarse grid: {n_rows_coarse}x{n_cols_coarse} = {n_rows_coarse * n_cols_coarse} points")

rows, cols, data = create_interpolation_matrix_pattern(n_rows_fine, n_cols_fine)

print(f"\nMatrix entries: {len(rows)}")
print(f"Expected sparsity: {100 * (1 - len(rows) / (n_rows_fine * n_cols_fine * n_rows_coarse * n_cols_coarse)):.1f}%")

# Show first few entries
print("\nFirst 10 entries (fine_idx -> coarse_idx: weight):")
for i in range(min(10, len(rows))):
    print(f"  {rows[i]} -> {cols[i]}: {data[i]}")

print("\n" + "="*80)
print("PATTERN FOUND!")
print("="*80)
print("Key insight: For 2x spacing grids with same boundaries:")
print("- Even row & even col: Direct copy from coarse[row//2, col//2]")
print("- Even row, odd col: Average of left-right neighbors")
print("- Odd row, even col: Average of up-down neighbors")
print("- Odd row & odd col: Average of 4 diagonal neighbors")
print("\nThis eliminates ALL coordinate lookups and Grid.index() calls!")
print("Matrix creation should be much faster with pure index arithmetic.")
