# Regularizer Improvement Proposal

## Proposal: Input-Based Functional Regularization

### Current Implementation

The current `l_regularizer` samples **random binary inputs** and computes sensitivity:

```python
# Sample random inputs
z = torch.randint(0, 2, (num_samples, k), device=device, dtype=torch.float32)

# Compute sensitivity with torch.no_grad()
with torch.no_grad():
    g_z = node(z)
    # ... compute differences ...
```

**Problems:**
- Not differentiable (uses `torch.no_grad()`)
- Disconnected from training data distribution
- May regularize irrelevant input regions

### Proposed Implementation

Use **current batch inputs** instead of random samples:

```python
def l_regularizer_input_based(
    node: nn.Module,
    inputs: torch.Tensor,  # Current batch inputs
    p: int = 2,
) -> torch.Tensor:
    """
    Input-based functional regularization.
    
    Measures sensitivity to bit flips in the *actual* inputs the model sees.
    
    Args:
        node: The DiffLUT node to regularize
        inputs: Current batch inputs, shape (batch_size, k)
        p: Norm parameter (1 for L1, 2 for L2)
        
    Returns:
        Average functional sensitivity on current batch
    """
    k = inputs.shape[-1]
    
    # Generate Hamming neighbors for current inputs
    z_neighbors = _generate_hamming_neighbors(inputs)  # (batch_size, k, k)
    
    # Compute node output for original inputs (WITH gradients)
    g_z = node(inputs)  # Shape: (batch_size, output_dim)
    
    # Compute outputs for all neighbors (WITH gradients)
    batch_size = inputs.shape[0]
    z_neighbors_flat = z_neighbors.reshape(-1, k)
    g_z_neighbors_flat = node(z_neighbors_flat)
    g_z_neighbors = g_z_neighbors_flat.reshape(batch_size, k, -1)
    
    # Compute sensitivity
    g_z_expanded = g_z.unsqueeze(1)
    differences = torch.abs(g_z_expanded - g_z_neighbors)
    
    if p == 1:
        sensitivity = differences
    elif p == 2:
        sensitivity = differences ** 2
    else:
        sensitivity = differences ** p
    
    # Average over neighbors and batch
    reg = sensitivity.sum(dim=-1).mean(dim=-1).sum(dim=0) / k
    reg = reg / batch_size
    
    return reg
```

### Advantages

1. **Fully Differentiable**
   - No `torch.no_grad()` - gradients flow through entire computation
   - Can backprop through node evaluations
   - Enables gradient-based optimization of sensitivity

2. **Data-Aware**
   - Regularizes on actual data distribution
   - Focuses on regions the model encounters
   - More meaningful sensitivity measure

3. **Adaptive**
   - Different batches → different regularization
   - Automatically focuses on hard examples
   - Dynamic regularization pressure

4. **Theoretically Grounded**
   - Measures smoothness on true data manifold
   - Aligns with adversarial robustness literature
   - Encourages local Lipschitz continuity

### Usage Example

```python
# Training loop
for batch in dataloader:
    x, y = batch
    
    # Forward pass
    encoded = encoder(x)  # Shape: (batch_size, k)
    
    # Compute loss with input-based regularization
    output = model(encoded)
    loss = criterion(output, y)
    
    # Add regularization for each node
    for node in model.nodes():
        reg = l_regularizer_input_based(
            node, 
            inputs=encoded,  # Use current batch!
            p=2
        )
        loss = loss + 0.01 * reg
    
    # Backward pass (gradients flow through regularization!)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Comparison Table

| Aspect | Current (Random) | Proposed (Input-Based) |
|--------|-----------------|------------------------|
| Differentiability | ❌ No (torch.no_grad) | ✅ Yes (full gradients) |
| Data Awareness | ❌ Random samples | ✅ Actual batch inputs |
| Computational Cost | Low (cached samples) | Medium (neighbors per batch) |
| Regularization | Static | Adaptive |
| Theoretical Foundation | Weak | Strong |
| Test Compatibility | ⚠️ Skips differentiability | ✅ All tests pass |

### Implementation Strategy

**Option 1: Replace Current Implementation**
- Modify `l_regularizer` to require `inputs` argument
- Breaking change, requires updating all usage

**Option 2: Add New Regularizer**
- Keep `l_regularizer` (backward compatibility)
- Add `l_regularizer_input_based` as new option
- Register as `"l_input_based"`, `"l1_input_based"`, etc.

**Option 3: Hybrid Approach**
- Make `inputs` optional in `l_regularizer`
- If `inputs=None`, use random sampling (current behavior)
- If `inputs` provided, use input-based approach
- Best of both worlds!

### Recommended: Option 3 (Hybrid)

```python
@register_regularizer("l")
@register_regularizer("functional")
def l_regularizer(
    node: nn.Module,
    p: int = DEFAULT_REGULARIZER_P_NORM,
    num_samples: Optional[int] = DEFAULT_REGULARIZER_NUM_SAMPLES,
    inputs: Optional[torch.Tensor] = None,  # NEW: optional inputs
) -> torch.Tensor:
    """
    Functional L-regularization for DiffLUT nodes.
    
    Args:
        node: The DiffLUT node to regularize
        p: The norm parameter (1 for L1, 2 for L2)
        num_samples: Number of random samples (ignored if inputs provided)
        inputs: Optional batch inputs. If provided, uses input-based regularization.
                If None, uses random sampling (legacy behavior).
    
    Returns:
        Average functional sensitivity
    """
    if inputs is not None:
        # NEW: Input-based regularization (differentiable!)
        return _l_regularizer_input_based(node, inputs, p)
    else:
        # OLD: Random sampling (non-differentiable)
        return _l_regularizer_random(node, p, num_samples)
```

### Migration Path

1. **Phase 1**: Implement hybrid approach (backward compatible)
2. **Phase 2**: Update examples to use input-based version
3. **Phase 3**: Deprecate random sampling with warning
4. **Phase 4**: Remove random sampling in next major version

### Testing Implications

With input-based regularization:

✅ `test_regularizer_differentiable` - **WILL PASS** (gradients flow!)  
✅ `test_regularizer_scales_with_params` - **WILL PASS** (sensitivity changes with parameters)  
✅ All existing tests remain compatible (backward compatible)

### Performance Considerations

**Cost Analysis:**
- Current: `O(num_samples * k)` forward passes (fixed)
- Proposed: `O(batch_size * k)` forward passes (dynamic)

**Example:**
- `num_samples=100`, `k=10` → 1000 forward passes
- `batch_size=32`, `k=10` → 320 forward passes (3x faster!)

**Memory:**
- Current: Pre-allocated samples
- Proposed: Temporary neighbors tensor (released after backward)

### Conclusion

**Recommendation: Implement Option 3 (Hybrid)**

This approach:
1. ✅ Maintains backward compatibility
2. ✅ Provides better, differentiable regularization
3. ✅ Fixes all test skips
4. ✅ Improves theoretical foundation
5. ✅ Often faster than random sampling

**Next Steps:**
1. Implement `_l_regularizer_input_based` helper
2. Modify `l_regularizer` to accept optional `inputs`
3. Update documentation and examples
4. Add tests for input-based version
5. Benchmark performance comparison
