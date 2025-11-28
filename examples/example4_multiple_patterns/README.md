# Example 4: Multiple Patterns

## Goal

Learn multiple patterns from multiple training examples simultaneously.

## What You'll Learn

- Batch training
- Gradient accumulation
- Learning multiple patterns
- Training loop with epochs
- Convergence

## Training Examples

- [A, B] → C
- [A, A] → D
- [B, A] → C

## Key Concepts

### Batch Training

Process multiple examples together:
1. Forward pass for all examples
2. Compute losses
3. Average gradients
4. Update weights once per batch

### Gradient Averaging

$$\frac{\partial L}{\partial W} = \frac{1}{N} \sum_{i=1}^{N} \frac{\partial L_i}{\partial W}$$

### Training Loop

```
For each epoch:
    For each batch:
        Forward pass
        Compute losses
        Backward pass
        Average gradients
        Update weights
```

## Running

```bash
cd build
make example4
./examples/example4_multiple_patterns/example4
```

## Expected Output

- Training examples listed
- Loss decreasing over epochs
- Model learning all patterns

## Next Steps

- **Example 5**: Add feed-forward layers
- **Example 6**: Complete transformer architecture

