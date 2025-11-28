# Example 5: Feed-Forward Layers

## Goal

Add non-linearity and depth with feed-forward networks.

## What You'll Learn

- Feed-forward network architecture
- ReLU activation function
- Residual connections
- Layer composition
- Why non-linearity is needed

## Model Architecture

```
Input → Attention → Feed-Forward → Output
                ↓
         Residual connection
```

## Feed-Forward Network

Two linear transformations with ReLU:

$$\text{FFN}(x) = \text{ReLU}(x W_1 + b_1) W_2 + b_2$$

For 2x2 case:
- $W_1$: 2×2 matrix
- $W_2$: 2×2 matrix
- ReLU: element-wise $\max(0, x)$

## Key Concepts

### ReLU Activation

$$\text{ReLU}(x) = \max(0, x)$$

Properties:
- Non-linear (enables complex functions)
- Simple derivative
- Prevents negative activations

### Residual Connections

$$\text{Output} = x + \text{FFN}(x)$$

Benefits:
- Enables gradient flow
- Allows identity mapping
- Helps training stability

### Universal Approximation

Feed-forward networks with non-linear activations can approximate any continuous function.

## Running

```bash
cd build
make example5
./examples/example5_feedforward/example5
```

## Expected Output

- Attention output
- FFN hidden layer (before ReLU)
- FFN activated (after ReLU)
- FFN output
- Final output with residual

## Next Steps

- **Example 6**: Complete transformer with all components

