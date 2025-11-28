# Example 3: Full Backpropagation

## Goal

Understand complete gradient flow through all components of the transformer.

## What You'll Learn

- Backpropagation through attention
- Matrix calculus rules
- Gradient flow through Q, K, V
- Gradients for all weight matrices
- Complete training loop

## Model Architecture

All weights are trainable:
- $W_Q$: Query projection
- $W_K$: Key projection
- $W_V$: Value projection
- $W_O$: Output projection

## Backpropagation Flow

1. Loss → Logits
2. Logits → Context
3. Context → Attention output
4. Attention → Q, K, V
5. Q, K, V → $W_Q$, $W_K$, $W_V$

## Key Concepts

### Matrix Calculus

For $C = AB$:
- $\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} B^T$
- $\frac{\partial L}{\partial B} = A^T \frac{\partial L}{\partial C}$

### Attention Gradients

- Through attention weights
- Through Q × K^T computation
- Through scaling factor

### Softmax Jacobian

Complex gradient due to normalization coupling all inputs.

## Running

```bash
cd build
make example3
./examples/example3_full_backprop/example3
```

## Expected Output

- Complete forward pass
- All gradient computations
- Updated weights for all matrices
- Demonstration of full gradient flow

## Next Steps

- **Example 4**: Multiple training examples
- **Example 5**: Feed-forward networks
- **Example 6**: Complete transformer

