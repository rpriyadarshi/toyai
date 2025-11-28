# Example 2: Single Training Step

## Goal

Understand how one weight update works. This example shows the complete training process for a single step.

## What You'll Learn

- How loss is computed
- How gradients are calculated
- How weights are updated
- How predictions improve after training

## Model Architecture

Same as Example 1, but with trainable output projection matrix $W_O$.

## Training Process

1. **Forward Pass**: Compute prediction (same as Example 1)
2. **Compute Loss**: Measure prediction error
3. **Compute Gradients**: Calculate how to change weights
4. **Update Weights**: Apply gradient descent
5. **Verify Improvement**: See prediction improve

## Key Concepts

### Loss Function

Cross-entropy loss: $L = -\log(P(\text{target}))$

- Lower when model is confident and correct
- Higher when model is wrong or uncertain

### Gradient Computation

For softmax + cross-entropy:
$$\frac{\partial L}{\partial \text{logit}_i} = P(i) - \mathbf{1}[i = \text{target}]$$

### Gradient Descent

$$W_{\text{new}} = W_{\text{old}} - \eta \cdot \frac{\partial L}{\partial W}$$

Where $\eta$ is the learning rate.

## Running

```bash
cd build
make example2
./examples/example2_single_step/example2
```

## Expected Output

- Initial probabilities (roughly equal)
- Loss value
- Gradients
- Updated weights
- Improved probabilities (target token should have higher probability)

## Next Steps

- **Example 3**: Full backpropagation through all weights
- **Example 4**: Training on multiple patterns
- **Example 5**: Add feed-forward layers
- **Example 6**: Complete transformer

