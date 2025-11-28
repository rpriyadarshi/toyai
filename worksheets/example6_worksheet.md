# Hand Calculation Worksheet: Example 6

## Complete Transformer - Full Architecture

### Initial Values

**Token Embeddings:**
- A = [1, 0]
- B = [0, 1]

**Transformer Block 1 Weights:**
```
WQ1 = [0.1, 0.0]
      [0.0, 0.1]

WK1 = [0.1, 0.0]
      [0.0, 0.1]

WV1 = [0.1, 0.0]
      [0.0, 0.1]

W1_1 = [0.2, 0.1]  (FFN first layer)
       [0.1, 0.2]

W2_1 = [0.2, 0.1]  (FFN second layer)
       [0.1, 0.2]
```

**Output Projection:**
```
WO = [0.1, 0.0]
     [0.0, 0.1]
```

**Input:** [A, B]

---

## Part 1: Input Embeddings

```
X = [1, 0]   ← Token A
    [0, 1]   ← Token B
```

---

## Part 2: Transformer Block 1

### Step 1: Attention

**Q, K, V Projections:**
```
Q1 = X × WQ1 = [0.1, 0.0]
               [0.0, 0.1]

K1 = X × WK1 = [0.1, 0.0]
               [0.0, 0.1]

V1 = X × WV1 = [0.1, 0.0]
               [0.0, 0.1]
```

**Attention Computation:**
- Scores: [0.0, 0.00707] (for position 1)
- Weights: [0.498, 0.502]
- Attention output: [0.0498, 0.0502] (for position 1)

**Full attention output:**
```
attn1_output = [0.0498, 0.0502]
               [0.0498, 0.0502]
```

### Step 2: Layer Normalization + Residual

**Layer Normalization:**

For each row, compute:
- Mean: μ = (x₁ + x₂) / 2
- Variance: σ² = ((x₁ - μ)² + (x₂ - μ)²) / 2
- Normalized: (x - μ) / √(σ² + ε)

**For row 0:**
```
μ₀ = (0.0498 + 0.0502) / 2 = 0.05
σ²₀ = ((0.0498 - 0.05)² + (0.0502 - 0.05)²) / 2
    = (0.000004 + 0.000004) / 2
    = 0.000004
σ₀ = √0.000004 ≈ 0.002

norm1[0][0] = (0.0498 - 0.05) / 0.002 = -0.1
norm1[0][1] = (0.0502 - 0.05) / 0.002 = 0.1
```

**For row 1:** (similar calculation)

**Residual Connection:**
```
norm1_input = attn1_output + X

norm1_input[0][0] = 0.0498 + 1.0 = 1.0498
norm1_input[0][1] = 0.0502 + 0.0 = 0.0502

norm1_input[1][0] = 0.0498 + 0.0 = 0.0498
norm1_input[1][1] = 0.0502 + 1.0 = 1.0502
```

**After layer norm:**
```
norm1 = LayerNorm(norm1_input)
      ≈ [0.707, -0.707]  (normalized)
        [-0.707, 0.707]
```

### Step 3: Feed-Forward Network

**FFN Hidden Layer:**
```
ffn_hidden = norm1 × W1_1

norm1 = [0.707, -0.707]
        [-0.707, 0.707]

W1_1 = [0.2, 0.1]
       [0.1, 0.2]

ffn_hidden[0][0] = 0.707 × 0.2 + (-0.707) × 0.1 = 0.1414 - 0.0707 = 0.0707
ffn_hidden[0][1] = 0.707 × 0.1 + (-0.707) × 0.2 = 0.0707 - 0.1414 = -0.0707

ffn_hidden[1][0] = -0.707 × 0.2 + 0.707 × 0.1 = -0.1414 + 0.0707 = -0.0707
ffn_hidden[1][1] = -0.707 × 0.1 + 0.707 × 0.2 = -0.0707 + 0.1414 = 0.0707

ffn_hidden = [0.0707, -0.0707]
             [-0.0707, 0.0707]
```

**ReLU Activation:**
```
ffn_activated[0][0] = max(0, 0.0707) = 0.0707
ffn_activated[0][1] = max(0, -0.0707) = 0.0  ← Zeroed out!
ffn_activated[1][0] = max(0, -0.0707) = 0.0
ffn_activated[1][1] = max(0, 0.0707) = 0.0707

ffn_activated = [0.0707, 0.0]
                [0.0, 0.0707]
```

**FFN Output:**
```
ffn_output = ffn_activated × W2_1

ffn_activated = [0.0707, 0.0]
                [0.0, 0.0707]

W2_1 = [0.2, 0.1]
       [0.1, 0.2]

ffn_output[0][0] = 0.0707 × 0.2 + 0.0 × 0.1 = 0.01414
ffn_output[0][1] = 0.0707 × 0.1 + 0.0 × 0.2 = 0.00707

ffn_output[1][0] = 0.0 × 0.2 + 0.0707 × 0.1 = 0.00707
ffn_output[1][1] = 0.0 × 0.1 + 0.0707 × 0.2 = 0.01414

ffn_output = [0.01414, 0.00707]
             [0.00707, 0.01414]
```

### Step 4: Layer Normalization + Residual (Again)

**Residual:**
```
block1_input = ffn_output + norm1

block1_input = [0.01414, 0.00707] + [0.707, -0.707]
               [0.00707, 0.01414]   [-0.707, 0.707]

              = [0.72114, -0.69993]
                [-0.69993, 0.72114]
```

**Layer Normalization:**
```
block1_output = LayerNorm(block1_input)
               ≈ [0.707, -0.707]
                 [-0.707, 0.707]  (normalized)
```

---

## Part 3: Output Projection

**Context vector (from position 1):**
```
context = [block1_output[1][0], block1_output[1][1]]
         = [-0.707, 0.707]
```

**Logits:**
```
logits[0] = context[0] × WO[0][0] + context[1] × WO[1][0]
          = -0.707 × 0.1 + 0.707 × 0.0
          = -0.0707

logits[1] = context[0] × WO[0][1] + context[1] × WO[1][1]
          = -0.707 × 0.0 + 0.707 × 0.1
          = 0.0707

logits[2] = -0.0707  (similar to logit[0])
logits[3] = 0.0707   (similar to logit[1])
```

**Probabilities (Softmax):**
```
logits = [-0.0707, 0.0707, -0.0707, 0.0707]

1. Max: 0.0707
2. Subtract: [-0.1414, 0.0, -0.1414, 0.0]
3. Exponentiate: [0.868, 1.0, 0.868, 1.0]
4. Sum: 3.736
5. Normalize: [0.232, 0.268, 0.232, 0.268]

Probs = [0.232, 0.268, 0.232, 0.268]
```

---

## Part 4: Architecture Summary

### Components Used

1. ✓ **Embeddings**: Token → Vector
2. ✓ **Attention**: Q, K, V mechanism
3. ✓ **Layer Normalization**: Stabilizes training
4. ✓ **Residual Connections**: Enables gradient flow
5. ✓ **Feed-Forward**: Adds non-linearity
6. ✓ **Output Projection**: Context → Logits
7. ✓ **Softmax**: Logits → Probabilities

### Data Flow

```
Input → Embeddings
     → Attention
     → LayerNorm + Residual
     → Feed-Forward
     → LayerNorm + Residual
     → Output Projection
     → Softmax
     → Probabilities
```

---

## Verification

### Check 1: All Components Present

- ✓ Attention mechanism
- ✓ Layer normalization
- ✓ Residual connections
- ✓ Feed-forward network
- ✓ Complete pipeline

### Check 2: Dimensions Match

- ✓ All matrix operations compatible
- ✓ Residuals add correctly
- ✓ Output has correct shape

### Check 3: Non-Linearity Applied

- ✓ ReLU zeroes negative values
- ✓ Layer norm normalizes
- ✓ Multiple transformations applied

---

## Key Insights

1. **Complete Architecture**: All transformer components
2. **Layer Composition**: Each layer processes previous output
3. **Normalization**: Layer norm stabilizes training
4. **Residuals**: Enable deep networks
5. **End-to-End**: Complete forward pass

---

## Common Mistakes

1. **Forgetting layer norm**: Must normalize after each sub-layer
2. **Wrong residual order**: Norm after addition, not before
3. **Dimension errors**: Check all matrix sizes

---

## Congratulations!

You've computed a complete transformer forward pass by hand! 

The principles you've learned apply directly to:
- GPT models
- BERT models
- All transformer-based architectures

The math is identical - only the scale changes!

