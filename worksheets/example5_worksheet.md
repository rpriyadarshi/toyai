# Hand Calculation Worksheet: Example 5

## Feed-Forward Layers - Adding Non-Linearity

### Initial Values

**Token Embeddings:**
- A = [1, 0]
- B = [0, 1]

**Attention Weights:**
```
WQ = [0.1, 0.0]
     [0.0, 0.1]

WK = [0.1, 0.0]
     [0.0, 0.1]

WV = [0.1, 0.0]
     [0.0, 0.1]
```

**Feed-Forward Weights:**
```
W1 = [0.2, 0.1]
     [0.1, 0.2]

W2 = [0.2, 0.1]
     [0.1, 0.2]
```

**Input:** [A, B]

---

## Part 1: Attention (Same as Before)

### Forward Pass Through Attention

**Embeddings:**
```
X = [1, 0]
    [0, 1]
```

**Q, K, V:**
```
Q = [0.1, 0.0]
    [0.0, 0.1]

K = [0.1, 0.0]
    [0.0, 0.1]

V = [0.1, 0.0]
    [0.0, 0.1]
```

**Attention scores (position 1):**
- score[1][0] = 0.0
- score[1][1] = 0.01
- Scaled: [0.0, 0.00707]

**Attention weights:**
- p₀ ≈ 0.498
- p₁ ≈ 0.502

**Attention output:**
```
attn_output = [0.0498, 0.0502]  (for position 1)
```

**Full attention output matrix:**
```
attn_output = [0.0498, 0.0502]
              [0.0498, 0.0502]  (simplified, both rows similar)
```

---

## Part 2: Feed-Forward Network

### Step 1: First Linear Layer

**FFN_hidden = attn_output × W1**

```
attn_output = [0.0498, 0.0502]
              [0.0498, 0.0502]

W1 = [0.2, 0.1]
     [0.1, 0.2]

FFN_hidden[0][0] = 0.0498 × 0.2 + 0.0502 × 0.1 = 0.00996 + 0.00502 = 0.01498
FFN_hidden[0][1] = 0.0498 × 0.1 + 0.0502 × 0.2 = 0.00498 + 0.01004 = 0.01502

FFN_hidden[1][0] = 0.0498 × 0.2 + 0.0502 × 0.1 = 0.01498
FFN_hidden[1][1] = 0.0498 × 0.1 + 0.0502 × 0.2 = 0.01502

FFN_hidden = [0.01498, 0.01502]
             [0.01498, 0.01502]
```

### Step 2: ReLU Activation

**ReLU(x) = max(0, x)**

```
FFN_activated[0][0] = max(0, 0.01498) = 0.01498
FFN_activated[0][1] = max(0, 0.01502) = 0.01502
FFN_activated[1][0] = max(0, 0.01498) = 0.01498
FFN_activated[1][1] = max(0, 0.01502) = 0.01502

FFN_activated = [0.01498, 0.01502]
                [0.01498, 0.01502]
```

**Note:** All values are positive, so ReLU doesn't change them.

**If we had negative values:**
- Example: max(0, -0.5) = 0.0 (would zero out negative)

### Step 3: Second Linear Layer

**FFN_output = FFN_activated × W2**

```
FFN_activated = [0.01498, 0.01502]
                [0.01498, 0.01502]

W2 = [0.2, 0.1]
     [0.1, 0.2]

FFN_output[0][0] = 0.01498 × 0.2 + 0.01502 × 0.1 = 0.002996 + 0.001502 = 0.004498
FFN_output[0][1] = 0.01498 × 0.1 + 0.01502 × 0.2 = 0.001498 + 0.003004 = 0.004502

FFN_output[1][0] = 0.01498 × 0.2 + 0.01502 × 0.1 = 0.004498
FFN_output[1][1] = 0.01498 × 0.1 + 0.01502 × 0.2 = 0.004502

FFN_output = [0.004498, 0.004502]
             [0.004498, 0.004502]
```

---

## Part 3: Residual Connection

**Output = attn_output + FFN_output**

```
attn_output = [0.0498, 0.0502]
              [0.0498, 0.0502]

FFN_output = [0.004498, 0.004502]
             [0.004498, 0.004502]

Final_output[0][0] = 0.0498 + 0.004498 = 0.054298
Final_output[0][1] = 0.0502 + 0.004502 = 0.054702

Final_output[1][0] = 0.0498 + 0.004498 = 0.054298
Final_output[1][1] = 0.0502 + 0.004502 = 0.054702

Final_output = [0.054298, 0.054702]
               [0.054298, 0.054702]
```

---

## Part 4: Comparison

### Without Feed-Forward

**Output = [0.0498, 0.0502]**

### With Feed-Forward + Residual

**Output = [0.054298, 0.054702]**

**Change:**
- Increased by ~0.0045 in each dimension
- Non-linearity from ReLU adds capacity
- Residual preserves original information

---

## Part 5: Why This Matters

### Non-Linearity

**Without ReLU:**
- Two linear layers = one linear layer
- No additional capacity

**With ReLU:**
- Enables non-linear transformations
- Can learn complex functions
- Universal approximation property

### Residual Connection

**Benefits:**
1. **Gradient Flow**: Direct path for gradients
2. **Identity Mapping**: If FFN learns nothing, output = input
3. **Training Stability**: Easier to train deep networks

**Mathematical:**
- Without residual: output = FFN(input)
- With residual: output = input + FFN(input)
- If FFN(x) ≈ 0, then output ≈ input (identity)

---

## Verification

### Check 1: Dimensions Match

- ✓ attn_output: 2×2
- ✓ FFN_hidden: 2×2
- ✓ FFN_activated: 2×2
- ✓ FFN_output: 2×2
- ✓ Final_output: 2×2

### Check 2: ReLU Applied

- ✓ All positive values pass through
- ✓ Negative values would be zeroed

### Check 3: Residual Adds Value

- ✓ Final output > attention output
- ✓ Information from FFN is added

---

## Key Insights

1. **FFN Adds Capacity**: Non-linearity enables complex functions
2. **ReLU is Simple**: max(0, x) is easy to compute
3. **Residuals Help**: Enable deep networks
4. **Composition**: Attention + FFN = more powerful model

---

## Common Mistakes

1. **Forgetting ReLU**: Must apply non-linearity
2. **Wrong residual**: Add, don't replace
3. **Dimension mismatch**: Check matrix sizes

---

## Next Steps

- Try different FFN sizes
- Experiment with other activations
- See Example 6 for complete transformer

