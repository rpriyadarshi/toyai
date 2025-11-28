# Hand Calculation Worksheet: Example 1

## Forward Pass Only - No Training

### Initial Values

**Token Embeddings:**
- A = [1, 0]
- B = [0, 1]
- C = [1, 1]
- D = [0, 0]

**Projection Weights:**
```
WQ = [0.1, 0.0]
     [0.0, 0.1]

WK = [0.1, 0.0]
     [0.0, 0.1]

WV = [0.1, 0.0]
     [0.0, 0.1]
```

**Output Projection:**
```
WO = [0.1, 0.0]
     [0.0, 0.1]
```

### Step 1: Token Embeddings

Input sequence: [A, B]

```
X = [1, 0]   ← Token A
    [0, 1]   ← Token B
```

### Step 2: Q, K, V Projections

**Q = X × WQ:**

```
Q[0][0] = 1×0.1 + 0×0.0 = 0.1
Q[0][1] = 1×0.0 + 0×0.1 = 0.0
Q[1][0] = 0×0.1 + 1×0.0 = 0.0
Q[1][1] = 0×0.0 + 1×0.1 = 0.1

Q = [0.1, 0.0]
    [0.0, 0.1]
```

**K = X × WK:** (same as Q, since WK = WQ)

```
K = [0.1, 0.0]
    [0.0, 0.1]
```

**V = X × WV:** (same as Q)

```
V = [0.1, 0.0]
    [0.0, 0.1]
```

### Step 3: Attention Scores

For position 1 (B), compute attention to positions 0 and 1:

**K^T = K^T:**

```
K^T = [0.1, 0.0]
      [0.0, 0.1]
```

**Raw Scores = Q × K^T:**

```
For position 1:
score[1][0] = Q[1] · K[0] = [0.0, 0.1] · [0.1, 0.0] = 0.0×0.1 + 0.1×0.0 = 0.0
score[1][1] = Q[1] · K[1] = [0.0, 0.1] · [0.0, 0.1] = 0.0×0.0 + 0.1×0.1 = 0.01
```

**Scale by 1/√2 ≈ 0.707:**

```
scaled_score[1][0] = 0.0 × 0.707 = 0.0
scaled_score[1][1] = 0.01 × 0.707 = 0.00707
```

### Step 4: Attention Weights (Softmax)

For position 1, scores = [0.0, 0.00707]

**Softmax:**
1. Find max: max(0.0, 0.00707) = 0.00707
2. Subtract max: [0.0 - 0.00707, 0.00707 - 0.00707] = [-0.00707, 0.0]
3. Exponentiate: [exp(-0.00707), exp(0.0)] ≈ [0.9929, 1.0]
4. Sum: 0.9929 + 1.0 = 1.9929
5. Normalize: [0.9929/1.9929, 1.0/1.9929] ≈ [0.498, 0.502]

**Attention weights for position 1:**
- Weight on position 0 (A): ≈ 0.498
- Weight on position 1 (B): ≈ 0.502

### Step 5: Context Vector

**Context = weighted sum of values:**

```
context[0] = 0.498 × V[0][0] + 0.502 × V[1][0]
           = 0.498 × 0.1 + 0.502 × 0.0
           = 0.0498

context[1] = 0.498 × V[0][1] + 0.502 × V[1][1]
           = 0.498 × 0.0 + 0.502 × 0.1
           = 0.0502

Context = [0.0498, 0.0502]
```

### Step 6: Output Logits

**Logits = Context × WO (adapted for 4 tokens):**

For simplicity, we'll compute:
- logit(A) = context[0] × 0.1 + context[1] × 0.0 = 0.0498 × 0.1 = 0.00498
- logit(B) = context[0] × 0.0 + context[1] × 0.0 = 0.0
- logit(C) = context[0] × 0.0 + context[1] × 0.1 = 0.0502 × 0.1 = 0.00502
- logit(D) = context[0] × 0.0 + context[1] × 0.0 = 0.0

### Step 7: Probabilities (Softmax)

**Logits = [0.00498, 0.0, 0.00502, 0.0]**

1. Max: 0.00502
2. Subtract: [-0.00004, -0.00502, 0.0, -0.00502]
3. Exponentiate: [exp(-0.00004), exp(-0.00502), exp(0.0), exp(-0.00502)]
                ≈ [0.99996, 0.995, 1.0, 0.995]
4. Sum: ≈ 3.99
5. Normalize: [0.250, 0.249, 0.251, 0.249]

**Final Probabilities:**
- P(A) ≈ 0.250 (25.0%)
- P(B) ≈ 0.249 (24.9%)
- P(C) ≈ 0.251 (25.1%)
- P(D) ≈ 0.249 (24.9%)

### Verification

- ✓ Probabilities sum to 1.0 (approximately)
- ✓ All probabilities are positive
- ✓ Roughly equal probabilities (model hasn't learned yet)

### Common Mistakes

1. **Forgetting scaling factor**: Always divide by √d_k
2. **Softmax numerical issues**: Always subtract max before exponentiating
3. **Dimension mismatches**: Check matrix dimensions at each step
4. **Wrong attention position**: Make sure you're computing attention for the correct position

### Next Steps

Compare your hand calculation to the code output. They should match (within rounding error).

Then proceed to Example 2 to see how training changes these probabilities!

