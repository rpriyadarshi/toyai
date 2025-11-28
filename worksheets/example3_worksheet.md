# Hand Calculation Worksheet: Example 3

## Full Backpropagation - All Weights Trainable

### Initial Values

**Token Embeddings:**
- A = [1, 0]
- B = [0, 1]

**All Weights (Trainable):**
```
WQ = [0.1, 0.0]
     [0.0, 0.1]

WK = [0.1, 0.0]
     [0.0, 0.1]

WV = [0.1, 0.0]
     [0.0, 0.1]

WO = [0.1, 0.0]
     [0.0, 0.1]
```

**Training Example:**
- Input: [A, B]
- Target: C (index 2)
- Learning rate: η = 0.1

---

## Part 1: Forward Pass

### Step 1-5: Same as Example 2

**Context vector:** [0.0498, 0.0502]

**Logits:** [0.00498, 0.0, 0.00502, 0.0]

**Probabilities:** [0.250, 0.249, 0.251, 0.249]

**Loss:** L = -log(0.251) ≈ 1.383

---

## Part 2: Backward Pass

### Step 1: Gradient w.r.t. Logits

```
dL/dlogits = [0.250, 0.249, -0.749, 0.249]
```

### Step 2: Gradient w.r.t. Context

**Through output projection:**

For logit[i] = context · WO_column_i

dL/dcontext = sum over i: (dL/dlogit[i] × WO_column_i)

**Simplified for 2×2 case:**
```
dL/dcontext[0] = dL/dlogit[0] × WO[0][0] + dL/dlogit[2] × WO[0][0]
                = 0.250 × 0.1 + (-0.749) × 0.0
                = 0.025

dL/dcontext[1] = dL/dlogit[0] × WO[1][0] + dL/dlogit[2] × WO[1][1]
                = 0.250 × 0.0 + (-0.749) × 0.1
                = -0.0749
```

**dL/dcontext = [0.025, -0.0749]**

### Step 3: Gradient Through Attention Output

**dL/doutput (for position 1):**
```
dL/doutput[1][0] = dL/dcontext[0] = 0.025
dL/doutput[1][1] = dL/dcontext[1] = -0.0749
```

### Step 4: Gradient Through Attention Weights

**From output = weights × V:**

**dL/dweights = dL/doutput × V^T**

```
V^T = [0.1, 0.0]
      [0.0, 0.1]

dL/dweights[1][0] = dL/doutput[1][0] × V^T[0][0] + dL/doutput[1][1] × V^T[1][0]
                  = 0.025 × 0.1 + (-0.0749) × 0.0
                  = 0.0025

dL/dweights[1][1] = dL/doutput[1][0] × V^T[0][1] + dL/doutput[1][1] × V^T[1][1]
                  = 0.025 × 0.0 + (-0.0749) × 0.1
                  = -0.00749
```

**dL/dweights = [0.0, 0.0]  (for position 0)**
              **[0.0025, -0.00749]  (for position 1)**

### Step 5: Gradient Through Attention Scores

**Softmax backward:**

For softmax output s and gradient dL/ds:
dL/dx_i = s_i × (dL/ds_i - sum_j(dL/ds_j × s_j))

**For position 1:**
- s = [0.498, 0.502] (attention weights)
- dL/ds = [0.0025, -0.00749]

**Compute dot product:**
```
dot = dL/ds[0] × s[0] + dL/ds[1] × s[1]
    = 0.0025 × 0.498 + (-0.00749) × 0.502
    = 0.001245 - 0.00376
    = -0.002515
```

**Gradient w.r.t. scores:**
```
dL/dscores[1][0] = s[0] × (dL/ds[0] - dot)
                  = 0.498 × (0.0025 - (-0.002515))
                  = 0.498 × 0.005015
                  = 0.002497

dL/dscores[1][1] = s[1] × (dL/ds[1] - dot)
                  = 0.502 × (-0.00749 - (-0.002515))
                  = 0.502 × (-0.004975)
                  = -0.002497
```

**dL/dscores = [0.0, 0.0]  (for position 0)**
             **[0.002497, -0.002497]  (for position 1)**

### Step 6: Gradient Through Scaled Scores

**Unscale (multiply by scale factor):**

Scale factor = 1/√2 ≈ 0.707

```
dL/dscaled_scores = dL/dscores × scale_factor

dL/dscaled_scores[1][0] = 0.002497 × 0.707 ≈ 0.001765
dL/dscaled_scores[1][1] = -0.002497 × 0.707 ≈ -0.001765
```

### Step 7: Gradient Through Q × K^T

**From scores = Q × K^T:**

**dL/dQ = dL/dscores × K**

```
dL/dQ[1][0] = dL/dscores[1][0] × K[0][0] + dL/dscores[1][1] × K[1][0]
            = 0.001765 × 0.1 + (-0.001765) × 0.0
            = 0.0001765

dL/dQ[1][1] = dL/dscores[1][0] × K[0][1] + dL/dscores[1][1] × K[1][1]
            = 0.001765 × 0.0 + (-0.001765) × 0.1
            = -0.0001765
```

**dL/dK = dL/dscores^T × Q**

```
dL/dscores^T = [0.0,       0.001765]
               [0.0,      -0.001765]

dL/dK[0][0] = dL/dscores^T[0][0] × Q[0][0] + dL/dscores^T[0][1] × Q[1][0]
            = 0.0 × 0.1 + 0.001765 × 0.0
            = 0.0

dL/dK[0][1] = dL/dscores^T[0][0] × Q[0][1] + dL/dscores^T[0][1] × Q[1][1]
            = 0.0 × 0.0 + 0.001765 × 0.1
            = 0.0001765
```

(Similar for K[1][0] and K[1][1])

**dL/dV = weights^T × dL/doutput**

```
weights^T = [0.498, 0.502]
            [0.498, 0.502]  (simplified)

dL/dV[0][0] = weights^T[0][0] × dL/doutput[1][0]
            = 0.498 × 0.025
            = 0.01245

dL/dV[0][1] = weights^T[0][1] × dL/doutput[1][1]
            = 0.502 × (-0.0749)
            = -0.0376
```

### Step 8: Gradients Through Projections

**dL/dWQ = X^T × dL/dQ**

```
X^T = [1, 0]
      [0, 1]

dL/dWQ[0][0] = X^T[0][0] × dL/dQ[0][0] + X^T[0][1] × dL/dQ[1][0]
            = 1 × 0.0 + 0 × 0.0001765
            = 0.0

dL/dWQ[0][1] = X^T[0][0] × dL/dQ[0][1] + X^T[0][1] × dL/dQ[1][1]
            = 1 × 0.0 + 0 × (-0.0001765)
            = 0.0

dL/dWQ[1][0] = X^T[1][0] × dL/dQ[0][0] + X^T[1][1] × dL/dQ[1][0]
            = 0 × 0.0 + 1 × 0.0001765
            = 0.0001765

dL/dWQ[1][1] = X^T[1][0] × dL/dQ[0][1] + X^T[1][1] × dL/dQ[1][1]
            = 0 × 0.0 + 1 × (-0.0001765)
            = -0.0001765
```

**Similar calculations for WK and WV...**

---

## Part 3: Weight Updates

### Update WQ

```
WQ_new[0][0] = 0.1 - 0.1 × 0.0 = 0.1
WQ_new[0][1] = 0.0 - 0.1 × 0.0 = 0.0
WQ_new[1][0] = 0.0 - 0.1 × 0.0001765 = -0.00001765
WQ_new[1][1] = 0.1 - 0.1 × (-0.0001765) = 0.10001765
```

### Update WK, WV, WO

**Similar process for all weight matrices...**

---

## Verification

### Check 1: Gradient Flow

- ✓ Loss → Logits
- ✓ Logits → Context
- ✓ Context → Attention output
- ✓ Attention → Q, K, V
- ✓ Q, K, V → WQ, WK, WV

### Check 2: Matrix Dimensions

- ✓ All matrix multiplications have compatible dimensions
- ✓ Gradients have same shape as weights

### Check 3: Gradient Magnitudes

- ✓ Gradients are small (expected for one step)
- ✓ Signs make sense (negative for target, positive for others)

---

## Key Insights

1. **Chain Rule**: Each step multiplies local gradients
2. **Matrix Calculus**: Specific rules for matrix operations
3. **Softmax Complexity**: Jacobian couples all inputs
4. **Complete Flow**: Gradients flow through entire network

---

## Common Mistakes

1. **Wrong matrix dimensions**: Always check compatibility
2. **Transpose errors**: Remember when to transpose
3. **Softmax backward**: Complex due to normalization
4. **Sign errors**: Double-check gradient signs

---

## Next Steps

- Try multiple training steps
- See Example 4 for batch training
- Experiment with different architectures

