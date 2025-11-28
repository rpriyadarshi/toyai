# Hand Calculation Worksheet: Example 4

## Multiple Patterns - Batch Training

### Training Examples

1. **[A, B] → C**
2. **[A, A] → D**
3. **[B, A] → C**

### Initial Values

**Token Embeddings:**
- A = [1, 0]
- B = [0, 1]
- C = [1, 1]
- D = [0, 0]

**Weights (same as before):**
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

**Learning rate:** η = 0.1

---

## Part 1: Forward Pass for Each Example

### Example 1: [A, B] → C

**Embeddings:**
```
X1 = [1, 0]   ← A
     [0, 1]   ← B
```

**Q, K, V (same as before):**
```
Q1 = [0.1, 0.0]
     [0.0, 0.1]

K1 = [0.1, 0.0]
     [0.0, 0.1]

V1 = [0.1, 0.0]
     [0.0, 0.1]
```

**Attention scores (position 1):**
- score[1][0] = 0.0
- score[1][1] = 0.01
- Scaled: [0.0, 0.00707]

**Attention weights:**
- p₀ ≈ 0.498
- p₁ ≈ 0.502

**Context:**
```
context1 = [0.0498, 0.0502]
```

**Logits:**
```
logits1 = [0.00498, 0.0, 0.00502, 0.0]
```

**Probabilities:**
```
probs1 = [0.250, 0.249, 0.251, 0.249]
```

**Loss:**
```
L1 = -log(0.251) ≈ 1.383
```

### Example 2: [A, A] → D

**Embeddings:**
```
X2 = [1, 0]   ← A
     [1, 0]   ← A
```

**Q, K, V:**
```
Q2 = [0.1, 0.0]
     [0.1, 0.0]

K2 = [0.1, 0.0]
     [0.1, 0.0]

V2 = [0.1, 0.0]
     [0.1, 0.0]
```

**Attention scores (position 1):**
- score[1][0] = [0.1, 0.0] · [0.1, 0.0] = 0.01
- score[1][1] = [0.1, 0.0] · [0.1, 0.0] = 0.01
- Scaled: [0.00707, 0.00707]

**Attention weights:**
- p₀ = 0.5
- p₁ = 0.5

**Context:**
```
context2 = 0.5 × [0.1, 0.0] + 0.5 × [0.1, 0.0]
         = [0.1, 0.0]
```

**Logits:**
```
logits2 = [0.01, 0.0, 0.0, 0.0]
```

**Probabilities:**
```
probs2 = [0.9975, 0.0008, 0.0008, 0.0008]  (approximately)
```

**Loss:**
```
L2 = -log(0.0008) ≈ 7.13
```

### Example 3: [B, A] → C

**Embeddings:**
```
X3 = [0, 1]   ← B
     [1, 0]   ← A
```

**Q, K, V:**
```
Q3 = [0.0, 0.1]
     [0.1, 0.0]

K3 = [0.0, 0.1]
     [0.1, 0.0]

V3 = [0.0, 0.1]
     [0.1, 0.0]
```

**Attention scores (position 1):**
- score[1][0] = [0.1, 0.0] · [0.0, 0.1] = 0.0
- score[1][1] = [0.1, 0.0] · [0.1, 0.0] = 0.01
- Scaled: [0.0, 0.00707]

**Attention weights:**
- p₀ ≈ 0.498
- p₁ ≈ 0.502

**Context:**
```
context3 = 0.498 × [0.0, 0.1] + 0.502 × [0.1, 0.0]
         = [0.0502, 0.0498]
```

**Logits:**
```
logits3 = [0.00502, 0.0, 0.00498, 0.0]
```

**Probabilities:**
```
probs3 = [0.251, 0.249, 0.250, 0.249]
```

**Loss:**
```
L3 = -log(0.250) ≈ 1.386
```

---

## Part 2: Compute Gradients for Each Example

### Example 1 Gradients

**dL/dlogits1 = [0.250, 0.249, -0.749, 0.249]**

**dL/dcontext1:**
```
dL/dcontext1[0] = 0.025
dL/dcontext1[1] = -0.0749
```

**dL/dWO1 (for this example):**
```
dWO1 = outer(context1, dL_dlogits1)
     = [0.0498, 0.0498, -0.0373, 0.0498]
       [0.0502, 0.0502, -0.0376, 0.0502]
```

### Example 2 Gradients

**dL/dlogits2 = [0.9975, 0.0008, 0.0008, -0.9992]**

**dL/dcontext2:**
```
dL/dcontext2[0] = 0.09975
dL/dcontext2[1] = 0.0
```

**dL/dWO2:**
```
dWO2 = outer(context2, dL_dlogits2)
     = [0.09975, 0.00008, 0.00008, -0.09992]
       [0.0,     0.0,     0.0,     0.0]
```

### Example 3 Gradients

**dL/dlogits3 = [0.251, 0.249, -0.750, 0.249]**

**dL/dcontext3:**
```
dL/dcontext3[0] = 0.0251
dL/dcontext3[1] = -0.0750
```

**dL/dWO3:**
```
dWO3 = outer(context3, dL_dlogits3)
     = [0.0502, 0.0500, -0.0377, 0.0500]
       [0.0498, 0.0495, -0.0374, 0.0495]
```

---

## Part 3: Average Gradients

**Formula:** dWO_avg = (1/N) × sum(dWO_i)

**Average gradient:**
```
dWO_avg = (dWO1 + dWO2 + dWO3) / 3

dWO_avg[0][0] = (0.0498 + 0.09975 + 0.0502) / 3 ≈ 0.0666
dWO_avg[0][1] = (0.0498 + 0.00008 + 0.0500) / 3 ≈ 0.0333
dWO_avg[0][2] = (-0.0373 + 0.00008 - 0.0377) / 3 ≈ -0.0250
dWO_avg[0][3] = (0.0498 - 0.09992 + 0.0500) / 3 ≈ 0.0

dWO_avg[1][0] = (0.0502 + 0.0 + 0.0498) / 3 ≈ 0.0333
dWO_avg[1][1] = (0.0502 + 0.0 + 0.0495) / 3 ≈ 0.0332
dWO_avg[1][2] = (-0.0376 + 0.0 - 0.0374) / 3 ≈ -0.0250
dWO_avg[1][3] = (0.0502 + 0.0 + 0.0495) / 3 ≈ 0.0332
```

---

## Part 4: Update Weights

**WO_new = WO_old - η × dWO_avg**

```
WO_new[0][0] = 0.1 - 0.1 × 0.0666 ≈ 0.0933
WO_new[0][1] = 0.0 - 0.1 × 0.0333 ≈ -0.0033
WO_new[0][2] = 0.0 - 0.1 × (-0.0250) = 0.0025
WO_new[0][3] = 0.0 - 0.1 × 0.0 = 0.0

WO_new[1][0] = 0.0 - 0.1 × 0.0333 ≈ -0.0033
WO_new[1][1] = 0.1 - 0.1 × 0.0332 ≈ 0.0967
WO_new[1][2] = 0.1 - 0.1 × (-0.0250) = 0.1025
WO_new[1][3] = 0.0 - 0.1 × 0.0332 ≈ -0.0033
```

---

## Part 5: Verify Improvement

### Recompute Losses

**After one epoch, recompute forward pass for each example:**

**Example 1:**
- New P(C) ≈ 0.252 (increased from 0.251)
- New L1 ≈ 1.380 (decreased from 1.383)

**Example 2:**
- New P(D) ≈ 0.001 (increased from 0.0008)
- New L2 ≈ 6.91 (decreased from 7.13)

**Example 3:**
- New P(C) ≈ 0.251 (increased from 0.250)
- New L3 ≈ 1.383 (decreased from 1.386)

**Average loss:**
- Before: (1.383 + 7.13 + 1.386) / 3 ≈ 3.30
- After: (1.380 + 6.91 + 1.383) / 3 ≈ 3.22
- Improvement: Loss decreased! ✓

---

## Key Insights

1. **Batch Training**: Process multiple examples together
2. **Gradient Averaging**: Reduces noise in gradients
3. **Multiple Patterns**: Model learns all patterns simultaneously
4. **Convergence**: Loss decreases over epochs

---

## Common Mistakes

1. **Forgetting to average**: Must divide by batch size
2. **Wrong order**: Forward all, then backward all, then average
3. **Not recomputing**: Must verify improvement after update

---

## Next Steps

- Try more epochs
- Experiment with batch size
- See Example 5 for feed-forward layers

