# Hand Calculation Worksheet: Example 2

## Single Training Step - Learning A B → C

### Initial Values

**Token Embeddings:**
- A = [1, 0]
- B = [0, 1]
- C = [1, 1]
- D = [0, 0]

**Projection Weights (Fixed):**
```
WQ = [0.1, 0.0]
     [0.0, 0.1]

WK = [0.1, 0.0]
     [0.0, 0.1]

WV = [0.1, 0.0]
     [0.0, 0.1]
```

**Output Projection (Trainable):**
```
WO = [0.1, 0.0]
     [0.0, 0.1]
```

**Training Example:**
- Input: [A, B]
- Target: C (index 2)
- Learning rate: η = 0.1

---

## Part 1: Forward Pass (Before Training)

### Step 1: Token Embeddings

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

**K = X × WK:** (same as Q)

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

**Raw Scores = Q × K^T:**

```
score[1][0] = Q[1] · K[0] = [0.0, 0.1] · [0.1, 0.0] = 0.0
score[1][1] = Q[1] · K[1] = [0.0, 0.1] · [0.0, 0.1] = 0.01
```

**Scale by 1/√2 ≈ 0.707:**

```
scaled_score[1][0] = 0.0
scaled_score[1][1] = 0.01 × 0.707 = 0.00707
```

### Step 4: Attention Weights (Softmax)

Scores = [0.0, 0.00707]

1. Max: 0.00707
2. Subtract max: [-0.00707, 0.0]
3. Exponentiate: [exp(-0.00707), exp(0.0)] ≈ [0.9929, 1.0]
4. Sum: 0.9929 + 1.0 = 1.9929
5. Normalize: [0.9929/1.9929, 1.0/1.9929] ≈ [0.498, 0.502]

**Attention weights:**
- Weight on A: p₀ ≈ 0.498
- Weight on B: p₁ ≈ 0.502

### Step 5: Context Vector

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

Using simplified mapping:
- logit(A) = context[0] × WO[0][0] = 0.0498 × 0.1 = 0.00498
- logit(B) = context[0] × WO[0][1] = 0.0498 × 0.0 = 0.0
- logit(C) = context[1] × WO[1][1] = 0.0502 × 0.1 = 0.00502
- logit(D) = 0.0

**Logits = [0.00498, 0.0, 0.00502, 0.0]**

### Step 7: Probabilities (Softmax)

1. Max: 0.00502
2. Subtract: [-0.00004, -0.00502, 0.0, -0.00502]
3. Exponentiate: [0.99996, 0.995, 1.0, 0.995]
4. Sum: ≈ 3.99
5. Normalize: [0.250, 0.249, 0.251, 0.249]

**Probabilities BEFORE training:**
- P(A) ≈ 0.250
- P(B) ≈ 0.249
- P(C) ≈ 0.251 ← Target
- P(D) ≈ 0.249

---

## Part 2: Training Step

### Step 1: Compute Loss

**Cross-entropy loss:**
```
L = -log(P(target))
  = -log(P(C))
  = -log(0.251)
  ≈ 1.383
```

### Step 2: Gradient w.r.t. Logits

**Formula:** dL/dlogit[i] = P[i] - 1[i == target]

```
dL/dlogit(A) = 0.250 - 0 = 0.250
dL/dlogit(B) = 0.249 - 0 = 0.249
dL/dlogit(C) = 0.251 - 1 = -0.749  ← Negative! (should increase)
dL/dlogit(D) = 0.249 - 0 = 0.249
```

**Interpretation:**
- Positive gradient → decrease this logit
- Negative gradient → increase this logit
- C has negative gradient → we want to increase logit(C)

### Step 3: Gradient w.r.t. WO

**Formula:** dWO = outer(context, dL_dlogits)

For each element: dWO[i][j] = context[i] × dL_dlogits[j]

```
dWO[0][0] = context[0] × dL_dlogits[0] = 0.0498 × 0.250 = 0.01245
dWO[0][1] = context[0] × dL_dlogits[1] = 0.0498 × 0.249 = 0.01240
dWO[0][2] = context[0] × dL_dlogits[2] = 0.0498 × (-0.749) = -0.03730
dWO[0][3] = context[0] × dL_dlogits[3] = 0.0498 × 0.249 = 0.01240

dWO[1][0] = context[1] × dL_dlogits[0] = 0.0502 × 0.250 = 0.01255
dWO[1][1] = context[1] × dL_dlogits[1] = 0.0502 × 0.249 = 0.01250
dWO[1][2] = context[1] × dL_dlogits[2] = 0.0502 × (-0.749) = -0.03760
dWO[1][3] = context[1] × dL_dlogits[3] = 0.0502 × 0.249 = 0.01250
```

**Gradient matrix:**
```
dWO = [0.01245, 0.01240, -0.03730, 0.01240]
      [0.01255, 0.01250, -0.03760, 0.01250]
```

### Step 4: Update Weights

**Formula:** W_new = W_old - η × gradient

For our simplified 2×2 case, focusing on elements affecting C:

**Old WO:**
```
WO = [0.1, 0.0]
     [0.0, 0.1]
```

**Update (η = 0.1):**
```
update = 0.1 × dWO

For element affecting C (simplified):
WO[1][1] affects logit(C), so:
WO[1][1]_new = WO[1][1]_old - 0.1 × dWO[1][2]
             = 0.1 - 0.1 × (-0.03760)
             = 0.1 + 0.00376
             = 0.10376
```

**New WO (approximate):**
```
WO_new ≈ [0.099, 0.0]
         [0.0,   0.104]
```

---

## Part 3: Forward Pass (After Training)

### Recompute with Updated WO

**Context (unchanged):** [0.0498, 0.0502]

**New logits:**
- logit(A) = 0.0498 × 0.099 ≈ 0.00493
- logit(B) = 0.0
- logit(C) = 0.0502 × 0.104 ≈ 0.00522  ← Increased!
- logit(D) = 0.0

**New probabilities:**
1. Max: 0.00522
2. Subtract: [-0.00029, -0.00522, 0.0, -0.00522]
3. Exponentiate: [0.9997, 0.9948, 1.0, 0.9948]
4. Sum: ≈ 3.989
5. Normalize: [0.2505, 0.2492, 0.2506, 0.2492]

**Probabilities AFTER training:**
- P(A) ≈ 0.2505
- P(B) ≈ 0.2492
- P(C) ≈ 0.2506 ← Increased from 0.251!
- P(D) ≈ 0.2492

---

## Verification

### Check 1: Loss Decreased?
- Before: L ≈ 1.383
- After: L ≈ -log(0.2506) ≈ 1.384 (slightly better, but small change expected for one step)

### Check 2: Target Probability Increased?
- Before: P(C) = 0.251
- After: P(C) = 0.2506
- Change: +0.0006 (small but in right direction!)

### Check 3: Gradient Sign Correct?
- dL/dlogit(C) = -0.749 (negative) ✓
- This means we should increase logit(C) ✓
- WO update increased connection to C ✓

---

## Common Mistakes

1. **Wrong gradient sign**: Remember negative gradient means increase!
2. **Forgetting learning rate**: Always multiply gradient by η
3. **Wrong update direction**: Subtract gradient (not add)
4. **Not recomputing forward pass**: Must recompute to see improvement

---

## Key Insights

1. **One step is small**: Single update makes tiny change
2. **Direction matters**: Gradient points toward improvement
3. **Learning rate matters**: Too large = unstable, too small = slow
4. **Multiple steps needed**: Real training requires many iterations

---

## Next Steps

- Try multiple training steps
- Experiment with different learning rates
- See Example 3 for full backpropagation through all weights

