# Hand Calculation Worksheet: Example 7

## Character Recognition - Forward Pass Only

### Initial Values

**Input Image (2×2 pixels, vertical line pattern):**
```
Input = [0.1, 0.9]
        [0.1, 0.9]

Flattened: [0.1, 0.9, 0.1, 0.9]
Interpretation: Vertical line (represents digit '1')
```

**Hidden Layer Weights:**
```
W1 = [0.2, 0.3]
     [0.1, 0.4]

b1 = [0.1, 0.05]
     [0.0, 0.0]

(Only first 2 elements of b1 are used)
```

**Output Layer Weights:**
```
W2 = [0.3, 0.2]
     [0.1, 0.4]

b2 = [0.05, 0.1]
     [0.0, 0.0]

(Adapted for 4 outputs)
```

### Step 1: Input Representation

Input image (2×2 pixels):
```
Input = [0.1, 0.9]   ← Row 0: light, dark
        [0.1, 0.9]   ← Row 1: light, dark

Flattened: [0.1, 0.9, 0.1, 0.9]
Pattern: Vertical line (digit '1')
```

### Step 2: Hidden Layer Computation

**Compute: hidden = W1 × input + b1**

Since we have 4 inputs and 2 hidden neurons, we map:
- hidden[0] uses input[0] and input[1]
- hidden[1] uses input[2] and input[3]

**hidden[0] = W1[0][0] × input[0] + W1[0][1] × input[1] + b1[0]**

```
hidden[0] = 0.2 × 0.1 + 0.3 × 0.9 + 0.1
          = 0.02 + 0.27 + 0.1
          = 0.39
```

**hidden[1] = W1[1][0] × input[2] + W1[1][1] × input[3] + b1[1]**

```
hidden[1] = 0.1 × 0.1 + 0.4 × 0.9 + 0.05
          = 0.01 + 0.36 + 0.05
          = 0.42
```

**Hidden (before ReLU):**
```
hidden = [0.39, 0.42]
```

### Step 3: ReLU Activation

**ReLU: max(0, x) applied element-wise**

```
hidden_relu[0] = max(0, 0.39) = 0.39
hidden_relu[1] = max(0, 0.42) = 0.42
```

**Hidden (after ReLU):**
```
hidden_relu = [0.39, 0.42]
```

**Key Insight:** ReLU outputs continuous values, not binary 0/1. There is no "firing" - the neuron outputs 0.39, not "fired" or "not fired".

### Step 4: Output Layer Computation

**Compute: logits = W2 × hidden_relu + b2**

For 4 output classes, we compute:
- logit[0] = W2[0][0] × hidden[0] + W2[0][1] × hidden[1] + b2[0]
- logit[1] = W2[1][0] × hidden[0] + W2[1][1] × hidden[1] + b2[1]
- logit[2] = W2[0][0] × hidden[0] + W2[0][1] × hidden[1] + b2[2] (adapted)
- logit[3] = W2[1][0] × hidden[0] + W2[1][1] × hidden[1] + b2[3] (adapted)

**logit[0] (Digit 0):**
```
logit[0] = 0.3 × 0.39 + 0.2 × 0.42 + 0.05
         = 0.117 + 0.084 + 0.05
         = 0.251
```

**logit[1] (Digit 1):**
```
logit[1] = 0.1 × 0.39 + 0.4 × 0.42 + 0.1
         = 0.039 + 0.168 + 0.1
         = 0.307
```

**logit[2] (Digit 2):**
```
logit[2] = 0.3 × 0.39 + 0.2 × 0.42 + 0.0
         = 0.117 + 0.084 + 0.0
         = 0.201
```

**logit[3] (Digit 3):**
```
logit[3] = 0.1 × 0.39 + 0.4 × 0.42 + 0.0
         = 0.039 + 0.168 + 0.0
         = 0.207
```

**Logits (raw scores):**
```
logits = [0.251, 0.307, 0.201, 0.207]
         [Digit 0, Digit 1, Digit 2, Digit 3]
```

### Step 5: Softmax Calculation

**Convert logits to probabilities**

**Logits = [0.251, 0.307, 0.201, 0.207]**

1. **Find maximum:**
   ```
   max = 0.307
   ```

2. **Subtract maximum (for numerical stability):**
   ```
   shifted = [0.251 - 0.307, 0.307 - 0.307, 0.201 - 0.307, 0.207 - 0.307]
           = [-0.056, 0.0, -0.106, -0.100]
   ```

3. **Exponentiate:**
   ```
   exp_values = [exp(-0.056), exp(0.0), exp(-0.106), exp(-0.100)]
               ≈ [0.945, 1.0, 0.899, 0.905]
   ```

4. **Sum:**
   ```
   sum = 0.945 + 1.0 + 0.899 + 0.905
       = 3.749
   ```

5. **Normalize (divide by sum):**
   ```
   prob[0] = 0.945 / 3.749 ≈ 0.252
   prob[1] = 1.0 / 3.749 ≈ 0.267
   prob[2] = 0.899 / 3.749 ≈ 0.240
   prob[3] = 0.905 / 3.749 ≈ 0.241
   ```

**Probabilities:**
```
probabilities = [0.252, 0.267, 0.240, 0.241]
                [Digit 0, Digit 1, Digit 2, Digit 3]
```

**Verification:**
- ✓ Sum ≈ 1.0 (0.252 + 0.267 + 0.240 + 0.241 = 1.000)
- ✓ All probabilities are positive
- ✓ Largest logit (0.307) gets highest probability (0.267)

### Step 6: Prediction (Argmax)

**Select class with highest probability:**

```
probabilities = [0.252, 0.267, 0.240, 0.241]
                 [Digit 0, Digit 1, Digit 2, Digit 3]
                              ↑
                         Highest (0.267)
```

**Prediction: Digit 1 (confidence: 26.7%)**

### Summary

**Forward Pass:**
1. Input: [0.1, 0.9, 0.1, 0.9] (vertical line pattern)
2. Hidden: [0.39, 0.42] (after ReLU)
3. Logits: [0.251, 0.307, 0.201, 0.207]
4. Probabilities: [0.252, 0.267, 0.240, 0.241]
5. Prediction: Digit 1 (26.7% confidence)

### Key Insights

1. **Continuous Activations**: ReLU outputs 0.39 and 0.42 - continuous values, not binary
2. **No "Firing"**: Neurons don't "fire" or "not fire" - they output continuous values
3. **Softmax Classification**: Same function used in transformers (Examples 1-6)
4. **Discrete Decision**: Made via argmax on continuous probabilities
5. **Universal Principles**: Same fundamentals apply to transformers and classification

### Verification

Compare your hand calculation to the code output. They should match (within rounding error).

**Expected Code Output:**
- Hidden (after ReLU): [0.39, 0.42]
- Logits: [0.251, 0.307, 0.201, 0.207] (approximately)
- Probabilities: [0.252, 0.267, 0.240, 0.241] (approximately)
- Prediction: Digit 1

### Common Mistakes

1. **Forgetting ReLU**: Always apply max(0, x) to hidden layer
2. **Softmax numerical issues**: Always subtract max before exponentiating
3. **Dimension mismatches**: Check that matrix dimensions match at each step
4. **Confusing continuous with discrete**: Remember all activations are continuous until argmax

### Next Steps

1. Verify your calculations match the code output
2. Try a different input pattern (e.g., horizontal line for digit "0")
3. Explain why all values are continuous (no "firing")
4. Compare to transformer examples (Examples 1-6) - same softmax, same principles

