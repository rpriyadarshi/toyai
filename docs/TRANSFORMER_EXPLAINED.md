# Generative AI: The Matrix Core of Transformers

## Table of Contents
1. [What is the Matrix Core?](#what-is-the-matrix-core)
2. [How Transformers are Organized](#how-transformers-are-organized)
3. [The Attention Mechanism](#the-attention-mechanism)
4. [Hand-Calculable 2x2 Example](#hand-calculable-2x2-example)
5. [Training: Forward Pass, Loss, and Backpropagation](#training-forward-pass-loss-and-backpropagation)
6. [Inference: Forward Pass Only](#inference-forward-pass-only)
7. [Why Each Step Matters](#why-each-step-matters)

---

## What is the Matrix Core?

Generative AI, particularly transformer-based models like GPT, is fundamentally built on **matrix operations**. The "matrix core" refers to the mathematical foundation where:

1. **Data is represented as matrices** (embeddings)
2. **Transformations are learned weight matrices** 
3. **Computations are matrix multiplications**

### Why Matrices?

- **Parallelism**: Matrix operations can be massively parallelized on GPUs
- **Expressiveness**: Linear algebra can approximate any function when combined with non-linearities
- **Gradient flow**: Derivatives of matrix operations are well-defined, enabling learning

---

## How Transformers are Organized

A transformer consists of stacked layers, each containing:

```
Input Embeddings
      ↓
┌─────────────────────────────────────┐
│         TRANSFORMER BLOCK           │
│  ┌──────────────────────────────┐   │
│  │   Multi-Head Self-Attention  │   │
│  │   (Q, K, V matrices)         │   │
│  └──────────────────────────────┘   │
│              ↓                      │
│  ┌──────────────────────────────┐   │
│  │   Add & Layer Normalization  │   │
│  └──────────────────────────────┘   │
│              ↓                      │
│  ┌──────────────────────────────┐   │
│  │   Feed-Forward Network       │   │
│  │   (Two linear layers + ReLU) │   │
│  └──────────────────────────────┘   │
│              ↓                      │
│  ┌──────────────────────────────┐   │
│  │   Add & Layer Normalization  │   │
│  └──────────────────────────────┘   │
└─────────────────────────────────────┘
      ↓ (Repeat N times)
Output Layer
```

---

## The Attention Mechanism

The core innovation is **Self-Attention**, computed as:

```
Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V
```

### Breaking This Down:

| Component | What It Is | Why It Matters |
|-----------|------------|----------------|
| **Q** (Query) | What am I looking for? | Encodes the current position's question |
| **K** (Key) | What do I contain? | Encodes what each position offers |
| **V** (Value) | What information do I hold? | The actual content to retrieve |
| **Q × K^T** | Compatibility scores | How relevant is each position? |
| **√d_k** | Scaling factor | Prevents vanishing gradients in softmax |
| **softmax** | Normalize to probabilities | Creates weighted combination |
| **× V** | Weighted sum | Aggregates relevant information |

---

## Hand-Calculable 2x2 Example

Let's work through a **complete example** with 2x2 matrices you can verify by hand.

### Setup

```
Input X (2 tokens, 2-dimensional embeddings):
X = [1  0]
    [0  1]

Weight matrices (learned parameters):
W_Q = [1  0]    W_K = [0  1]    W_V = [1  1]
      [0  1]          [1  0]          [0  1]
```

### Step 1: Compute Q, K, V

```
Q = X × W_Q = [1  0] × [1  0] = [1  0]
              [0  1]   [0  1]   [0  1]

K = X × W_K = [1  0] × [0  1] = [0  1]
              [0  1]   [1  0]   [1  0]

V = X × W_V = [1  0] × [1  1] = [1  1]
              [0  1]   [0  1]   [0  1]
```

**WHY**: Each input token gets transformed into three different views:
- Q: "What am I searching for?"
- K: "What can I be found by?"
- V: "What information do I carry?"

### Step 2: Compute Attention Scores

```
Scores = Q × K^T = [1  0] × [0  1] = [0  1]
                   [0  1]   [1  0]   [1  0]
```

**WHY**: Each cell (i,j) tells us: "How much should token i attend to token j?"
- Score[0,0]=0: Token 0 has zero compatibility with itself (via K)
- Score[0,1]=1: Token 0 should attend to token 1
- Score[1,0]=1: Token 1 should attend to token 0
- Score[1,1]=0: Token 1 has zero compatibility with itself

### Step 3: Scale the Scores

```
d_k = 2 (dimension)
√d_k = √2 ≈ 1.414

Scaled = Scores / √d_k = [0     0.707]
                         [0.707   0  ]
```

**WHY**: Without scaling, large dimensions cause dot products to grow large, pushing softmax into regions with tiny gradients. Scaling keeps values in a "learnable" range.

### Step 4: Apply Softmax

```
softmax(row 0) = softmax([0, 0.707])
    e^0 = 1.0
    e^0.707 ≈ 2.028
    sum = 3.028
    result = [1/3.028, 2.028/3.028] ≈ [0.33, 0.67]

softmax(row 1) = softmax([0.707, 0])
    ≈ [0.67, 0.33]

Attention Weights = [0.33  0.67]
                    [0.67  0.33]
```

**WHY**: Softmax converts arbitrary scores to probabilities (sum=1). Each row is a probability distribution over "which tokens to attend to."

### Step 5: Compute Output

```
Output = Attention_Weights × V

Output = [0.33  0.67] × [1  1] = [0.33×1 + 0.67×0  0.33×1 + 0.67×1]
         [0.67  0.33]   [0  1]   [0.67×1 + 0.33×0  0.67×1 + 0.33×1]

       = [0.33  1.0 ]
         [0.67  1.0 ]
```

**WHY**: Each output row is a weighted combination of all V rows, where weights come from the attention distribution. Token 0's output is 33% itself + 67% token 1.

---

## Training: Forward Pass, Loss, and Backpropagation

### Forward Pass (Same as Above)

Compute: `X → Q,K,V → Attention → Output → Prediction`

### Loss Computation

For language models, we use **Cross-Entropy Loss**:

```
Loss = -Σ y_true × log(y_pred)
```

#### Example:
```
Target (one-hot): [1, 0] (token 0 should come next)
Prediction (after softmax): [0.7, 0.3]

Loss = -1×log(0.7) - 0×log(0.3) = 0.357
```

**WHY**: Cross-entropy penalizes confident wrong predictions heavily. If we predicted [0.01, 0.99] when target was [1, 0], loss would be -log(0.01) = 4.6 (much worse!).

### Backpropagation

We compute gradients of Loss with respect to each weight matrix.

#### For Our 2x2 Example:

**Step A: Gradient of Loss w.r.t. Output**
```
dL/dOutput = y_pred - y_true = [0.7-1, 0.3-0] = [-0.3, 0.3]
```

**Step B: Gradient w.r.t. V (through Attention × V)**
```
dL/dV = Attention^T × dL/dOutput
```

Using our attention weights [0.33, 0.67] and gradient [-0.3, 0.3]:
```
dL/dV[0] = 0.33 × [-0.3, 0.3] = [-0.099, 0.099]
dL/dV[1] = 0.67 × [-0.3, 0.3] = [-0.201, 0.201]
```

**Step C: Gradient w.r.t. W_V**
```
dL/dW_V = X^T × dL/dV
```

**WHY**: The chain rule decomposes complex gradients into products of simpler gradients. Each matrix multiplication's gradient involves the transpose of the other operand.

#### Weight Update:
```
W_V_new = W_V - learning_rate × dL/dW_V
```

Example with lr=0.1:
```
W_V_new = [1  1] - 0.1 × [-0.099  0.099] ≈ [1.01  0.99]
          [0  1]         [-0.201  0.201]   [0.02  0.98]
```

**WHY**: We move weights in the direction that decreases loss. Negative gradient = direction of steepest descent.

---

## Inference: Forward Pass Only

During inference, we skip:
1. Loss computation (no target labels)
2. Gradient computation
3. Weight updates

Just run the forward pass and use the output!

```
Input "Hello" → Embedding → Attention → Output → Softmax → "World" (highest probability)
```

---

## Why Each Step Matters

### 1. **Matrix Multiplication (X × W)**
- **What**: Linear transformation
- **Why**: Changes the representation space; learned weights encode patterns
- **Meaning**: Rotating and scaling the embedding space to highlight useful features

### 2. **Q, K, V Decomposition**
- **What**: Three different projections of the same input
- **Why**: Separates "what to search for" from "what to match against" from "what to return"
- **Meaning**: Like a database query where you search by key but return value

### 3. **Dot Product (Q × K^T)**
- **What**: Similarity measurement
- **Why**: Dot product measures alignment/compatibility
- **Meaning**: Higher dot product = vectors pointing same direction = similar concepts

### 4. **Scaling (/ √d_k)**
- **What**: Normalize by dimension
- **Why**: Variance of dot products scales with dimension
- **Meaning**: Keeps gradients healthy regardless of model size

### 5. **Softmax**
- **What**: Convert scores to probabilities
- **Why**: Need weights that sum to 1 for meaningful combination
- **Meaning**: "Spend 100% of attention across all tokens"

### 6. **Weighted Sum (Attention × V)**
- **What**: Aggregate information
- **Why**: Combines information from attended positions
- **Meaning**: "My new representation = weighted average of everyone's content"

### 7. **Cross-Entropy Loss**
- **What**: Measure prediction error
- **Why**: Differentiable, penalizes confident mistakes
- **Meaning**: "How surprised are we by the correct answer?"

### 8. **Backpropagation**
- **What**: Compute gradients via chain rule
- **Why**: Need direction to improve weights
- **Meaning**: "How much did each weight contribute to the error?"

### 9. **Gradient Descent Update**
- **What**: Adjust weights opposite to gradient
- **Why**: Move toward lower loss
- **Meaning**: "Make the model slightly better at this example"

---

## Key Insights

1. **Attention is dynamic routing**: Unlike fixed convolutions, attention learns WHICH information to combine based on content.

2. **Everything is differentiable**: Every operation has a well-defined gradient, enabling end-to-end learning.

3. **Matrices enable parallelism**: All attention computations can happen simultaneously on GPU.

4. **The magic is in the weights**: Q, K, V transformations are where knowledge is stored.

---

## Summary Table

| Operation | Math | Complexity | Purpose |
|-----------|------|------------|---------|
| Embedding lookup | X[i] | O(1) | Convert tokens to vectors |
| Q/K/V projection | X × W | O(n×d²) | Create attention components |
| Attention scores | Q × K^T | O(n²×d) | Compute all-pairs compatibility |
| Softmax | exp(x)/Σexp | O(n²) | Normalize to probabilities |
| Attention output | Weights × V | O(n²×d) | Aggregate information |
| Loss | -Σy×log(p) | O(vocab) | Measure prediction error |
| Backprop | Chain rule | O(forward) | Compute gradients |

Where n = sequence length, d = dimension, vocab = vocabulary size.

---

## Next: See the Code

Check out `src/tiny_transformer.cpp` for a complete C++ implementation with these exact 2x2 matrices!
