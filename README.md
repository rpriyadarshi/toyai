# ToyAI: Understanding the Matrix Core of Generative AI

A hands-on educational implementation demonstrating the fundamental matrix operations at the heart of transformer-based generative AI models (GPT, BERT, LLaMA, etc.).

## 🎯 What is Generative AI's "Matrix Core"?

At its heart, generative AI is built on **dense matrix operations**. The "matrix core" refers to:

1. **Linear Projections** - Matrix multiplications that transform embeddings to Query (Q), Key (K), and Value (V) spaces
2. **Attention Computation** - Q×K^T to compute similarity scores
3. **Value Aggregation** - Weighted sum of values using attention weights
4. **Feedforward Networks** - Additional matrix operations for non-linear transformations

### Why Matrices?

- **Expressiveness**: Linear transformations can learn complex relationships
- **Parallelization**: GPUs/TPUs have dedicated "tensor cores" for matrix multiply
- **Gradient Flow**: Clean derivatives enable effective training via backpropagation
- **Compositional**: Stacking matrix operations creates depth and representational power

## 🏗️ How is the Attention Mechanism Organized?

The scaled dot-product attention formula:

```
Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V
```

### Step-by-Step Organization

| Step | Operation | Purpose | Mathematical Form |
|------|-----------|---------|-------------------|
| 1 | Input Embedding | Convert tokens to vectors | X ∈ ℝ^(n×d) |
| 2 | Linear Projection | Create Q, K, V | Q=XW_q, K=XW_k, V=XW_v |
| 3 | Attention Scores | Measure similarity | S = Q × K^T |
| 4 | Scaling | Stabilize gradients | S' = S / √d_k |
| 5 | Softmax | Convert to probabilities | A = softmax(S') |
| 6 | Value Aggregation | Weighted combination | Output = A × V |

## 📝 2x2 Matrix Example (Hand-Calculable)

This repository includes a complete implementation using **2x2 matrices** that you can verify with pen and paper.

### Example Forward Pass

```
Input X:        Q = X × Wq:      K = X × Wk:      V = X × Wv:
[1  0]          [0.5  0  ]       [1  0]           [0.5  0.5]
[0  1]          [0    0.5]       [0  1]           [0.5  0.5]

Scores = Q × K^T / √2:    Attention Weights (softmax):    Output = Weights × V:
[0.354  0   ]              [0.588  0.412]                  [0.5  0.5]
[0      0.354]             [0.412  0.588]                  [0.5  0.5]
```

### Key Insight: Why Each Step Matters

1. **Q × K^T**: Computes ALL pairwise dot products at once. `Score[i][j]` = how much token i should attend to token j.

2. **÷ √d_k**: Without scaling, dot products grow with dimension (variance ≈ d_k). Large values → softmax saturation → vanishing gradients.

3. **Softmax**: Converts arbitrary scores to a probability distribution. Each row sums to 1, making it a proper weighted average.

4. **× V**: The actual information retrieval. High attention = more contribution from that value vector.

## 🔄 Training: Backpropagation Through Attention

The backward pass uses the **chain rule** to compute gradients:

```
Loss ← Output ← Attention ← (Q, K, V) ← Weights
```

### Gradient Flow (Simplified)

```
dL/dOutput     = 0.5 × (output - target)           // MSE gradient
dL/dWeights    = dOutput × V^T                      // Matmul backward (left)
dL/dV          = Weights^T × dOutput                // Matmul backward (right)  
dL/dScores     = softmax_backward(weights, dWeights) // Softmax Jacobian
dL/dQ          = dScores × K                        // Q×K^T backward
dL/dK          = dScores^T × Q                      // Q×K^T backward
dL/dWq         = X^T × dL/dQ                        // Projection backward
```

## 📁 Repository Contents

```
toyai/
├── README.md              # This file - conceptual explanation
├── transformer_2x2.cpp    # Complete C++ implementation
└── LICENSE                # MIT License
```

## 🛠️ Building and Running

### Prerequisites
- C++11 compatible compiler (g++, clang++)

### Compile and Run

```bash
# Compile
g++ -std=c++11 -O2 transformer_2x2.cpp -o transformer_2x2

# Run
./transformer_2x2
```

### Expected Output

The program demonstrates:
1. **Forward Pass**: Step-by-step attention computation with printed intermediate values
2. **Training Loop**: Loss decreasing over 10 epochs
3. **Backpropagation Trace**: Detailed gradient computation for each operation

## 🧮 Mathematical Deep Dive

### Why These Specific Operations?

| Operation | Mathematical Reason | Practical Benefit |
|-----------|--------------------| ------------------|
| **Matrix Multiply** | Universal function approximator (with non-linearity) | Captures any linear relationship |
| **Transpose** | Aligns dimensions for dot product computation | Efficient batch similarity |
| **Softmax** | Differentiable argmax, creates valid probability | Enables gradient-based training |
| **Scaling** | Keeps variance O(1) regardless of dimension | Numerical stability |

### Softmax Jacobian (The Tricky Part)

For softmax output s, the Jacobian is:
```
∂s_i/∂x_j = s_i × (δ_ij - s_j)
```

Where δ_ij is the Kronecker delta. This means:
- Each output depends on ALL inputs (through the normalization sum)
- The gradient has form: `dL/dx = s ⊙ (dL/ds - dot(dL/ds, s))`

## 🔗 Connection to Real Transformers

Our 2x2 example demonstrates the **exact same math** as production models:

| Aspect | Our Example | GPT-3 |
|--------|-------------|-------|
| Embedding dimension | 2 | 12,288 |
| Attention heads | 1 | 96 |
| Sequence length | 2 | 2,048 |
| Parameters | 12 | 175 billion |
| Core math | **Identical** | **Identical** |

Real models add:
- **Multi-head attention**: Multiple parallel attention operations
- **Layer normalization**: Stabilizes training
- **Residual connections**: x + Attention(x) enables gradient flow
- **Positional encodings**: Inject sequence order information
- **Causal masking**: Prevent looking at future tokens (autoregressive)

## 📚 Further Reading

1. **"Attention Is All You Need"** (Vaswani et al., 2017) - Original transformer paper
2. **"The Illustrated Transformer"** (Jay Alammar) - Visual explanation
3. **"Deep Learning"** (Goodfellow, Bengio, Courville) - Mathematical foundations

## 🎓 Key Takeaways

1. **Matrix multiply is the fundamental operation** - Everything else is scaffolding around it
2. **Attention learns what to combine** - Q, K determine relevance; V provides content
3. **Scaling is crucial** - Prevents numerical instability in deep networks
4. **Backprop is just chain rule** - Complex system, simple local gradients
5. **2x2 matrices capture the essence** - Scale changes nothing about the math

## License

MIT License - see [LICENSE](LICENSE) file.
