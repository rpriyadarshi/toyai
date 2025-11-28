# ToyAI - Understanding Generative AI from First Principles

A minimal, educational implementation of transformer/attention mechanisms with 2x2 matrices you can calculate by hand.

## What's Inside?

### 📚 Documentation
- **[docs/TRANSFORMER_EXPLAINED.md](docs/TRANSFORMER_EXPLAINED.md)** - Complete explanation of:
  - What is the matrix core of generative AI?
  - How transformers are organized
  - The attention mechanism (Q, K, V)
  - Hand-calculable 2x2 examples
  - Training: forward pass, loss, and backpropagation
  - Inference: forward pass only
  - WHY each step matters

### 💻 Code
- **[src/tiny_transformer.cpp](src/tiny_transformer.cpp)** - C++ implementation featuring:
  - Complete self-attention layer with 2x2 matrices
  - Verbose forward pass with explanations
  - Full backpropagation implementation
  - Training loop demonstration
  - Extensive comments explaining WHY each operation

## Quick Start

### Build
```bash
cd src
g++ -std=c++17 -o tiny_transformer tiny_transformer.cpp -lm
```

### Run
```bash
./tiny_transformer
```

### What You'll See
1. **Inference**: Step-by-step forward pass through attention
2. **Training**: One training step with full gradient computation
3. **Learning**: Watch loss decrease over 100 epochs

## Key Concepts

### The Matrix Core
Generative AI is fundamentally **matrix operations**:
- Data → Embedding matrices
- Transformations → Weight matrices (learned)
- Computation → Matrix multiplication

### Attention Formula
```
Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V
```

### Training Chain
```
Input → Forward Pass → Loss → Backpropagation → Weight Update
```

## Why 2x2 Matrices?

Everything uses 2×2 matrices so you can:
- ✅ Verify every calculation with pen and paper
- ✅ Understand the math without being overwhelmed
- ✅ See the exact same operations that happen in GPT/BERT (just scaled up)

## Example Calculation

```
Input X = [1  0]    Weight W_Q = [1  0]
          [0  1]                 [0  1]

Q = X × W_Q = [1  0]  ← Each token gets its own query vector
              [0  1]
```

See the full worked example in [docs/TRANSFORMER_EXPLAINED.md](docs/TRANSFORMER_EXPLAINED.md)!

## License

See [LICENSE](LICENSE) file.
