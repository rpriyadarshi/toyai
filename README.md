# ToyAI: Progressive Transformer Mastery

A comprehensive, hands-on educational system for mastering transformer-based generative AI from first principles. Learn by doing, verify by hand calculation.

## 🎯 What is This?

This is a **complete learning system** that takes you from understanding basic concepts to implementing a full transformer, all using **2x2 matrices** that you can verify with pen and paper.

### Key Features

- **Progressive Learning**: 6 examples, each building on the previous
- **Hand-Calculable**: Every number can be computed by hand
- **Complete Theory**: Full mathematical derivations and intuitive explanations
- **Modular Code**: Object-oriented design with clear separation of concerns
- **Comprehensive Book**: Single source of truth document covering everything

## 📚 The Book

**`BOOK.md`** is your primary learning resource. It contains:

- **Part I: Foundations** - Why transformers? Matrix operations, embeddings, attention intuition
- **Part II: Progressive Examples** - 6 examples from simple forward pass to complete transformer
- **Full Theory** - Mathematical derivations, proofs, and intuitive explanations
- **Hand-Calculation Guides** - Step-by-step worksheets for each example

The book is written in Markdown (convertible to Word/PDF) and serves as the single source of truth.

## 🏗️ Structure

```
toyai-1/
├── BOOK.md                    # Main book document (start here!)
├── README.md                   # This file
├── LICENSE                      # MIT License
├── CMakeLists.txt              # Build system
├── src/
│   └── core/                   # Core OOP classes
│       ├── Matrix.hpp/cpp      # 2x2 matrix operations
│       ├── Embedding.hpp/cpp   # Token embeddings
│       ├── LinearProjection.hpp/cpp  # Q/K/V projections
│       ├── Attention.hpp/cpp   # Scaled dot-product attention
│       ├── Softmax.hpp/cpp     # Softmax activation
│       ├── Loss.hpp/cpp        # Loss functions
│       ├── Optimizer.hpp/cpp   # Gradient descent
│       └── TransformerBlock.hpp/cpp  # Complete transformer block
├── examples/
│   ├── example1_forward_only/  # Forward pass only
│   ├── example2_single_step/   # One training step
│   ├── example3_full_backprop/ # Complete backpropagation
│   ├── example4_multiple_patterns/  # Multiple examples
│   ├── example5_feedforward/   # Add feed-forward layer
│   └── example6_complete/      # Full transformer
└── worksheets/                 # Hand-calculation guides
    ├── example1_worksheet.md
    └── ...
```

## 🚀 Getting Started

### 1. Read the Book

Start with **`BOOK.md`**. Read Part I (Foundations) to understand the core concepts.

### 2. Work Through Examples

Each example in `examples/` demonstrates one new concept:

- **Example 1**: Forward pass only - understand how predictions are made
- **Example 2**: Single training step - see how one weight update works
- **Example 3**: Full backpropagation - complete gradient flow
- **Example 4**: Multiple patterns - batch training
- **Example 5**: Feed-forward layers - non-linearity and depth
- **Example 6**: Complete transformer - everything together

### 3. Verify by Hand

Use the worksheets in `worksheets/` to compute each example step-by-step on paper.

### 4. Build and Run

```bash
# Build all examples
mkdir build && cd build
cmake ..
make

# Run an example
./examples/example1_forward_only/example1
```

## 📖 Learning Path

1. **Read BOOK.md Chapter 1-4** (Foundations)
2. **Study Example 1** - Read chapter, understand code, do worksheet
3. **Study Example 2** - Add training concepts
4. **Study Example 3** - Master backpropagation
5. **Study Example 4** - Understand batch learning
6. **Study Example 5** - Add non-linearity
7. **Study Example 6** - Complete architecture

Each example builds on the previous, so work through them sequentially.

## 🎓 What You'll Master

- **Matrix Operations**: Why matrices? How do they enable learning?
- **Embeddings**: Converting tokens to vectors
- **Attention Mechanism**: Query, Key, Value - what they mean and why
- **Softmax**: Converting scores to probabilities
- **Loss Functions**: Measuring prediction error
- **Gradient Descent**: How models learn
- **Backpropagation**: Computing gradients through complex functions
- **Feed-Forward Networks**: Adding non-linearity
- **Complete Architecture**: Layer norm, residuals, multi-layer

## 🔬 Hand Calculation

Every example uses **2x2 matrices** so you can:

- Compute every step by hand
- Verify code output matches your calculations
- Build deep intuition through manual computation
- Catch errors by comparing results

The worksheets provide templates and step-by-step guides.

## 🧮 Theory Meets Practice

Each concept has:

- **Intuition**: Plain-language explanation
- **Formal Definition**: Mathematical notation
- **Derivation**: Step-by-step proofs
- **Implementation**: Working code
- **Hand Calculation**: Paper-and-pencil verification

## 🔗 Connection to Real Models

The math is **identical** to production transformers:

| Concept | Our Example | GPT-3 |
|---------|------------|-------|
| Embedding dimension | 2 | 12,288 |
| Attention heads | 1 | 96 |
| Sequence length | 2 | 2,048 |
| Parameters | ~20 | 175 billion |
| **Core math** | **Identical** | **Identical** |

Scale changes nothing about the fundamental operations.

## 📚 Prerequisites

- Basic linear algebra (matrix multiplication, dot products)
- Basic calculus (derivatives, chain rule)
- C++ basics (or willingness to learn)
- **No deep learning experience required!**

## 🛠️ Building

### Requirements

- C++11 compatible compiler (g++, clang++)
- CMake 3.10+

### Build Instructions

```bash
mkdir build
cd build
cmake ..
make
```

All examples will be built. Run them individually to see step-by-step output.

## 📝 License

MIT License - see [LICENSE](LICENSE) file.

## 🎯 Philosophy

This project follows these principles:

1. **Progressive Complexity**: Start simple, add one concept at a time
2. **Hand-Verifiable**: Every number can be computed by hand
3. **Complete Theory**: Full mathematical backdrop, not just code
4. **Single Source of Truth**: BOOK.md is the master document
5. **Modular Design**: Clean OOP structure, easy to understand and extend

## 🤝 Contributing

This is an educational project. Suggestions for clarity, additional examples, or improved explanations are welcome!

---

**Start your journey**: Open `BOOK.md` and begin with Part I: Foundations.
