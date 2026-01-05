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

The book is organized in two ways:

### Individual Chapters (`book/` directory)
- **Modular format**: Each chapter is a separate file for easy navigation
- **Start with**: [Introduction](book/00-introduction.md) for philosophy and approach
- **Then**: [Table of Contents](book/00-index.md) for navigation
- **Read online**: View chapters directly in your editor or on GitHub

### Complete PDF Book
Generate a professional, downloadable PDF with all content:

```bash
python3 scripts/build_book.py
```

This creates `output/Understanding_Transformers_Complete.pdf` containing:
- **Part I: Foundations** - Neural networks, learning algorithms, matrix operations, embeddings, attention intuition
- **Part II: Progressive Examples** - 7 examples from simple forward pass to complete transformer
- **Appendices** - Reference materials
- **Worksheets** - All hand-calculation guides
- **Code Examples** - Complete C++ implementations

The PDF includes:
- Professional formatting with table of contents
- Working internal links
- Code syntax highlighting
- Mathematical formulas properly rendered
- All content in one downloadable file

See [scripts/UPDATE_BOOK.md](scripts/UPDATE_BOOK.md) for details on updating the book.

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

Start with **[Introduction](book/00-introduction.md)** to understand the philosophy and approach of this book. Then read **[Table of Contents](book/00-index.md)** or **[BOOK.md](BOOK.md)** for navigation. Read Part I (Foundations, Chapters 1-8) to understand the core concepts. **[Chapter 1: Neural Networks and the Perceptron](book/01-neural-networks-perceptron.md)** explains the fundamental building blocks with physical analogies - start there!

### 2. Work Through Examples

Each example in `examples/` demonstrates one new concept:

- **Example 9**: Forward pass only - understand how predictions are made
- **Example 10**: Single training step - see how one weight update works
- **Example 11**: Full backpropagation - complete gradient flow
- **Example 12**: Multiple patterns - batch training
- **Example 13**: Feed-forward layers - non-linearity and depth
- **Example 14**: Complete transformer - everything together
- **Example 15**: Character recognition - complete classification example

### 3. Verify by Hand

Use the worksheets in the [`worksheets/`](worksheets/) directory to compute each example step-by-step on paper.

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

1. **Read Introduction** - Understand the philosophy and approach ([book/00-introduction.md](book/00-introduction.md))
2. **Read Foundations (Chapters 1-5)** - Start with terminology, then matrix operations, embeddings, attention, and why transformers
   - [Chapter 1: Neural Networks and the Perceptron](book/01-neural-networks-perceptron.md) (**START HERE!**)
   - [Chapter 2: Multilayer Networks and Architecture](book/02-multilayer-networks-architecture.md)
   - [Chapter 3: Learning Algorithms](book/03-learning-algorithms.md)
   - [Chapter 4: Training Neural Networks](book/04-training-neural-networks.md)
   - [Chapter 5: The Matrix Core](book/05-matrix-core.md)
   - [Chapter 6: Embeddings: Tokens to Vectors](book/06-embeddings.md)
   - [Chapter 7: Attention Intuition](book/07-attention-intuition.md)
   - [Chapter 8: Why Transformers?](book/08-why-transformers.md)
2. **Study Example 9** - Read chapter, understand code, do worksheet
3. **Study Example 10** - Add training concepts
4. **Study Example 11** - Master backpropagation
5. **Study Example 12** - Understand batch learning
6. **Study Example 13** - Add non-linearity
7. **Study Example 14** - Complete architecture
8. **Study Example 15** - Character recognition

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

**Start your journey**: Read [Introduction](book/00-introduction.md) first, then proceed to [Part I: Foundations](book/00-index.md#part-i-foundations).
