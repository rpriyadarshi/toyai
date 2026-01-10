# Understanding Transformers: From First Principles to Mastery

**A Progressive Learning System with Hand-Calculable Examples**

---

## Table of Contents

### Front Matter

- **[Preface](00a-preface.md)** - What this book is, how to use it, prerequisites, and what you'll gain
- **[Introduction](00c-introduction.md)** - Philosophy and approach: mathematics as operations, explicit transformations, and the operational framework

---

### Part I: Foundations

**Part I** establishes the core concepts needed to understand transformers. Read these chapters in order—each builds on previous material.

1. **[Chapter 1: Neural Networks and the Perceptron](01-neural-networks-perceptron.md)** - **START HERE!** The fundamental building block: single neurons, activation functions, decision boundaries, and physical analogies that connect abstract mathematics to tangible reality

2. **[Chapter 2: Probability and Statistics](02-probability-statistics.md)** - Probability distributions, expected value, entropy, cross-entropy, and statistical concepts essential for understanding loss functions, softmax, and normalization

3. **[Chapter 3: Multilayer Networks and Architecture](03-multilayer-networks-architecture.md)** - Layers, network design, feedforward networks, and how hierarchical learning emerges from simple building blocks

4. **[Chapter 4: Learning Algorithms](04-learning-algorithms.md)** - Loss functions, gradient descent, and backpropagation: the three fundamental algorithms that enable neural networks to learn

5. **[Chapter 5: Training Neural Networks](05-training-neural-networks.md)** - Training loops, batch processing, epochs, and the transition from neural networks to transformers

6. **[Chapter 6: Embeddings: Tokens to Vectors](06-embeddings.md)** - How discrete tokens (words, characters) become continuous vectors that neural networks can process

7. **[Chapter 7: Attention Intuition](07-attention-intuition.md)** - Query/Key/Value mechanism, attention scores, and how transformers create context-aware representations

8. **[Chapter 8: Why Transformers?](08-why-transformers.md)** - The problems transformers solve, why previous architectures struggled, and what makes transformers uniquely powerful

---

### Part II: Progressive Examples

**Part II** provides hands-on examples that build from simple to complex. Each example includes step-by-step mathematics, hand-calculation worksheets, and working C++ code. Work through these sequentially—each builds on the previous.

9. **[Example 1: Minimal Forward Pass](09-example1-forward-pass.md)** - Understand how a transformer makes predictions (no training yet). Learn forward pass computation, attention mechanism step-by-step, context creation, and how predictions are made

10. **[Example 2: Single Training Step](10-example2-single-step.md)** - Understand how one weight update works. Learn loss functions, gradient computation, gradient descent, and how models learn from examples

11. **[Example 3: Full Backpropagation](11-example3-full-backprop.md)** - Understand complete gradient flow through all components. Learn backpropagation through attention, matrix calculus, gradient flow through Q/K/V, and the complete training loop

12. **[Example 4: Multiple Patterns](12-example4-multiple-patterns.md)** - Learn multiple patterns from multiple examples. Learn batch training, gradient accumulation, pattern learning, and convergence

13. **[Example 5: Feed-Forward Layers](13-example5-feedforward.md)** - Add non-linearity and depth. Learn feed-forward networks, non-linear activations (ReLU), residual connections, and layer composition

14. **[Example 6: Complete Transformer](14-example6-complete.md)** - Full implementation with all components. Learn complete transformer architecture, layer normalization, multiple layers, and end-to-end training

15. **[Example 7: Character Recognition](15-example7-character-recognition.md)** - Understand how neural networks perform classification tasks using continuous activations. Learn feed-forward networks for classification, ReLU activation, how continuous outputs become discrete predictions via softmax, and the connection to transformer concepts

---

### Appendices

Quick reference materials for matrix calculus, terminology, calculation tips, and common mistakes.

- **[Appendix A: Matrix Operations and Calculus Reference](appendix-a-matrix-calculus.md)** - Matrix operations, matrix inverse, and essential matrix calculus formulas for backpropagation

- **[Appendix B: Terminology Reference](appendix-b-terminology-reference.md)** - Quick reference for all terminology with physical analogies and intuitive explanations

- **[Appendix C: Hand Calculation Tips](appendix-c-hand-calculation-tips.md)** - Practical guidance for computing examples by hand, avoiding common calculation errors, and verifying results

- **[Appendix D: Common Mistakes and Solutions](appendix-d-common-mistakes.md)** - Frequently encountered errors, why they occur, and how to fix them

---

### Back Matter

- **[Conclusion](conclusion.md)** - Summary of what you've mastered, how to extend your knowledge, and connections to production transformer models

---

## How to Use This Book

This book is designed for progressive learning. Follow this path for maximum understanding:

### 1. Read the Front Matter

- **Start with the [Preface](00a-preface.md)**: Understand what this book is, its pedagogical approach, learning path, prerequisites, and what you'll gain
- **Read the [Introduction](00c-introduction.md)**: Understand the philosophical framework—mathematics as operations, explicit transformations, and why intermediate states matter

### 2. Master the Foundations (Part I)

Read **Chapters 1-8 in order**. Each chapter builds on previous material:

- **Chapter 1** is the foundation—start here even if you have prior experience
- **Chapter 2** covers probability and statistics foundations needed for learning algorithms
- **Chapters 3-5** cover neural network fundamentals, learning algorithms, and training
- **Chapters 6-8** establish transformer-specific concepts

**Do not skip ahead.** The material is cumulative, and later chapters assume understanding of earlier concepts.

### 3. Work Through Examples (Part II)

Work through **Examples 1-7 sequentially**. Each example adds one new concept:

- **Example 1**: Forward pass only (no training)
- **Example 2**: Single training step
- **Example 3**: Complete backpropagation
- **Example 4**: Batch training
- **Example 5**: Non-linearity and depth
- **Example 6**: Complete transformer
- **Example 7**: Classification task

For each example:
1. **Read the chapter** to understand the concepts
2. **Work through the worksheet** to compute by hand
3. **Study the code** to see the implementation
4. **Run the code** to verify your calculations

### 4. Use the Appendices

Refer to appendices as needed:
- **Appendix A**: When you need matrix calculus formulas
- **Appendix B**: When terminology is unclear
- **Appendix C**: When doing hand calculations
- **Appendix D**: When debugging errors

### 5. Read the Conclusion

The conclusion summarizes what you've learned and provides guidance for extending your knowledge to production models.

---

## Key Features

- **Hand-Calculable**: Every example uses 2×2 matrices that you can compute by hand
- **Progressive**: Each chapter/example builds on previous material
- **Complete**: Theory, mathematics, code, and worksheets for every concept
- **Explicit**: All intermediate states shown, no compressed notation
- **Verifiable**: Every calculation can be checked manually

---

**Navigation**: Use the table of contents above to jump to any chapter. For best results, read sequentially from the beginning.
