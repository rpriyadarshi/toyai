# Preface

**Navigation:**
- [Table of Contents](00b-toc.md) | [Next: Introduction →](00c-introduction.md)

---

## What This Book Is

This book is a progressive learning system for understanding transformers—the architecture behind modern language models like GPT, BERT, and others. It takes you from first principles to a complete implementation, with every step verifiable by hand.

### Scope and Purpose

Transformers have revolutionized natural language processing and are the foundation of modern generative AI. Understanding them deeply—not just knowing what they do, but understanding how and why they work—is essential for anyone serious about AI.

This book provides that understanding through a unique approach: every mathematical operation is broken down into explicit steps, every intermediate value is shown, and every computation can be verified manually.

### Pedagogical Method

Most AI education presents mathematics in compressed form, hiding intermediate states and making it difficult to understand what's actually happening. This book takes the opposite approach: every transformation is explicit, every intermediate value is shown, and every computation can be verified by hand.

We use 2×2 matrices throughout so that every calculation can be done manually. This constraint forces clarity and ensures that nothing is hidden behind large-scale computations. The same principles apply to larger systems—the difference is scale, not kind.

### Learning Path

The book is organized into two parts:

**Part I: Foundations** establishes the core concepts:
- Neural networks and their components (Chapter 1)
- Probability and statistics foundations (Chapter 2)
- Multilayer networks and architecture (Chapter 3)
- Learning algorithms: loss functions, gradient descent, backpropagation (Chapter 4)
- Training neural networks (Chapter 5)
- Embeddings and vector spaces (Chapter 6)
- Attention mechanisms (Chapter 7)
- Why transformers solve problems that previous architectures could not (Chapter 8)

**Part II: Progressive Examples** builds from simple to complex:
- Example 1: Minimal forward pass (no training)
- Example 2: Single training step
- Example 3: Full backpropagation
- Example 4: Multiple training patterns
- Example 5: Feed-forward layers
- Example 6: Complete transformer
- Example 7: Character recognition

Each example includes:
- Step-by-step mathematical derivations with intermediate values shown
- Hand-calculation worksheets for verification
- Working C++ code that mirrors the mathematics exactly
- Clear connections between theory and implementation

### How to Use This Book

1. **Read the foundations first** (Chapters 1-8) to understand core concepts. Do not skip ahead—each chapter builds on previous material.

2. **Work through examples sequentially** (Examples 1-7). Each example builds on the previous one. Attempting later examples without completing earlier ones will leave gaps in understanding.

3. **Use the worksheets** to verify calculations by hand. The worksheets are not optional—they are essential for building intuition and catching errors.

4. **Run the code** to see concepts in action. The code is designed to be readable and to mirror the mathematical operations exactly. Study it alongside the mathematics.

5. **Refer to appendices** as needed for quick reference on matrix calculus, terminology, calculation tips, and common mistakes.

### Prerequisites

- **Basic algebra**: Comfort with variables, equations, and basic operations
- **Basic matrix operations**: Understanding of matrix multiplication (we will review this, but prior familiarity helps)
- **Willingness to compute by hand**: We provide worksheets, but you must do the work
- **Ability to read C++ code**: The code is straightforward, but basic programming familiarity is assumed
- **No prior AI or machine learning experience required**: We build from first principles

You do not need "mathematical maturity" to use this book. All necessary transformation patterns are shown explicitly with intermediate steps, so you don't need to have seen them before. Prior exposure to compressed mathematical notation is not required—we teach you the patterns as you go. Understanding comes from explicit execution and verification, which requires only the willingness to compute by hand and follow step-by-step instructions. If you can do that, you can understand this material. The book is designed to teach you the transformation patterns explicitly, not to assume you already know them.

### What You Will Gain

By the end of this book, you will:
- Understand transformers at a deep, mechanistic level
- Be able to trace every computation step by step
- Verify any calculation by hand
- Implement transformer components from scratch
- Debug transformer implementations effectively
- Understand why transformers work, not just what they do

This understanding is not superficial. It is the kind of understanding that comes from seeing every step, computing every value, and verifying every result. It is the understanding that enables real engineering work.

---

**Navigation:**
- [Table of Contents](00b-toc.md) | [Next: Introduction →](00c-introduction.md)