# Introduction

## On Mathematics and Computation

This book treats mathematics as a system of operations on symbols, not as abstract philosophy or aesthetic expression. This section establishes the operational framework that guides every presentation in this book.

### What Mathematics Is

Mathematics, as used here, consists of three components:

1. **Values**, represented as symbols (numbers, vectors, matrices, functions)
2. **Rules**, which define how symbols may be transformed (addition, multiplication, differentiation, matrix operations)
3. **Equivalence**, where different sequences of valid transformations produce the same result

A mathematical expression describes how one set of symbols transforms into another. There is nothing mystical about this process. Mathematics does not "reveal universal truths" or function as a "language of the universe." It is a practical, mechanical system for manipulating symbols consistently.

### What Theorem Proving Is

A theorem is not an act of insight or elegance. It is a demonstration that two or more independent sequences of valid transformations, starting from the same assumptions, arrive at exactly the same set of symbols. That convergence—not brevity, not beauty, not philosophical depth—is what establishes correctness.

Consider a simple example: proving that $2 + 3 = 5$. One transformation path might be:
- Start: $2 + 3$
- Apply definition of addition: $5$
- Result: $5$

Another path might be:
- Start: $2 + 3$
- Apply commutativity: $3 + 2$
- Apply definition: $5$
- Result: $5$

Both paths converge to the same symbol: $5$. That convergence is the proof. The theorem is verified not by elegance, but by the mechanical fact that all valid transformation sequences produce the same result.

### The Problem with Compressed Notation

A persistent failure in mathematics education is the systematic hiding of intermediate states. Multiple transformations are routinely compressed into single expressions, making equations appear sophisticated while preventing the reader from seeing how values actually change at each step.

Consider the expression $f(g(h(x)))$. This notation demands that the reader:
1. Execute $h(x)$ mentally
2. Visualize its output
3. Feed it into $g$
4. Visualize again
5. Feed into $f$
6. Infer the final meaning

This is mental stack execution without a debugger. No engineer would accept this in code, yet it is normalized in mathematical presentation.

The same computation, written with explicit intermediate states, becomes:
- $y_1 = h(x)$
- $y_2 = g(y_1)$
- $y_3 = f(y_2)$

Now the system is observable, debuggable, teachable, and verifiable. Nothing about this makes it "less mathematical." It makes it honest.

### Mathematical Maturity and Prior Knowledge

When mathematics is presented in compressed form, it often assumes that readers have already seen similar transformation patterns before. This assumption is sometimes called "mathematical maturity"—the ability to recognize and apply common transformation patterns without seeing them worked out explicitly.

Operationally, mathematical maturity means having seen and memorized common transformation patterns. When you encounter compressed notation like $f(g(h(x)))$, you can understand it because you've seen similar patterns before and know how to mentally execute the transformations. This works well if you've had extensive prior exposure to mathematical notation and proofs.

However, this creates a barrier for learners who haven't seen these patterns before. Compressed notation becomes difficult not because the mathematics itself is inherently complex, but because it assumes prior exposure to similar patterns. If you haven't seen how these transformations work before, the compressed form provides no way to learn them.

This book removes that barrier by showing all transformation patterns explicitly, step by step. You don't need to have seen these patterns before—we'll show them to you. Every transformation is demonstrated with intermediate states, so you can learn the patterns as you go rather than needing to recognize them from prior experience.

Understanding comes from explicit execution and verification, which anyone can do. You don't need mathematical maturity to use this book; you need only the willingness to follow step-by-step instructions and compute by hand. The book is designed to teach you the transformation patterns explicitly, not to assume you already know them.

### Why Intermediate States Matter

Humans cannot intuit long chains of transformations without inspecting intermediate results. This is a cognitive limitation, not an intellectual one. Understanding comes from observing how values evolve step by step, not from reading compressed expressions.

Any system—mathematical or computational—that cannot be inspected during execution cannot be reliably understood, debugged, or trusted. This principle is fundamental to engineering and computer science, yet it is routinely violated in mathematical presentation.

### The Approach of This Book

This book follows a strict operational principle:

**Every transformation must be explicit and observable.**

This means:
- All derivations are broken into individual steps
- Intermediate values are shown explicitly
- When possible, values are computed by hand
- The same steps are implemented in code without collapsing or reordering transformations
- No transformation is used if it cannot be shown step by step
- No equation is used if it cannot be executed
- No result is trusted if it cannot be inspected

This approach aligns mathematics with engineering and computer science practice, where systems are understood through execution, inspection, and verification—not through compressed notation or symbolic performance.

The goal is not to impress, but to make systems understandable, reproducible, and correct.

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
- Neural networks and their components
- Matrix operations and transformations
- Embeddings and vector spaces
- Attention mechanisms
- Why transformers solve problems that previous architectures could not

**Part II: Progressive Examples** builds from simple to complex:
- Example 1: Minimal forward pass (no training)
- Example 2: Single training step
- Example 3: Full backpropagation
- Example 4: Multiple training patterns
- Example 5: Feed-forward layers
- Example 6: Complete transformer

Each example includes:
- Step-by-step mathematical derivations with intermediate values shown
- Hand-calculation worksheets for verification
- Working C++ code that mirrors the mathematics exactly
- Clear connections between theory and implementation

### How to Use This Book

1. **Read the foundations first** (Chapters 1-5) to understand core concepts. Do not skip ahead—each chapter builds on previous material.

2. **Work through examples sequentially** (Examples 1-6). Each example builds on the previous one. Attempting later examples without completing earlier ones will leave gaps in understanding.

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
- [Next: Table of Contents →](00-index.md)
