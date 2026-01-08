# Introduction

**Navigation:**
- [← Previous: Preface](00a-preface.md) | [Table of Contents](00b-toc.md) | [Next: Chapter 1 →](01-neural-networks-perceptron.md)

---

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

**Navigation:**
- [← Previous: Preface](00a-preface.md) | [Table of Contents](00b-toc.md) | [Next: Chapter 1 →](01-neural-networks-perceptron.md)
