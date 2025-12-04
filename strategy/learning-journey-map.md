# Learning Journey Map: Book Chapters → Examples → LinkedIn Posts

Complete mapping of your learning path from book chapters through examples to LinkedIn content.

## Overview

This map shows how each book chapter and example translates to LinkedIn posts. Use this to plan your content and ensure everything stays in sync.

## Part I: Foundations (Weeks 1-6)

### Week 1-2: Neural Network Fundamentals + Matrix Core

**Book Chapters**:
- [Chapter 1: Neural Network Fundamentals](../book/01-neural-network-fundamentals.md)
- [Chapter 2: The Matrix Core](../book/02-matrix-core.md)

**Key Concepts**:
- Perceptrons, neural networks, activation functions
- Matrix operations, multiplication, transformations
- Why matrices are fundamental to AI

**LinkedIn Posts** (6 posts total):

1. **Post 1 (Monday)**: "I'm an EDA engineer learning AI from scratch. Here's why I'm starting with 2x2 matrices."
   - Type: Reflection/Journey
   - Hook: Credibility + Journey
   - Content: Why 2x2 matrices, EDA background, learning approach
   - Links: Chapter 1, Chapter 2
   - Visual: Matrix diagram from book/images/

2. **Post 2 (Wednesday)**: "Matrix multiplication is the foundation of AI. Here's a hand calculation."
   - Type: Hand Calculation
   - Hook: Simple + Proof
   - Content: Work through 2x2 matrix multiplication by hand
   - Links: Chapter 2, Matrix code
   - Visual: Hand calculation photo, matrix multiplication diagram

3. **Post 3 (Friday)**: "I built a neural network in C++. No frameworks, just math."
   - Type: Code Walkthrough
   - Hook: Contrarian + Proof
   - Content: Show Matrix class implementation, basic operations
   - Links: src/core/Matrix.hpp, Chapter 2
   - Visual: Code snippet, output

4. **Post 4 (Monday)**: "Why matrices? Here's the math that makes AI possible."
   - Type: Math Behind X
   - Hook: Question + Answer
   - Content: Explain why matrices are fundamental, show transformations
   - Links: Chapter 2
   - Visual: Vector rotation diagram, transformation examples

5. **Post 5 (Wednesday)**: "From EDA signal processing to AI matrix operations: Same math, different domain."
   - Type: EDA to AI
   - Hook: Comparison
   - Content: Connect EDA concepts to matrix operations
   - Links: Chapter 2
   - Visual: Comparison diagram

6. **Post 6 (Friday)**: "Week 2 reflection: What I learned about the foundation of AI."
   - Type: Reflection
   - Hook: Journey
   - Content: Key insights from weeks 1-2
   - Links: Chapters 1-2
   - Visual: Progress summary

---

### Week 3-4: Embeddings & Attention Intuition

**Book Chapters**:
- [Chapter 3: Embeddings: Tokens to Vectors](../book/03-embeddings.md)
- [Chapter 4: Attention Intuition](../book/04-attention-intuition.md)

**Key Concepts**:
- Token embeddings, semantic spaces
- Query/Key/Value, attention mechanism, dot product

**LinkedIn Posts** (6 posts total):

7. **Post 7 (Monday)**: "Embeddings: Converting words to vectors, worked out by hand."
   - Type: Hand Calculation
   - Hook: Simple + Proof
   - Content: Show embedding calculation with 2x2 example
   - Links: Chapter 3, src/core/Embedding.hpp
   - Visual: Embedding space diagram, hand calculation

8. **Post 8 (Wednesday)**: "Attention mechanism in 28 lines of C++."
   - Type: I Built X in Y Lines
   - Hook: Proof + Comparison
   - Content: Show Attention class implementation
   - Links: src/core/Attention.hpp, Chapter 4
   - Visual: Code snippet, attention diagram

9. **Post 9 (Friday)**: "Why Q, K, V? Here's the intuition with a simple example."
   - Type: Math Behind X
   - Hook: Question + Answer
   - Content: Explain Q/K/V with 2x2 matrices
   - Links: Chapter 4
   - Visual: Q/K/V diagram, attention example diagram

10. **Post 10 (Monday)**: "I calculated attention scores by hand. Here's every step."
    - Type: Hand Calculation
    - Hook: Proof
    - Content: Work through attention calculation step-by-step
    - Links: Chapter 4, worksheets/example1_worksheet.md
    - Visual: Hand calculation photo, attention flow diagram

11. **Post 11 (Wednesday)**: "Dot product attention: The math that makes transformers work."
    - Type: Math Behind X
    - Hook: Simple
    - Content: Explain dot product, show with 2x2 example
    - Links: Chapter 4
    - Visual: Dot product visualization

12. **Post 12 (Friday)**: "From EDA routing to AI attention: Same concept, different domain."
    - Type: EDA to AI
    - Hook: Comparison
    - Content: Connect EDA signal routing to attention mechanism
    - Links: Chapter 4
    - Visual: Comparison diagram

---

### Week 5-6: Why Transformers

**Book Chapter**:
- [Chapter 5: Why Transformers?](../book/05-why-transformers.md)

**Key Concepts**:
- Sequence modeling, long-range dependencies
- RNN limitations, transformer advantages

**LinkedIn Posts** (6 posts total):

13. **Post 13 (Monday)**: "RNNs vs Transformers: Why I'm building transformers in C++."
    - Type: Comparison
    - Hook: Contrarian
    - Content: Compare RNN and transformer approaches
    - Links: Chapter 5
    - Visual: RNN limitation diagram, transformer architecture

14. **Post 14 (Wednesday)**: "The problem transformers solve (with hand calculations)."
    - Type: Math Behind X
    - Hook: Simple + Proof
    - Content: Show long-range dependency problem, transformer solution
    - Links: Chapter 5
    - Visual: Sequence modeling diagram

15. **Post 15 (Friday)**: "From EDA signal routing to AI attention: Same concept, different domain."
    - Type: EDA to AI
    - Hook: Comparison
    - Content: Deep dive on EDA/AI connection
    - Links: Chapter 5
    - Visual: Comparison diagram

16. **Post 16 (Monday)**: "Why parallelization matters: RNNs vs Transformers."
    - Type: Math Behind X
    - Hook: Question + Answer
    - Content: Explain parallelization advantage
    - Links: Chapter 5
    - Visual: Parallel processing diagram

17. **Post 17 (Wednesday)**: "I implemented the core transformer operation. Here's the code."
    - Type: Code Walkthrough
    - Hook: Proof
    - Content: Show attention implementation
    - Links: src/core/Attention.cpp, Chapter 5
    - Visual: Code snippet

18. **Post 18 (Friday)**: "6 weeks in: What I've learned about transformers so far."
    - Type: Reflection
    - Hook: Journey
    - Content: Mid-journey reflection
    - Links: Chapters 1-5
    - Visual: Progress summary

---

## Part II: Progressive Examples (Weeks 7-12)

### Week 7-8: Forward Pass (Example 1)

**Example**:
- [Example 1: Minimal Forward Pass](../examples/example1_forward_only/)
- [Book Chapter 6: Example 1](../book/06-example1-forward-pass.md)
- [Worksheet 1](../worksheets/example1_worksheet.md)

**Key Concepts**:
- Complete forward pass, embeddings → Q/K/V → attention → output

**LinkedIn Posts** (6 posts total):

19. **Post 19 (Monday)**: "I calculated a transformer forward pass by hand. Here's every step."
    - Type: Hand Calculation
    - Hook: Proof
    - Content: Work through complete forward pass
    - Links: Example 1, Worksheet 1, Chapter 6
    - Visual: Hand calculation photo, forward pass flow diagram

20. **Post 20 (Wednesday)**: "Building transformers in C++: Forward pass implementation."
    - Type: Code Walkthrough
    - Hook: Contrarian + Proof
    - Content: Show complete forward pass code
    - Links: examples/example1_forward_only/main.cpp, Example 1
    - Visual: Code snippet, architecture diagram

21. **Post 21 (Friday)**: "Why I verify every number with pen and paper."
    - Type: Reflection
    - Hook: Credibility
    - Content: Explain verification approach, benefits
    - Links: Example 1, Worksheet 1
    - Visual: Before/after comparison

22. **Post 22 (Monday)**: "Token embeddings to probabilities: The complete flow."
    - Type: Math Behind X
    - Hook: Simple
    - Content: Show end-to-end flow with calculations
    - Links: Example 1, Chapter 6
    - Visual: Complete flow diagram

23. **Post 23 (Wednesday)**: "I built a transformer forward pass in C++. No frameworks needed."
    - Type: I Built X in Y Lines
    - Hook: Contrarian + Proof
    - Content: Show key code sections
    - Links: Example 1
    - Visual: Code snippet

24. **Post 24 (Friday)**: "From theory to implementation: What I learned building Example 1."
    - Type: Reflection
    - Hook: Journey
    - Content: Key insights from first example
    - Links: Example 1
    - Visual: Progress summary

---

### Week 9-10: Training (Examples 2-3)

**Examples**:
- [Example 2: Single Training Step](../examples/example2_single_step/)
- [Example 3: Full Backpropagation](../examples/example3_full_backprop/)
- [Book Chapters 7-8](../book/07-example2-single-step.md), [Chapter 8](../book/08-example3-full-backprop.md)
- [Worksheets 2-3](../worksheets/example2_worksheet.md), [Worksheet 3](../worksheets/example3_worksheet.md)

**Key Concepts**:
- Loss functions, gradients, backpropagation, weight updates

**LinkedIn Posts** (6 posts total):

25. **Post 25 (Monday)**: "How transformers learn: One training step, calculated by hand."
    - Type: Hand Calculation
    - Hook: Proof
    - Content: Work through one training step
    - Links: Example 2, Worksheet 2, Chapter 7
    - Visual: Hand calculation, training loop diagram

26. **Post 26 (Wednesday)**: "Backpropagation through a transformer: The math, the code."
    - Type: Math Behind X + Code
    - Hook: Simple + Proof
    - Content: Show backpropagation math and implementation
    - Links: Example 3, Worksheet 3, Chapter 8
    - Visual: Backward pass flow diagram, code snippet

27. **Post 27 (Friday)**: "Gradient descent: From formula to C++ implementation."
    - Type: Code Walkthrough
    - Hook: Proof
    - Content: Show optimizer implementation
    - Links: src/core/Optimizer.cpp, Example 2
    - Visual: Gradient descent path diagram, code

28. **Post 28 (Monday)**: "I calculated gradients by hand. Here's how backpropagation works."
    - Type: Hand Calculation
    - Hook: Proof
    - Content: Work through gradient calculation
    - Links: Example 3, Worksheet 3
    - Visual: Hand calculation, gradient visualization

29. **Post 29 (Wednesday)**: "Loss functions: Measuring how wrong the model is."
    - Type: Math Behind X
    - Hook: Simple
    - Content: Explain cross-entropy loss with example
    - Links: src/core/Loss.cpp, Example 2
    - Visual: Loss diagram

30. **Post 30 (Friday)**: "Training a transformer: What I learned from Examples 2-3."
    - Type: Reflection
    - Hook: Journey
    - Content: Insights from training examples
    - Links: Examples 2-3
    - Visual: Progress summary

---

### Week 11-12: Complete Transformer (Examples 4-6)

**Examples**:
- [Example 4: Multiple Patterns](../examples/example4_multiple_patterns/)
- [Example 5: Feed-Forward Layers](../examples/example5_feedforward/)
- [Example 6: Complete Transformer](../examples/example6_complete/)
- [Book Chapters 9-11](../book/09-example4-multiple-patterns.md), [Chapter 10](../book/10-example5-feedforward.md), [Chapter 11](../book/11-example6-complete.md)
- [Worksheets 4-6](../worksheets/example4_worksheet.md), [Worksheet 5](../worksheets/example5_worksheet.md), [Worksheet 6](../worksheets/example6_worksheet.md)

**Key Concepts**:
- Batch training, feed-forward networks, complete architecture

**LinkedIn Posts** (6 posts total):

31. **Post 31 (Monday)**: "I built a complete transformer in C++. Here's the architecture."
    - Type: Code Walkthrough
    - Hook: Proof
    - Content: Show complete transformer implementation
    - Links: Example 6, Chapter 11
    - Visual: Complete transformer architecture diagram, code

32. **Post 32 (Wednesday)**: "Feed-forward networks: Adding non-linearity to transformers."
    - Type: Math Behind X
    - Hook: Simple
    - Content: Explain FFN with calculations
    - Links: Example 5, Chapter 10
    - Visual: FFN structure diagram

33. **Post 33 (Friday)**: "From 2x2 matrices to understanding GPT: The journey."
    - Type: Reflection
    - Hook: Journey
    - Content: Complete journey reflection
    - Links: All examples, all chapters
    - Visual: Journey summary

34. **Post 34 (Monday)**: "Batch training: Learning from multiple examples at once."
    - Type: Math Behind X
    - Hook: Simple
    - Content: Explain batch training with example
    - Links: Example 4, Chapter 9
    - Visual: Batch training diagram

35. **Post 35 (Wednesday)**: "Complete transformer: Every component explained."
    - Type: Math Behind X
    - Hook: Simple
    - Content: Walk through complete architecture
    - Links: Example 6, Chapter 11
    - Visual: Complete architecture diagram

36. **Post 36 (Friday)**: "What I learned building transformers from scratch (12-week reflection)."
    - Type: Reflection
    - Hook: Journey
    - Content: Complete 12-week reflection
    - Links: All content
    - Visual: Complete journey summary

---

## Quick Reference: Post Types by Day

**Monday**: Deep technical (Hand Calculation, Math Behind X)
**Wednesday**: Code walkthrough (I Built X in Y Lines, Code Walkthrough)
**Friday**: Reflection (Journey, EDA to AI, Reflection)

---

## Integration Checklist

For each post, ensure:
- [ ] Links to relevant book chapter
- [ ] Links to relevant example code
- [ ] Links to relevant worksheet
- [ ] References specific line numbers in code
- [ ] Uses diagrams from book/images/
- [ ] Includes hand calculation or code snippet
- [ ] Has clear CTA
- [ ] Uses appropriate hashtags
- [ ] Follows template from post-templates.md
- [ ] Uses hook from viral-hooks.md

---

## Content Sync Points

**Repository Structure**:
- Book chapters: `book/01-*.md` through `book/11-*.md`
- Examples: `examples/example1_*/` through `examples/example6_*/`
- Worksheets: `worksheets/example1_worksheet.md` through `example6_worksheet.md`
- Code: `src/core/*.hpp` and `src/core/*.cpp`
- Diagrams: `book/images/*.svg`

**Update Process**:
1. Study chapter → Do worksheet → Implement code
2. Create post using template
3. Link to all relevant resources
4. Update content calendar with actual date
5. Track engagement metrics

