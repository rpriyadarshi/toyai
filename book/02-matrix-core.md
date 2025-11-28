## Chapter 2: The Matrix Core

### Why Matrices?

Neural networks are fundamentally built on **matrix operations**. Every layer, every transformation, every computation involves matrices. But why?

**The answer:** Matrices are the mathematical tool that lets us:
1. Transform data efficiently
2. Learn patterns from examples
3. Compute gradients for training
4. Parallelize on GPUs/TPUs

### What is a Matrix?

A **matrix** is a rectangular array of numbers. For our 2×2 case:

$$A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$$

**Why 2×2?** Small enough to compute by hand, but captures all the essential operations.

### Matrix Multiplication: The Core Operation

**Matrix multiplication** is how neural networks transform data.

For matrices $A$ (2×2) and $B$ (2×2):

$$C = AB = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \begin{bmatrix} e & f \\ g & h \end{bmatrix} = \begin{bmatrix} ae+bg & af+bh \\ ce+dg & cf+dh \end{bmatrix}$$

**What this means:**
- Each element of $C$ is a **weighted combination** of elements from $A$ and $B$
- The weights come from the matrix structure itself
- This is how networks "mix" information

**Example:**
```
A = [1, 0]    B = [0.5, 0.5]
    [0, 1]        [0.5, 0.5]

C = A × B = [0.5, 0.5]
            [0.5, 0.5]
```

The identity matrix $A$ doesn't change $B$ - this is like a "pass-through" layer.

### Linear Transformations

**Matrix multiplication = linear transformation**

When we multiply a vector by a matrix, we:
- **Rotate** the vector in space
- **Scale** its components
- **Project** it to a new space

**Example:**
```
Vector: [1, 0]  (pointing in x-direction)
Matrix: [0, -1]  (rotation matrix)
        [1,  0]

Result: [0, 1]  (now pointing in y-direction - rotated 90°!)
```

**Why this matters for learning:**
- Different matrices = different transformations
- Learning = finding the right transformation
- Weights in matrices are what get updated during training

### Vector Spaces

**Vectors** are points in space. For 2D:
- $[1, 0]$ = point at (1, 0)
- $[0, 1]$ = point at (0, 1)
- $[0.5, 0.5]$ = point at (0.5, 0.5)

**Vector space** = all possible points/vectors

**Why this matters:**
- Embeddings live in vector spaces
- Attention computes similarity in vector space
- Learning = moving points in space to create patterns

### Dot Products: Measuring Similarity

**Dot product** of two vectors measures how "aligned" they are:

$$\mathbf{a} \cdot \mathbf{b} = a_1 b_1 + a_2 b_2$$

**Properties:**
- **High dot product** = vectors point in similar direction = similar
- **Low dot product** = vectors point in different directions = different
- **Zero dot product** = vectors are perpendicular = unrelated

**Example:**
```
[1, 0] · [1, 0] = 1×1 + 0×0 = 1    (same direction)
[1, 0] · [0, 1] = 1×0 + 0×1 = 0    (perpendicular)
[1, 0] · [-1, 0] = 1×(-1) + 0×0 = -1  (opposite direction)
```

**In attention:** Dot product between Query and Key measures how relevant they are!

### Transpose: Changing Perspective

**Transpose** swaps rows and columns:

$$A^T = \begin{bmatrix} a & c \\ b & d \end{bmatrix}$$

**Why transpose?**
- Matrix multiplication requires compatible dimensions
- $A \times B$ works if $A$ has $n$ columns and $B$ has $n$ rows
- Transpose lets us align dimensions: $A \times B^T$

**In attention:** We compute $Q \times K^T$ to get all pairwise dot products at once!

### Why Matrices Enable Learning

**1. Expressiveness:**
- Linear transformations can represent any linear relationship
- With non-linearities (ReLU), can approximate any function
- Multiple layers = composition of transformations = complex patterns

**2. Gradient Flow:**
- Matrix operations have clean derivatives
- Chain rule works beautifully: $\frac{d}{dW}(f(g(x))) = \frac{df}{dg} \frac{dg}{dW}$
- Enables backpropagation

**3. Parallelization:**
- GPUs have "tensor cores" optimized for matrix multiply
- Can process thousands of operations simultaneously
- Makes training feasible

**4. Composition:**
- Stack matrices: $f(g(x))$ where $f$ and $g$ are matrix operations
- Each layer adds complexity
- Deep networks = many composed transformations

### Matrix Calculus Basics

**For $C = AB$:**
- $\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} B^T$
- $\frac{\partial L}{\partial B} = A^T \frac{\partial L}{\partial C}$

**Why this matters:**
- Backpropagation needs these rules
- Gradients flow backward through matrix operations
- Enables training

### Learning Objectives Recap

- ✓ Understand why matrices are fundamental
- ✓ Master matrix multiplication
- ✓ See how linear transformations work
- ✓ Understand gradient flow through matrices

### Key Concepts Recap

- **Matrix multiplication**: Core operation for transformations
- **Linear transformations**: How matrices change vectors
- **Vector spaces**: Where embeddings and computations live
- **Transpose**: Tool for dimension alignment
- **Matrices enable learning**: Expressiveness + gradients + parallelization

---
---
**Navigation:**
- [← Index](00-index.md) | [← Previous: Why Transformers?](01-why-transformers.md) | [Next: Embeddings →](03-embeddings.md)
---
