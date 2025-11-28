## Chapter 3: Embeddings: Tokens to Vectors

### The Problem: Discrete vs. Continuous

**Tokens** (words, characters, subwords) are **discrete**:
- "cat" is just a symbol
- No mathematical relationship between "cat" and "dog"
- Can't do arithmetic: "cat" + "dog" = ???

**Neural networks** need **continuous** values:
- Matrix operations require numbers
- Gradients need smooth functions
- Learning needs measurable similarity

**Solution:** Convert discrete tokens to continuous vectors = **embeddings**

### What are Embeddings?

**Embeddings** map each token to a vector (point in space):

```
Token "cat" → Vector [0.3, 0.7, -0.2, ...]
Token "dog" → Vector [0.4, 0.6, -0.1, ...]
Token "mat" → Vector [-0.1, 0.2, 0.8, ...]
```

**Key insight:** Similar tokens should have similar vectors!

### One-Hot Encoding: The Starting Point

**One-hot encoding** is the simplest embedding:
- Vocabulary size = $V$
- Each token gets a vector of length $V$
- Only one element is 1, rest are 0

**Example (vocab: A, B, C, D):**
```
A → [1, 0, 0, 0]
B → [0, 1, 0, 0]
C → [0, 0, 1, 0]
D → [0, 0, 0, 1]
```

**Problems with one-hot:**
- Vectors are orthogonal (no similarity)
- Dimension = vocabulary size (huge for large vocabs!)
- No semantic relationships

### Learned Embeddings: The Solution

**Learned embeddings** are vectors that get updated during training:
- Start random
- Learn to capture semantic relationships
- Similar meanings → similar vectors

**Example (learned):**
```
"cat" → [0.3, 0.7, -0.2]
"dog" → [0.4, 0.6, -0.1]  (similar to "cat"!)
"mat" → [-0.1, 0.2, 0.8]  (different from "cat")
```

**How it works:**
- Embedding matrix: $E \in \mathbb{R}^{V \times d}$
- $V$ = vocabulary size
- $d$ = embedding dimension (e.g., 2 for our examples, 768 for BERT)
- Lookup: token $i$ → row $i$ of $E$

### Embedding Dimensions

**Dimension choice:**
- **Too small**: Can't capture enough information
- **Too large**: Overfitting, slow computation
- **Sweet spot**: Balance capacity and efficiency

**Our examples:** $d = 2$ (hand-calculable!)
**Real models:** $d = 768$ (BERT), $d = 12,288$ (GPT-3)

### Semantic Spaces

**Embeddings create a "semantic space":**
- Tokens with similar meanings are close together
- Tokens with different meanings are far apart
- Relationships emerge: "king" - "man" + "woman" ≈ "queen"

**Example in 2D:**
```
A = [1, 0]  (corner of space)
B = [0, 1]  (different corner)
C = [1, 1]  (combination)
D = [0, 0]  (origin)
```

### Fixed vs. Learned Embeddings

**Fixed embeddings** (our examples):
- Pre-defined, don't change
- Simple for learning
- Example: A=[1,0], B=[0,1]

**Learned embeddings** (real models):
- Updated during training
- Capture task-specific semantics
- Much more powerful

**In our examples:** We use fixed embeddings to focus on attention and training mechanics.

### Embedding Lookup

**Process:**
1. Token index: "cat" → index 42
2. Lookup: $E[42]$ → vector $[0.3, 0.7, -0.2, ...]$
3. Use vector in computations

**Mathematically:**
$$\text{embedding}(i) = E[i]$$

Where $E$ is the embedding matrix and $i$ is the token index.

### Why Embeddings Matter

**1. Enable computation:**
- Can't do math on "cat"
- Can do math on $[0.3, 0.7, -0.2]$

**2. Capture relationships:**
- Similar tokens → similar vectors
- Enables attention to find relevant tokens

**3. Learnable:**
- Embeddings adapt to task
- Better embeddings = better model

### Learning Objectives Recap

- ✓ Understand why embeddings are needed
- ✓ See how discrete tokens become vectors
- ✓ Understand embedding spaces and dimensions
- ✓ Know difference between fixed and learned embeddings

### Key Concepts Recap

- **Token vocabulary**: Set of all possible tokens
- **Embedding matrices**: Map tokens to vectors
- **Vector representations**: Continuous, learnable
- **Semantic spaces**: Where meaning lives

### Mathematical Foundations Recap

- **One-hot encoding**: Simple but limited
- **Embedding lookup**: $E[i]$ for token $i$
- **Embedding dimensions**: Balance capacity and efficiency

---
---
**Navigation:**
- [← Index](00-index.md) | [← Previous: The Matrix Core](02-matrix-core.md) | [Next: Attention Intuition →](04-attention-intuition.md)
---
