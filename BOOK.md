# Understanding Transformers: From First Principles to Mastery

**A Progressive Learning System with Hand-Calculable Examples**

---

## Table of Contents

### Part I: Foundations

1. [Why Transformers?](#chapter-1-why-transformers)
2. [The Matrix Core](#chapter-2-the-matrix-core)
3. [Embeddings: Tokens to Vectors](#chapter-3-embeddings-tokens-to-vectors)
4. [Attention Intuition](#chapter-4-attention-intuition)

### Part II: Progressive Examples

5. [Example 1: Minimal Forward Pass](#example-1-minimal-forward-pass)
6. [Example 2: Single Training Step](#example-2-single-training-step)
7. [Example 3: Full Backpropagation](#example-3-full-backpropagation)
8. [Example 4: Multiple Patterns](#example-4-multiple-patterns)
9. [Example 5: Feed-Forward Layers](#example-5-feed-forward-layers)
10. [Example 6: Complete Transformer](#example-6-complete-transformer)

### Appendices

- [A: Matrix Calculus Reference](#appendix-a-matrix-calculus-reference)
- [B: Hand Calculation Tips](#appendix-b-hand-calculation-tips)
- [C: Common Mistakes and Solutions](#appendix-c-common-mistakes-and-solutions)

---

## Chapter 1: Why Transformers?

### The Problem: Sequence Modeling

Imagine you're reading a sentence: "The cat sat on the mat." To understand this, you need to:
- Remember that "cat" is the subject
- Connect "sat" to "cat" (the cat did the sitting)
- Understand "on the mat" describes where the cat sat

This is **sequence modeling**: understanding how elements in a sequence relate to each other.

**Real-world applications:**
- **Language translation**: "Hello" → "Hola" (but context matters!)
- **Text generation**: Given "The weather is", predict "nice" or "terrible"
- **Question answering**: "Who wrote Hamlet?" requires understanding context
- **Code completion**: IDE suggests next token based on previous code

### The Challenge: Long-Range Dependencies

In the sentence "The cat that I saw yesterday sat on the mat", the word "sat" must connect to "cat" even though many words separate them. This is a **long-range dependency**.

**Why this is hard:**
- Information must flow across many positions
- Context from early in the sequence affects later predictions
- Traditional models struggle with this

### Previous Solutions: RNNs and Their Limitations

**Recurrent Neural Networks (RNNs)** were the previous solution:
- Process sequence one token at a time
- Maintain hidden state that carries information forward
- Can theoretically handle long sequences

**But RNNs have problems:**
1. **Sequential bottleneck**: Must process tokens one-by-one (can't parallelize)
2. **Vanishing gradients**: Information gets lost over long sequences
3. **Forgetting**: Early context fades as sequence gets longer

**Example of RNN limitation:**
```
Input: "The cat that I saw yesterday in the park near my house sat on the mat"
RNN state: [cat] → [saw] → [yesterday] → [park] → [house] → [sat] → [mat]
                    ↑                                    ↑
              "cat" info                            "cat" info is weak!
```

By the time we reach "sat", the RNN has forgotten much about "cat".

### The Transformer Solution: Attention

**Transformers solve this with attention:**
- Every position can directly attend to every other position
- No sequential bottleneck - all positions processed in parallel
- Information flows directly where needed

**Key insight:** Instead of forcing information through a sequential chain, let each position "look" at all other positions and decide what's relevant.

**Example with attention:**
```
Position "sat" can directly attend to:
- "cat" (high attention - subject)
- "mat" (high attention - object)
- "yesterday" (medium attention - time context)
- "the" (low attention - not very informative)
```

### Why Attention is Powerful

1. **Direct connections**: No information loss through sequential processing
2. **Parallel computation**: All positions computed simultaneously (fast on GPUs)
3. **Interpretable**: Can see what the model is "paying attention to"
4. **Scalable**: Works well with very long sequences

### Real-World Impact

Transformers power:
- **GPT models**: ChatGPT, GPT-4 (text generation)
- **BERT**: Google search, language understanding
- **Code models**: GitHub Copilot, Codex
- **Translation**: Google Translate
- **Image models**: Vision transformers (ViT)

### The Core Innovation

The transformer's innovation isn't a single breakthrough, but a combination:
1. **Self-attention**: Each position attends to all positions
2. **Parallel processing**: No sequential dependency
3. **Scaled dot-product**: Efficient attention computation
4. **Stacked layers**: Multiple attention layers for complex patterns

### Learning Objectives Recap

- ✓ Understand sequence modeling challenges
- ✓ See why RNNs struggle with long-range dependencies
- ✓ Understand how attention solves these problems
- ✓ Connect to real-world transformer applications

### Key Concepts Recap

- **Sequence-to-sequence tasks**: Input sequence → output sequence
- **Long-range dependencies**: Connections across many positions
- **Parallel computation**: All positions processed simultaneously
- **Attention mechanism**: Direct connections between positions

---

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

## Chapter 4: Attention Intuition

### The Core Question

When processing a sequence, each position needs to ask:
> **"Which other positions contain information relevant to me?"**

**Example:** In "The cat sat on the mat"
- Position "sat" needs to know about "cat" (subject)
- Position "mat" needs to know about "sat" (verb)
- Position "the" (first) is less relevant to "mat"

**Attention** is the mechanism that answers this question.

### The Query/Key/Value Metaphor

Think of attention like a **library search system**:

#### Query (Q): "What am I looking for?"

**Query** represents what information a position needs:
- "sat" needs: "What is the subject?"
- "mat" needs: "What verb describes location?"

**In vectors:** Query is a learned representation of "what I'm searching for"

#### Key (K): "What do I have to offer?"

**Key** represents what information each position contains:
- "cat" offers: "I am a noun, I am the subject"
- "sat" offers: "I am a verb, I describe action"
- "the" offers: "I am an article, I'm not very informative"

**In vectors:** Key is a learned representation of "what I advertise"

#### Value (V): "What is my actual content?"

**Value** is the actual information to retrieve:
- Once we decide "cat" is relevant, we retrieve its value
- Value contains the semantic content we actually use

**In vectors:** Value is what gets weighted and combined

### The Search Process

**Step 1: Match Query to Keys**
```
Query "sat": "I need the subject"
Key "cat": "I am a noun, I am the subject"  ← Match!
Key "the": "I am an article"                ← No match
```

**Step 2: Compute Relevance Scores**
- High score = Query matches Key = relevant
- Low score = Query doesn't match Key = not relevant

**Step 3: Convert to Probabilities (Softmax)**
- Scores → probabilities (attention weights)
- Sum to 1.0 (probability distribution)

**Step 4: Retrieve Values**
- Weighted sum of values
- High attention weight → more contribution from that value

### Search Engine Analogy

**Google Search:**
1. **Query**: Your search terms ("transformer attention")
2. **Keys**: Keywords on web pages
3. **Relevance**: How well keywords match query
4. **Values**: Actual webpage content
5. **Result**: Weighted combination of relevant pages

**Transformer Attention:**
1. **Query**: What position needs
2. **Keys**: What each position offers
3. **Relevance**: Dot product (similarity)
4. **Values**: Actual content to retrieve
5. **Output**: Weighted combination of values

**Same idea, different domain!**

### Database Query Analogy

**SQL Query:**
```sql
SELECT content FROM pages 
WHERE keywords MATCH "transformer attention"
ORDER BY relevance DESC
```

**Attention:**
```
SELECT values FROM positions
WHERE keys MATCH query
WEIGHT BY attention_scores
```

**Attention is like a learned, differentiable database query!**

### Why Three Components (Q, K, V)?

**Why not just one?** Because they serve different purposes:

1. **Q and K determine relevance** (what to attend to)
2. **V provides content** (what to retrieve)

**Separation allows:**
- Learning what to search for (Q)
- Learning what to advertise (K)
- Learning what content to provide (V)

**Example:** A position might:
- Search for "subject" (Q)
- Advertise "I'm a noun" (K)
- Provide "cat" meaning (V)

All three are learned separately!

### Attention Weights as Probabilities

**Attention weights** are probabilities:
- Each position gets a weight
- Weights sum to 1.0
- Higher weight = more attention

**Example:**
```
Position "sat" attending to:
- "cat": 0.6  (60% attention - subject!)
- "the": 0.1  (10% attention - not very relevant)
- "on": 0.2   (20% attention - preposition)
- "mat": 0.1  (10% attention - object)
Sum: 1.0 ✓
```

**Interpretation:** "sat" pays 60% attention to "cat" because it's the subject.

### How Relevance is Computed

**Dot product** measures alignment:

$$\text{score} = Q \cdot K = \sum_i Q_i K_i$$

**Why dot product?**
- High when Q and K point in similar direction
- Low when they point in different directions
- Zero when perpendicular (unrelated)

**After dot product:**
1. Scale by $\sqrt{d_k}$ (prevents large values)
2. Apply softmax (convert to probabilities)
3. Use as weights for values

### Information Retrieval Perspective

**Classic IR:** Given query, find relevant documents

**Attention:** Given query vector, find relevant position vectors

**Both:**
- Compute similarity (dot product)
- Rank by relevance
- Retrieve and combine content

**Difference:** Attention is **learned** - the model discovers what "relevant" means!

### The Magic: Learned Relevance

**Fixed relevance** (like keyword matching):
- "cat" always matches "cat"
- Can't learn new relationships

**Learned relevance** (attention):
- Model learns what makes positions relevant
- "cat" might become relevant to "feline" even if they don't share words
- Adapts to task

**This is why transformers are powerful!**

### Learning Objectives Recap

- ✓ Understand Q/K/V metaphor
- ✓ See attention as search mechanism
- ✓ Understand relevance computation
- ✓ Connect to information retrieval

### Key Concepts Recap

- **Query**: "What am I looking for?"
- **Key**: "What do I have to offer?"
- **Value**: "What is my actual content?"
- **Attention weights**: Probabilities of relevance

### Intuitive Explanations Recap

- **Search engine**: Query matches keywords, retrieves pages
- **Database query**: SELECT WHERE MATCH, ORDER BY relevance
- **Information retrieval**: Find relevant content, combine it

---

## Example 1: Minimal Forward Pass

**Goal**: Understand how a transformer makes predictions (no training yet)

**What You'll Learn**:
- Forward pass computation
- Attention mechanism step-by-step
- How context is created
- How predictions are made

### The Task

Given input sequence "A B", predict the next token. We'll compute probabilities for each possible token (A, B, C, D) without any training - just to see how the forward pass works.

### Model Architecture

- Fixed token embeddings
- Fixed Q, K, V projection matrices
- Scaled dot-product attention
- Output projection to vocabulary
- Softmax to get probabilities

### Step-by-Step Computation

1. **Token Embeddings**: Convert "A" and "B" to 2D vectors
2. **Q/K/V Projections**: Create Query, Key, Value vectors
3. **Attention Scores**: Compute similarity between queries and keys
4. **Attention Weights**: Apply softmax to get probability distribution
5. **Context Vector**: Weighted sum of values
6. **Output Logits**: Project context to vocabulary space
7. **Probabilities**: Apply softmax to get final predictions

### Hand Calculation Guide

See `worksheets/example1_worksheet.md` for step-by-step template.

### Theory

#### Attention Formula

The scaled dot-product attention is:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- $Q$: Query matrix (what we're looking for)
- $K$: Key matrix (what information is available)
- $V$: Value matrix (the actual content)
- $d_k$: Dimension of keys (scaling factor)

#### Why Scaling?

Without the $\sqrt{d_k}$ scaling, dot products grow with dimension, causing:
- Softmax saturation (probabilities near 0 or 1)
- Vanishing gradients
- Numerical instability

Scaling keeps variance approximately constant.

#### Softmax Properties

For input vector $\mathbf{x} = [x_1, x_2, \ldots, x_n]$:

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}$$

Properties:
- All outputs are positive
- Outputs sum to 1 (probability distribution)
- Differentiable everywhere
- Preserves relative ordering

### Code Implementation

See `examples/example1_forward_only/main.cpp`

### Exercises

1. Compute attention scores by hand for given Q, K matrices
2. Verify softmax computation
3. Trace through complete forward pass
4. Compare hand calculation to code output

---

## Example 2: Single Training Step

**Goal**: Understand how one weight update works

**What You'll Learn**:
- Loss functions
- Gradient computation
- Gradient descent
- How models learn from examples

### The Task

Train the model on example: Input "A B" → Target "C"

We'll update only the output projection matrix $W_O$ in this example to keep it simple.

### Model Architecture

- Same as Example 1
- But $W_O$ is now trainable
- $W_Q$, $W_K$, $W_V$ remain fixed

### Training Process

1. **Forward Pass**: Compute prediction (same as Example 1)
2. **Compute Loss**: Measure how wrong the prediction is
3. **Compute Gradients**: Calculate how to change weights
4. **Update Weights**: Actually change $W_O$ using gradient descent

### Loss Function

Cross-entropy loss for next-token prediction:

$$L = -\log P(y_{\text{target}})$$

Where $P(y_{\text{target}})$ is the model's predicted probability for the correct token.

Properties:
- Lower when model is confident and correct
- Higher when model is wrong or uncertain
- Differentiable (enables gradient descent)

### Gradient Computation

For softmax + cross-entropy, the gradient w.r.t. logits is:

$$\frac{\partial L}{\partial \text{logit}_i} = P(i) - \mathbf{1}[i = \text{target}]$$

Where $\mathbf{1}[\cdot]$ is the indicator function.

This elegant formula means:
- If model predicts too high probability for wrong token → push logit down
- If model predicts too low probability for correct token → push logit up

### Gradient Descent Update

$$W_{\text{new}} = W_{\text{old}} - \eta \cdot \frac{\partial L}{\partial W}$$

Where $\eta$ is the learning rate.

### Hand Calculation Guide

See `worksheets/example2_worksheet.md`

### Theory

#### Chain Rule Basics

For composite function $f(g(x))$:

$$\frac{df}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}$$

In our case:
$$L \leftarrow \text{softmax}(\text{logits}) \leftarrow \text{logits} \leftarrow W_O \times \text{context}$$

We compute gradients backward through this chain.

#### Why Gradient Descent Works

Gradient points in direction of steepest increase. To minimize loss, we move opposite to gradient (hence the minus sign).

With small learning rate, we take small steps toward the minimum.

### Code Implementation

See `examples/example2_single_step/main.cpp`

### Exercises

1. Compute loss by hand
2. Compute gradient w.r.t. logits
3. Compute gradient w.r.t. $W_O$
4. Perform one weight update
5. Verify prediction improves

---

## Example 3: Full Backpropagation

**Goal**: Understand complete gradient flow through all components

**What You'll Learn**:
- Backpropagation through attention
- Matrix calculus
- Gradient flow through Q, K, V
- Complete training loop

### The Task

Train on "A B" → "C" with all weights trainable: $W_Q$, $W_K$, $W_V$, $W_O$

### Model Architecture

- All projection matrices are trainable
- Complete gradient flow through attention mechanism

### Backpropagation Steps

1. **Loss → Logits**: $\frac{\partial L}{\partial \text{logits}}$ (from Example 2)
2. **Logits → $W_O$**: $\frac{\partial L}{\partial W_O}$ (from Example 2)
3. **Logits → Context**: $\frac{\partial L}{\partial \text{context}}$
4. **Context → Attention Weights**: $\frac{\partial L}{\partial \text{weights}}$
5. **Attention Weights → Scores**: $\frac{\partial L}{\partial \text{scores}}$ (softmax backward)
6. **Scores → Q, K**: $\frac{\partial L}{\partial Q}$, $\frac{\partial L}{\partial K}$
7. **Q, K → $W_Q$, $W_K$**: $\frac{\partial L}{\partial W_Q}$, $\frac{\partial L}{\partial W_K}$
8. **Context → V**: $\frac{\partial L}{\partial V}$
9. **V → $W_V$**: $\frac{\partial L}{\partial W_V}$

### Matrix Calculus

#### Matrix Multiplication Gradient

For $C = AB$:
- $\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} B^T$
- $\frac{\partial L}{\partial B} = A^T \frac{\partial L}{\partial C}$

#### Attention Gradients

For $\text{Output} = \text{Weights} \times V$:
- $\frac{\partial L}{\partial \text{Weights}} = \frac{\partial L}{\partial \text{Output}} V^T$
- $\frac{\partial L}{\partial V} = \text{Weights}^T \frac{\partial L}{\partial \text{Output}}$

For $\text{Scores} = QK^T$:
- $\frac{\partial L}{\partial Q} = \frac{\partial L}{\partial \text{Scores}} K$
- $\frac{\partial L}{\partial K} = \frac{\partial L}{\partial \text{Scores}}^T Q$

### Softmax Jacobian

The softmax Jacobian is:

$$\frac{\partial \text{softmax}_i}{\partial x_j} = \text{softmax}_i \cdot (\delta_{ij} - \text{softmax}_j)$$

Where $\delta_{ij}$ is the Kronecker delta.

This means each output depends on all inputs (through the normalization).

### Hand Calculation Guide

See `worksheets/example3_worksheet.md`

### Theory

#### Why Backpropagation Works

Backpropagation is just the chain rule applied systematically:
1. Forward pass: compute all intermediate values
2. Backward pass: compute gradients starting from loss
3. Each operation has a known local gradient
4. Chain rule multiplies local gradients together

#### Computational Graph

The computation forms a directed acyclic graph (DAG):
- Nodes: operations (matmul, softmax, etc.)
- Edges: data flow
- Backprop: reverse the edges, multiply gradients

### Code Implementation

See `examples/example3_full_backprop/main.cpp`

### Exercises

1. Trace complete gradient flow by hand
2. Compute all weight gradients
3. Verify gradient magnitudes make sense
4. Perform full training step
5. Compare to Example 2 (only $W_O$ trained)

---

## Example 4: Multiple Patterns

**Goal**: Learn multiple patterns from multiple examples

**What You'll Learn**:
- Batch training
- Gradient accumulation
- Pattern learning
- Convergence

### The Task

Train on multiple examples:
- "A B" → "C"
- "A A" → "D"
- "B A" → "C"

Learn all patterns simultaneously.

### Model Architecture

- Same as Example 3
- But process multiple examples

### Batch Training

Instead of one example at a time:
1. Process all examples in batch
2. Compute loss for each
3. Average gradients
4. Update weights once per batch

### Gradient Averaging

For batch of size $N$:

$$\frac{\partial L}{\partial W} = \frac{1}{N} \sum_{i=1}^{N} \frac{\partial L_i}{\partial W}$$

This averages gradients across examples.

### Training Loop

```
For each epoch:
    For each batch:
        Forward pass (all examples)
        Compute losses
        Backward pass (all examples)
        Average gradients
        Update weights
```

### Hand Calculation Guide

See `worksheets/example4_worksheet.md`

### Theory

#### Why Batch Training?

- **Stability**: Averaging reduces noise in gradients
- **Efficiency**: Process multiple examples in parallel
- **Generalization**: Model sees diverse patterns together

#### Convergence

With proper learning rate:
- Loss decreases over epochs
- Model learns all patterns
- Gradients become smaller (convergence)

### Code Implementation

See `examples/example4_multiple_patterns/main.cpp`

### Exercises

1. Compute batch loss
2. Average gradients across examples
3. Train for multiple epochs
4. Verify all patterns are learned
5. Plot loss over time

---

## Example 5: Feed-Forward Layers

**Goal**: Add non-linearity and depth

**What You'll Learn**:
- Feed-forward networks
- Non-linear activations (ReLU)
- Residual connections
- Layer composition

### The Task

Add a feed-forward network after attention:
- Attention output → Feed-Forward → Final output

### Model Architecture

```
Input → Embeddings → Attention → Feed-Forward → Output
                              ↓
                         Residual connection
```

### Feed-Forward Network

Two linear transformations with ReLU:

$$\text{FFN}(x) = \text{ReLU}(x W_1 + b_1) W_2 + b_2$$

For our 2x2 case, we'll use:
- $W_1$: 2×2 matrix
- $W_2$: 2×2 matrix
- ReLU: element-wise max(0, x)

### ReLU Activation

$$\text{ReLU}(x) = \max(0, x)$$

Properties:
- Non-linear (enables learning complex functions)
- Simple derivative (0 or 1)
- Prevents negative activations

### Residual Connections

$$\text{Output} = x + \text{FFN}(x)$$

Why?
- Enables gradient flow through deep networks
- Allows identity mapping (if FFN learns nothing, output = input)
- Helps with training stability

### Hand Calculation Guide

See `worksheets/example5_worksheet.md`

### Theory

#### Universal Approximation

Feed-forward networks with non-linear activations can approximate any continuous function (universal approximation theorem).

This is why adding FFN increases model capacity.

#### Why Residuals?

Without residuals, gradients can vanish in deep networks. Residuals provide "highway" for gradients to flow directly.

### Code Implementation

See `examples/example5_feedforward/main.cpp`

### Exercises

1. Compute FFN forward pass
2. Compute ReLU gradients
3. Trace gradient through FFN
4. Verify residual connection helps
5. Compare with/without residuals

---

## Example 6: Complete Transformer

**Goal**: Full implementation with all components

**What You'll Learn**:
- Complete transformer architecture
- Layer normalization
- Multiple layers
- End-to-end training

### The Task

Build complete transformer with:
- Multiple transformer blocks
- Layer normalization
- Residual connections everywhere
- Complete training pipeline

### Model Architecture

```
Input → Embeddings
     → Transformer Block 1 (Attention + FFN + LayerNorm + Residuals)
     → Transformer Block 2 (Attention + FFN + LayerNorm + Residuals)
     → Output Projection
     → Softmax
     → Probabilities
```

### Layer Normalization

Normalize across features (not batch):

$$\text{LayerNorm}(x) = \gamma \frac{x - \mu}{\sigma} + \beta$$

Where:
- $\mu$: mean of features
- $\sigma$: standard deviation of features
- $\gamma, \beta$: learnable parameters

Why?
- Stabilizes training
- Reduces internal covariate shift
- Enables larger learning rates

### Multiple Layers

Stack transformer blocks:
- Each block processes the output of previous
- Deeper = more complex patterns
- Residuals enable deep networks

### Complete Training

Full pipeline:
1. Forward through all layers
2. Compute loss
3. Backprop through all layers
4. Update all weights
5. Repeat for many epochs

### Hand Calculation Guide

See `worksheets/example6_worksheet.md`

### Theory

#### Deep Networks

Each layer learns increasingly abstract features:
- Layer 1: Local patterns
- Layer 2: Combinations of Layer 1 patterns
- Layer 3: High-level concepts

#### Why This Architecture Works

- **Attention**: Captures long-range dependencies
- **FFN**: Adds non-linearity and capacity
- **Residuals**: Enables gradient flow
- **LayerNorm**: Stabilizes training
- **Multiple layers**: Learns hierarchical representations

### Code Implementation

See `examples/example6_complete/main.cpp`

### Exercises

1. Trace through complete forward pass
2. Compute all intermediate values
3. Perform full backpropagation
4. Train complete model
5. Analyze learned representations

---

## Appendix A: Matrix Calculus Reference

### Basic Rules

1. **Scalar derivative**: $\frac{d}{dx}(ax) = a$

2. **Product rule**: $\frac{d}{dx}(f(x)g(x)) = f'(x)g(x) + f(x)g'(x)$

3. **Chain rule**: $\frac{d}{dx}(f(g(x))) = f'(g(x)) \cdot g'(x)$

### Matrix Operations

#### Matrix Multiplication

For $C = AB$ where $A \in \mathbb{R}^{m \times n}$, $B \in \mathbb{R}^{n \times p}$:

$$C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}$$

#### Transpose

$$(A^T)_{ij} = A_{ji}$$

#### Gradient Rules

For $C = AB$:
- $\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} B^T$
- $\frac{\partial L}{\partial B} = A^T \frac{\partial L}{\partial C}$

For $C = A^T$:
- $\frac{\partial L}{\partial A} = \left(\frac{\partial L}{\partial C}\right)^T$

---

## Appendix B: Hand Calculation Tips

### Organization

1. Write down all initial values clearly
2. Show intermediate steps
3. Label each computation
4. Check dimensions match
5. Verify final results

### Common Patterns

- Matrix multiplication: row × column
- Dot product: sum of element-wise products
- Softmax: exp, sum, divide
- Gradients: chain rule systematically

### Verification

- Check that probabilities sum to 1
- Verify gradient signs make sense
- Compare to code output
- Recompute if results don't match

---

## Appendix C: Common Mistakes and Solutions

### Mistake 1: Forgetting Scaling Factor

**Problem**: Not dividing by $\sqrt{d_k}$ in attention

**Solution**: Always include scaling: $\frac{QK^T}{\sqrt{d_k}}$

### Mistake 2: Softmax Numerical Instability

**Problem**: Computing $e^x$ for large $x$ causes overflow

**Solution**: Subtract max before exponentiating: $e^{x_i - \max(x)}$

### Mistake 3: Wrong Gradient Sign

**Problem**: Adding gradient instead of subtracting

**Solution**: Remember: $W_{\text{new}} = W_{\text{old}} - \eta \nabla L$

### Mistake 4: Dimension Mismatch

**Problem**: Trying to multiply incompatible matrices

**Solution**: Always check dimensions: $(m \times n) \times (n \times p) = (m \times p)$

---

## Conclusion

You've now mastered transformers from first principles! You can:

- Understand every component
- Compute everything by hand
- Implement from scratch
- Extend to larger models

The principles you've learned apply to GPT, BERT, and all transformer-based models. The math is identical - only the scale changes.

---

**Next Steps**:
- Implement larger models
- Experiment with different architectures
- Read original papers with full understanding
- Build your own transformer applications

---

*This book is a living document. As you work through examples, refer back to relevant chapters. Each example builds on previous ones, creating a complete understanding.*

