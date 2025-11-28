# Example 1: Minimal Forward Pass

## Goal

Understand how a transformer makes predictions **without any training**. This example focuses purely on the forward pass computation.

## What You'll Learn

- How tokens are converted to embeddings
- How Q, K, V vectors are created
- How attention computes context
- How context is mapped to token probabilities
- The complete forward pass flow

## Model Architecture

```
Input: [A, B]
  ↓
Embeddings: [1,0], [0,1]
  ↓
Q, K, V Projections (fixed weights)
  ↓
Attention: Creates context vector
  ↓
Output Projection: Context → Logits
  ↓
Softmax: Logits → Probabilities
```

## Key Concepts

### 1. Token Embeddings

Tokens (A, B, C, D) are converted to 2D vectors:
- A = [1, 0]
- B = [0, 1]
- C = [1, 1]
- D = [0, 0]

### 2. Q, K, V Projections

Each embedding is projected to three spaces:
- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I have to offer?"
- **Value (V)**: "What is my actual content?"

### 3. Attention

Computes how much each position should attend to others:
- Attention scores = Q × K^T / √d_k
- Attention weights = softmax(scores)
- Context = weighted sum of values

### 4. Output Projection

Maps the 2D context vector to 4 logits (one per token).

### 5. Softmax

Converts logits to probabilities that sum to 1.

## Running the Example

```bash
cd build
make example1
./examples/example1_forward_only/example1
```

## Hand Calculation

See `../../worksheets/example1_worksheet.md` for step-by-step hand calculation guide.

## Expected Output

The model will output probabilities for each token. Since no training has occurred, all tokens should have roughly equal probability (~25% each).

## Next Steps

- **Example 2**: Learn how one training step works
- **Example 3**: Understand full backpropagation
- **Example 4**: Train on multiple patterns
- **Example 5**: Add feed-forward layers
- **Example 6**: Complete transformer

