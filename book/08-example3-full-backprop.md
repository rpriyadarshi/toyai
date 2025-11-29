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

This example demonstrates complete backpropagation through all components. For the complete transformer architecture, see [Chapter 1: Neural Network Fundamentals](01-neural-network-fundamentals.md) - "Complete Transformer Architecture".

**Components:**
- **All projection matrices are trainable**: $W_Q$, $W_K$, $W_V$, $W_O$
- Complete gradient flow through attention mechanism
- Full backward pass from loss to all weights

**Model Architecture Diagram:**

```mermaid
graph LR
    Input["Input<br/>'A B'"] --> Forward["Forward Pass"]
    Forward --> Loss["Loss<br/>L"]
    Target["Target<br/>'C'"] --> Loss
    Loss --> Backward["Backward Pass<br/>Complete Gradient Flow"]
    Backward --> GradWO["∂L/∂WO<br/>Update WO"]
    Backward --> GradWQ["∂L/∂WQ<br/>Update WQ"]
    Backward --> GradWK["∂L/∂WK<br/>Update WK"]
    Backward --> GradWV["∂L/∂WV<br/>Update WV"]
    GradWO --> Update["Weight Updates<br/>All matrices"]
    GradWQ --> Update
    GradWK --> Update
    GradWV --> Update
    
    style Loss fill:#ffcdd2
    style Backward fill:#fff4e1
    style GradWO fill:#e8f5e9
    style GradWQ fill:#e8f5e9
    style GradWK fill:#e8f5e9
    style GradWV fill:#e8f5e9
    style Update fill:#c8e6c9
```

**Key Difference from Example 2:**
- Example 2: Only $W_O$ trainable (simple gradient flow)
- Example 3: All weights trainable (complete gradient flow through attention)

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

### Backpropagation Flow

```mermaid
graph TD
    Loss["Loss<br/>L"] --> GradLogits["∂L/∂logits"]
    GradLogits --> GradWO["∂L/∂WO<br/>Update WO"]
    GradLogits --> GradContext["∂L/∂context"]
    GradContext --> GradWeights["∂L/∂weights<br/>(attention)"]
    GradContext --> GradV["∂L/∂V"]
    GradV --> GradWV["∂L/∂WV<br/>Update WV"]
    GradWeights --> GradScores["∂L/∂scores<br/>(softmax backward)"]
    GradScores --> GradQ["∂L/∂Q"]
    GradScores --> GradK["∂L/∂K"]
    GradQ --> GradWQ["∂L/∂WQ<br/>Update WQ"]
    GradK --> GradWK["∂L/∂WK<br/>Update WK"]
    
    style Loss fill:#ffcdd2
    style GradWO fill:#e8f5e9
    style GradWV fill:#e8f5e9
    style GradWQ fill:#e8f5e9
    style GradWK fill:#e8f5e9
```

### Gradient Computation Flow

```mermaid
graph LR
    Forward["Forward Pass<br/>Compute all values"] --> Loss["Loss<br/>L"]
    Loss --> Backward["Backward Pass<br/>Compute gradients"]
    Backward --> Chain["Chain Rule<br/>∂L/∂x = ∂L/∂y × ∂y/∂x"]
    Chain --> GradWO["Gradient WO"]
    Chain --> GradWQ["Gradient WQ"]
    Chain --> GradWK["Gradient WK"]
    Chain --> GradWV["Gradient WV"]
    GradWO --> Update["Update All<br/>W_new = W_old - η×grad"]
    GradWQ --> Update
    GradWK --> Update
    GradWV --> Update
    
    style Forward fill:#e1f5ff
    style Loss fill:#ffcdd2
    style Backward fill:#fff4e1
    style Update fill:#e8f5e9
```

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

See [worksheet](../worksheets/example3_worksheet.md)

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

See [code](../examples/example3_full_backprop/main.cpp)

### Exercises

1. Trace complete gradient flow by hand
2. Compute all weight gradients
3. Verify gradient magnitudes make sense
4. Perform full training step
5. Compare to Example 2 (only $W_O$ trained)

---
---
**Navigation:**
- [← Index](00-index.md) | [← Previous: Example 2: Single Step](07-example2-single-step.md) | [Next: Example 4: Multiple Patterns →](09-example4-multiple-patterns.md)
---
