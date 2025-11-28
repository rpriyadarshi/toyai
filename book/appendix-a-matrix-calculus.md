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
---
**Navigation:**
- [← Index](00-index.md) | [← Previous: Example 6: Complete](10-example6-complete.md) | [Next: Appendix B →](appendix-b-hand-calculation-tips.md)
---
