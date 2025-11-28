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
---
**Navigation:**
- [← Index](00-index.md) | [← Previous: Appendix B](appendix-b-hand-calculation-tips.md) | [Next: Conclusion →](conclusion.md)
---
