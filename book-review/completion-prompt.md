# Book Completion Prompt: Understanding Transformers
## Comprehensive Guidance for Finishing the Book

**Purpose:** This document serves as a comprehensive prompt for completing "Understanding Transformers: From First Principles to Mastery." Use this as your guide when expanding chapters, examples, and supporting materials.

**Last Updated:** January 26, 2026

---

## Core Principles (ALWAYS MAINTAIN)

When writing or expanding any section, adhere to these principles:

1. **Explicit Transformations**: Show every step, never compress operations
2. **Hand Verifiability**: All calculations must be doable with 2×2 matrices
3. **Progressive Complexity**: Build from simple to complex, never assume prior knowledge
4. **Observability**: Show all intermediate values explicitly
5. **Physical Analogies**: Use concrete analogies to explain abstract concepts
6. **No Mathematical Maturity Assumed**: Teach transformation patterns explicitly

---

## Chapter Expansion Template

### For Theory Chapters (Chapters 6-7)

**Structure:**
1. **Introduction** (1-2 pages)
   - What is this concept?
   - Why does it matter?
   - How does it connect to previous chapters?
   - Learning objectives

2. **Intuitive Explanation** (2-3 pages)
   - Plain-language explanation
   - Physical analogies
   - Visual diagrams
   - Concrete examples

3. **Mathematical Foundations** (3-5 pages)
   - Formal definitions
   - Step-by-step derivations
   - All intermediate states shown
   - Worked example with 2×2 matrices

4. **Core Mechanism** (4-6 pages)
   - How it works step-by-step
   - Explicit matrix operations
   - All intermediate values
   - Multiple worked examples

5. **Connections** (2-3 pages)
   - How this builds on previous concepts
   - How this enables later concepts
   - Cross-references to examples
   - Real-world applications

6. **Worked Examples** (4-6 pages)
   - At least 2-3 detailed examples
   - Use 2×2 matrices
   - Show every intermediate step
   - Include verification

7. **Learning Objectives Recap** (1 page)
   - What readers should understand
   - Key takeaways
   - Self-assessment questions

**Target Length:** 15-25K words  
**Key Requirement:** Every mathematical operation must be shown with explicit intermediate values

---

## Example Expansion Template

### For Example Chapters (Examples 1-6)

**Structure:**
1. **Goal and Learning Objectives** (1-2 pages)
   - What will readers learn?
   - What concepts are demonstrated?
   - Prerequisites from previous examples

2. **Task Description** (2-3 pages)
   - Detailed problem statement
   - Input/output specification
   - Success criteria
   - Why this example matters

3. **Model Architecture** (3-4 pages)
   - What components are used?
   - How do they connect?
   - Dimensions and shapes
   - Diagram with clear labels
   - Connection to theory chapters

4. **Step-by-Step Computation** (8-15 pages) - **CRITICAL SECTION**
   
   For EACH operation, include:
   - **Operation Name**: What operation?
   - **Purpose**: Why this operation?
   - **Input Values**: Show explicitly (matrices, vectors, scalars)
   - **Intermediate Steps**: Show ALL intermediate values
   - **Output Values**: Show explicitly
   - **Dimension Check**: Verify dimensions match
   - **Theory Connection**: Reference relevant theory chapter
   - **Verification**: How to check this step
   
   Format:
   ```
   ### Step N: [Operation Name]
   
   **Purpose:** [Why we do this]
   
   **Input:**
   - [Component 1]: [explicit values]
   - [Component 2]: [explicit values]
   
   **Computation:**
   [Show step-by-step with intermediate values]
   
   **Intermediate Values:**
   - [Value 1]: [explicit number]
   - [Value 2]: [explicit number]
   
   **Output:**
   - [Result]: [explicit values]
   
   **Dimension Check:**
   - Input: [dimensions]
   - Output: [dimensions]
   - ✓ Dimensions match
   
   **Theory Connection:** See [Chapter X] for details on [concept]
   
   **Verification:** [How to check]
   ```

5. **Hand Calculation Guide** (3-4 pages)
   - Reference to worksheet
   - Tips for manual computation
   - Common calculation errors
   - Verification strategies
   - Troubleshooting

6. **Code Walkthrough** (4-5 pages)
   - How code implements mathematics
   - Key implementation details
   - How to run
   - Expected output
   - How to verify calculations match code
   - Common code pitfalls

7. **Discussion** (2-3 pages)
   - What we learned
   - How this builds on previous examples
   - What's different/new
   - What's next
   - Key insights

8. **Practice Problems** (2-3 pages)
   - Exercises for readers
   - Varying difficulty
   - Solutions reference (can be separate file)

**Target Length:** 8-12K words  
**Key Requirement:** Every mathematical step must show ALL intermediate values explicitly

---

## Specific Chapter Prompts

### Chapter 6: Embeddings - Expansion Prompt

**Current State:** 5.3K words, good concepts but insufficient depth  
**Target:** 15-20K words with rigorous mathematical treatment

**Sections to Write:**

#### Section 1: Introduction (Expand to 2 pages)
- Problem: Discrete tokens vs. continuous neural networks
- Solution: Embeddings bridge the gap
- Historical context: word2vec, GloVe, modern approaches
- Connection to previous chapters: Why embeddings are needed for neural networks
- Learning objectives: What readers will understand

#### Section 2: The Discrete-Continuous Problem (NEW - 3 pages)
- Why tokens are discrete
- Why neural networks need continuous values
- The fundamental mismatch
- Worked example: Why "cat" + "dog" doesn't work
- Solution preview: Embeddings

#### Section 3: Mathematical Foundations (NEW - 4 pages)
- Formal definition: Embedding as function E: V → R^d
- Embedding space properties
- Vector space structure
- Distance metrics: Euclidean, cosine similarity
- Worked example: 2D embedding space with 4 tokens

#### Section 4: One-Hot Encoding (Expand to 3 pages)
- Definition and construction
- Mathematical representation
- Worked example: 4-token vocabulary with explicit vectors
- Limitations: Why orthogonal vectors don't help learning
- Explicit calculation: Why one-hot fails for similarity

#### Section 5: Learned Embeddings (NEW - 6 pages)
- How embeddings are learned during training
- Initialization strategies: random, pretrained, etc.
- Update mechanism: Connection to backpropagation
- How embeddings change during training
- Worked example: Learning embeddings for vocabulary {A, B, C, D}
  - Initial random embeddings
  - Training step 1: Update based on context
  - Training step 2: Further updates
  - Final embeddings: What they learned
- Why learned embeddings capture semantics

#### Section 6: Embedding Dimensions (Expand to 3 pages)
- How to choose dimension d
- Trade-offs: Too small vs. too large
- Capacity vs. efficiency
- Examples from real models: BERT (768), GPT-3 (12,288)
- Our choice: d=2 for hand calculation
- Why 2D is sufficient for learning concepts

#### Section 7: Semantic Spaces (Expand to 4 pages)
- What semantic spaces are
- How meaning emerges in embedding space
- Worked example: 2D semantic space
  - Place tokens A, B, C, D in 2D space
  - Show how similar tokens cluster
  - Show how relationships emerge
- Visualization techniques
- Connection to attention: How embeddings enable attention to find relevant tokens
- Worked example: How attention uses embedding similarity

#### Section 8: Embedding Lookup Operations (NEW - 3 pages)
- Step-by-step lookup procedure
- Mathematical operation: E[i] for token index i
- Matrix operations involved
- Worked example: Looking up embeddings for sequence "A B"
  - Token "A" → index 0 → E[0] = [explicit vector]
  - Token "B" → index 1 → E[1] = [explicit vector]
  - Show all intermediate steps
- Efficiency considerations
- Batch lookup: How to handle sequences

#### Section 9: Connection to Attention Mechanism (NEW - 4 pages)
- How embeddings feed into Q/K/V projections
- Why good embeddings enable good attention
- Worked example: From embeddings to attention
  - Start with token embeddings
  - Project to Q, K, V
  - Show how embedding quality affects attention
- The embedding-attention feedback loop
- Why embeddings and attention learn together

#### Section 10: Real-World Context (NEW - 3 pages)
- Word2vec and similar models
- BERT/GPT embedding strategies
- How principles scale to production
- What changes with scale, what stays the same
- Modern embedding techniques

#### Section 11: Learning Objectives Recap (Expand to 2 pages)
- Detailed checklist of understanding
- Self-assessment questions
- Key takeaways
- What to review if unclear

**Key Requirements:**
- At least 4 worked examples with full calculations
- All matrix operations shown explicitly
- All intermediate values displayed
- Clear connection to attention mechanism
- Real-world context and scaling discussion

---

### Chapter 7: Attention Intuition - Expansion Prompt

**Current State:** 7.0K words, good intuition but lacks mathematical rigor  
**Target:** 20-25K words with complete mathematical treatment

**Sections to Write:**

#### Section 1: Introduction (Expand to 2 pages)
- The core question: "Which positions are relevant?"
- Why attention is needed
- Historical context: Bahdanau attention, self-attention, transformers
- Connection to previous chapters: Embeddings enable attention
- Learning objectives

#### Section 2: The Attention Problem (NEW - 4 pages)
- Formal problem statement
- What information each position needs
- How to determine relevance
- Worked example: "The cat sat on the mat"
  - Position "sat" needs: subject ("cat")
  - Position "mat" needs: verb ("sat")
  - How to find these relationships
- The challenge: How to learn relevance

#### Section 3: Query/Key/Value Deep Dive (NEW - 8 pages)
- Mathematical definition of Q, K, V
- How they're computed from embeddings
- Explicit matrix operations:
  - Q = XW_Q (show step-by-step)
  - K = XW_K (show step-by-step)
  - V = XW_V (show step-by-step)
- Worked example: Computing Q/K/V from 2×2 embeddings
  - Input: Embeddings for tokens A, B
  - Weight matrices W_Q, W_K, W_V (2×2 each)
  - Compute Q, K, V explicitly
  - Show all intermediate values
- Why three separate components
- What each component learns
- Physical analogy: Library search system (expand)

#### Section 4: Attention Score Computation (NEW - 5 pages)
- Dot product as similarity measure
- Mathematical definition: score = Q · K^T
- Step-by-step score calculation
- Worked example: Computing scores for 2-token sequence
  - Q matrix (2×2)
  - K matrix (2×2)
  - Compute QK^T explicitly
  - Show all intermediate values
- Scaling factor: Why √d_k
- Mathematical justification for scaling
- Worked example: With and without scaling

#### Section 5: Softmax and Attention Weights (Expand to 4 pages)
- Why softmax: Probabilistic interpretation
- Step-by-step softmax computation
- Worked example: Converting scores to weights
  - Raw scores: [explicit values]
  - Subtract max: [explicit values]
  - Exponentiate: [explicit values]
  - Normalize: [explicit values]
  - Final weights: [explicit values]
- Properties of attention weights: Sum to 1, non-negative
- Interpretation: What weights mean

#### Section 6: Context Vector Computation (NEW - 4 pages)
- Weighted sum of values
- Mathematical definition: C = Σ(attention_weight_i × V_i)
- Step-by-step computation
- Worked example: Full attention computation
  - Attention weights: [explicit values]
  - Value matrix V: [explicit values]
  - Compute weighted sum explicitly
  - Show all intermediate values
- Interpretation of context vector
- What information it contains

#### Section 7: Complete Attention Mechanism (NEW - 5 pages)
- End-to-end computation
- Full worked example with 2×2 matrices
  - Input: Token sequence "A B"
  - Step 1: Embeddings
  - Step 2: Q/K/V projections
  - Step 3: Attention scores
  - Step 4: Attention weights (softmax)
  - Step 5: Context vector
  - All intermediate values shown
- Verification: Check all dimensions, verify softmax sums to 1
- Connection to theory: How this implements attention

#### Section 8: Multi-Head Attention (NEW - 4 pages)
- Concept explanation
- Why multiple heads: Different types of relationships
- Simplified 2-head example
  - Head 1: Syntactic relationships
  - Head 2: Semantic relationships
  - How heads are combined
- Worked example: 2-head attention with 2×2 matrices
- Connection to single-head attention

#### Section 9: Attention Patterns (NEW - 3 pages)
- What patterns attention learns
- Examples: Causal, bidirectional, local, global
- Visualization techniques
- How to interpret attention weights
- Worked example: Interpreting attention patterns

#### Section 10: Connection to Embeddings (Expand to 3 pages)
- How embeddings enable attention
- Why good embeddings matter
- Worked example: Showing the connection
  - Start with embeddings
  - Show how they feed into Q/K/V
  - Show how embedding quality affects attention
- The embedding-attention learning loop

#### Section 11: Real-World Context (NEW - 3 pages)
- Attention in GPT, BERT, etc.
- How principles scale
- Production considerations
- What changes with scale, what stays the same

#### Section 12: Learning Objectives Recap (Expand to 2 pages)
- Detailed checklist
- Self-assessment questions
- Key takeaways

**Key Requirements:**
- At least 5 worked examples with full calculations
- All matrix operations shown explicitly with intermediate values
- Complete end-to-end attention computation
- Clear connection to embeddings
- Multi-head attention introduction

---

### Examples 1-6: Expansion Prompts

**General Template Applied to Each Example:**

#### Example N: [Title] - Expansion Prompt

**Current State:** [X]K words  
**Target:** 8-12K words with complete step-by-step calculations

**Critical Section: Step-by-Step Computation**

For EACH operation in the example, write:

```
### Step [N]: [Operation Name]

**Purpose:** [Why we perform this operation]

**Input Values:**
- [Component 1]: 
  ```
  [Explicit matrix/vector values]
  ```
- [Component 2]:
  ```
  [Explicit matrix/vector values]
  ```

**Computation:**

[Show the mathematical operation step-by-step]

**Intermediate Values:**
- [Intermediate 1]: [explicit number]
- [Intermediate 2]: [explicit number]
- [Intermediate 3]: [explicit number]

**Output Values:**
- [Result]:
  ```
  [Explicit matrix/vector values]
  ```

**Dimension Verification:**
- Input dimensions: [m×n]
- Output dimensions: [p×q]
- ✓ Dimensions are compatible

**Theory Connection:** 
This operation implements [concept] discussed in [Chapter X, Section Y]. 
Specifically, [explain connection].

**Verification:**
- Check: [specific verification step]
- Expected: [expected result]
- Actual: [actual result]
- ✓ Verification passed

**Common Pitfalls:**
- [Pitfall 1]: [How to avoid]
- [Pitfall 2]: [How to avoid]
```

**Example-Specific Requirements:**

**Example 1: Forward Pass**
- Show embeddings lookup explicitly
- Show Q/K/V projection with all intermediate values
- Show attention score computation step-by-step
- Show softmax computation with all intermediate values
- Show context vector computation
- Show output projection
- Show final softmax for predictions

**Example 2: Single Training Step**
- Show forward pass (can reference Example 1)
- Show loss computation explicitly
- Show gradient computation step-by-step
- Show weight update explicitly
- Show how one step changes predictions

**Example 3: Full Backpropagation**
- Show complete forward pass
- Show loss computation
- Show gradient flow through each component
- Show all intermediate gradients
- Show weight updates for all parameters
- Show how gradients flow backward

**Example 4: Multiple Patterns**
- Show batch processing explicitly
- Show gradient accumulation
- Show how multiple examples affect learning
- Show convergence over multiple steps

**Example 5: Feed-Forward Layers**
- Show feed-forward computation explicitly
- Show ReLU activation step-by-step
- Show residual connections
- Show how non-linearity helps

**Example 6: Complete Transformer**
- Show all components working together
- Show layer normalization explicitly
- Show multiple transformer blocks
- Show end-to-end computation
- Show complete training loop

---

## Appendix Expansion Prompts

### Appendix C: Hand Calculation Tips - Expansion Prompt

**Current State:** 741 bytes (minimal)  
**Target:** 5-8K words (comprehensive guide)

**Sections to Write:**

#### Section 1: Introduction (1 page)
- Why hand calculation matters
- Benefits of manual computation
- How to use this appendix
- When to use hand calculation vs. code

#### Section 2: Organization Strategies (2 pages)
- Setting up calculations
  - Workspace layout
  - Notation conventions
  - Labeling intermediate values
- Keeping track of values
  - Tables for intermediate results
  - Clear labeling
  - Systematic approach
- Workspace management
  - How to organize calculations
  - When to start fresh
  - How to check work

#### Section 3: Matrix Operations (3 pages)
- Matrix multiplication step-by-step
  - Row × column method
  - Worked example: 2×2 matrices
  - Common patterns
- Dimension checking
  - How to verify compatibility
  - Common dimension errors
  - How to catch mistakes early
- Error detection techniques
  - Symmetry checks
  - Range checks
  - Sanity checks
- Worked examples with verification

#### Section 4: Vector Operations (2 pages)
- Dot products
  - Step-by-step computation
  - Worked examples
- Vector addition/subtraction
  - Component-wise operations
  - Worked examples
- Norms and distances
  - Euclidean norm
  - When to use
  - Worked examples

#### Section 5: Softmax and Normalization (2 pages)
- Computing softmax by hand
  - Step-by-step procedure
  - Numerical stability tricks
    - Subtract max before exponentiating
    - Why this works
  - Worked example
- Verification methods
  - Check probabilities sum to 1
  - Check all values non-negative
  - Check reasonable values

#### Section 6: Gradient Computation (3 pages)
- Chain rule application
  - Systematic approach
  - How to break down complex functions
  - Worked example
- Common errors and fixes
  - Sign errors
  - Missing terms
  - Dimension errors
- Worked examples with verification

#### Section 7: Verification Strategies (2 pages)
- How to check your work
  - Recompute key steps
  - Check dimensions
  - Verify properties (e.g., softmax sums to 1)
- Red flags to watch for
  - Impossible values
  - Dimension mismatches
  - Sign errors
- Comparison with code output
  - How to compare
  - When discrepancies are OK
  - When to recompute

#### Section 8: Efficiency Tips (1 page)
- Time-saving techniques
- When to use approximations
- Calculator usage
- When to use code instead

---

### Appendix D: Common Mistakes and Solutions - Expansion Prompt

**Current State:** 932 bytes (minimal)  
**Target:** 5-8K words (comprehensive troubleshooting guide)

**Sections to Write:**

#### Section 1: Introduction (1 page)
- Why mistakes happen
- Learning from mistakes
- How to use this appendix
- Systematic debugging approach

#### Section 2: Mathematical Errors (3 pages)

**Dimension Mismatches:**
- Common causes
- How to identify
- Solutions with worked examples
- Prevention strategies

**Sign Errors:**
- Where they occur (gradients, updates)
- How to identify
- Solutions with worked examples
- Prevention strategies

**Index Errors:**
- Common causes
- How to identify
- Solutions with worked examples
- Prevention strategies

**Order of Operations:**
- Common mistakes
- How to identify
- Solutions with worked examples
- Prevention strategies

#### Section 3: Conceptual Errors (3 pages)

**Misunderstanding Attention:**
- Common misconceptions
- Clarifications with examples
- How to correct understanding
- Worked examples

**Confusing Q/K/V Roles:**
- Common confusions
- Clear explanations
- How to remember
- Worked examples

**Gradient Flow Misconceptions:**
- Common mistakes
- Correct understanding
- How to trace gradients
- Worked examples

**Embedding Misunderstandings:**
- Common confusions
- Clarifications
- How to understand embeddings
- Worked examples

#### Section 4: Implementation Errors (2 pages)

**Code Bugs:**
- Common bugs
- How to identify
- Fixes with code examples
- Prevention strategies

**Numerical Issues:**
- Overflow/underflow
- Precision problems
- Solutions
- Prevention strategies

**Off-by-One Errors:**
- Common causes
- How to identify
- Fixes
- Prevention strategies

#### Section 5: Pedagogical Pitfalls (2 pages)

**Common Learning Mistakes:**
- Skipping steps
- Not verifying
- Rushing through examples
- Solutions

**How to Avoid Them:**
- Best practices
- Study strategies
- When to slow down

**Recovery Strategies:**
- How to recover from mistakes
- When to ask for help
- How to learn from errors

#### Section 6: Troubleshooting Guide (2 pages)

**Systematic Debugging:**
- Step-by-step approach
- Where to start
- How to narrow down

**Common Symptoms and Causes:**
- Symptom: [description]
  - Possible causes: [list]
  - How to diagnose: [steps]
  - Solution: [fix]

**Step-by-Step Resolution:**
- General procedure
- Specific examples
- When to give up and start fresh

---

## Conclusion Expansion Prompt

**Current State:** 830 bytes (too brief)  
**Target:** 3-5K words (comprehensive synthesis)

**Sections to Write:**

#### Section 1: Synthesis: The Complete Picture (2 pages)
- How all concepts connect
  - From perceptron to transformer
  - How each chapter builds
  - The integrated system
- The transformer as unified architecture
  - How components work together
  - Why the design works
- Key insights and principles
  - Explicit transformations
  - Hand verifiability
  - Progressive complexity
  - Mechanistic understanding
- What makes transformers powerful
  - Attention mechanism
  - Parallel processing
  - Scalability

#### Section 2: Learning Journey Reflection (1 page)
- What readers have accomplished
  - From basic concepts to complete transformer
  - Skills developed
  - Understanding achieved
- The progression
  - Started with: Single perceptron
  - Learned: Neural networks, probability, learning
  - Mastered: Attention, embeddings, transformers
  - Can now: Implement, debug, extend

#### Section 3: Core Principles Revisited (1 page)
- Explicit transformations: Why it matters
- Hand verifiability: Why it's valuable
- Progressive complexity: How it enabled learning
- Mechanistic understanding: What it enables

#### Section 4: Scaling to Production Models (2 pages)
- How principles apply to GPT, BERT, etc.
  - Same math, different scale
  - What stays the same
  - What changes
- Scaling considerations
  - Computational requirements
  - Memory requirements
  - Training strategies
- The math is identical
  - 2×2 matrices → 12,288 dimensions
  - Same operations, different scale
  - What you learned applies directly

#### Section 5: Next Steps (1 page)
- Recommended reading
  - Original transformer paper
  - GPT papers
  - BERT paper
  - Other resources
- Projects to try
  - Implement larger models
  - Experiment with architectures
  - Build applications
- Advanced topics to explore
  - Multi-head attention details
  - Positional encodings
  - Different architectures
  - Optimization techniques

#### Section 6: Final Thoughts (1 page)
- Encouragement
  - You've mastered transformers from first principles
  - This understanding is deep and valuable
- Philosophy reminder
  - Explicit transformations enable understanding
  - Hand verifiability builds intuition
  - Mechanistic understanding enables engineering
- Call to action
  - Continue learning
  - Build projects
  - Share knowledge
- Continuing the journey
  - This is just the beginning
  - Apply what you've learned
  - Keep exploring

---

## Quality Checklist (Use for Every Section)

Before considering any section complete, verify:

### Content Quality
- [ ] All mathematical operations shown step-by-step
- [ ] All intermediate values explicitly displayed
- [ ] All calculations verifiable by hand (2×2 matrices)
- [ ] Consistent notation throughout
- [ ] Physical analogies where appropriate
- [ ] Clear connections between concepts

### Completeness
- [ ] Section meets target length
- [ ] All planned subsections included
- [ ] Worked examples provided (minimum 2-3)
- [ ] Cross-references added
- [ ] Learning objectives addressed

### Accuracy
- [ ] All mathematical derivations correct
- [ ] All calculations verified
- [ ] All cross-references accurate
- [ ] All examples work correctly
- [ ] Code examples match mathematics

### Pedagogical Quality
- [ ] Concepts build progressively
- [ ] Prerequisites clearly stated
- [ ] Learning objectives clear
- [ ] Examples illustrate concepts
- [ ] Practice opportunities provided (where appropriate)

### Style Consistency
- [ ] Matches existing book style
- [ ] Uses same notation conventions
- [ ] Follows same structure patterns
- [ ] Maintains same tone

### Textbook Publication Style
- [ ] No undefined terms; every technical term either defined in-section or given a forward ref (see *Section Title*)
- [ ] Cross-references readable in print (section/chapter name in text, not only a link)
- [ ] Same cross-reference pattern used throughout (e.g. "see *Section Title*" with link on section title)
- [ ] Forward refs are short and natural (no long parentheticals)

---

## Writing Guidelines

### Mathematical Notation
- Use LaTeX for all mathematical expressions
- Show all intermediate steps
- Use explicit values, not just symbols
- Verify dimensions at each step
- Include verification checkpoints

### Examples
- Always use 2×2 matrices for hand calculation
- Show all intermediate values
- Include verification steps
- Connect to theory chapters
- Provide multiple examples per concept

### Cross-References
- Reference relevant theory chapters from examples
- Reference relevant examples from theory chapters
- Use consistent reference format: `[Chapter X]` or `[Example Y]`
- Include section references when specific
- Cross-references must be **print-readable** (section/chapter name visible in text)
- Use **section titles** for within-chapter refs (e.g. *Entropy: Measuring Uncertainty*), not only "Chapter X"
- When a term is used **before** its definition, add a forward reference to the section where it is defined

### Diagrams
- Use mermaid diagrams for flow
- Reference SVG diagrams where available
- Ensure diagrams match text descriptions
- Include captions and explanations

### Code References
- Reference code examples from text
- Explain how code implements mathematics
- Show expected output
- Include verification steps

### Textbook Publication Style (Strict)

Follow these rules so the book reads like a published textbook in both print and digital:

- **Define-before-use:** Do not use a technical term before it is defined. If a term must appear earlier (e.g. in an intro), add a forward reference: "see *Section Title* below" or "we define X in *Section Title*."
- **Forward references:** Use the **section title** (e.g. *Information Theory Foundations*, *Cross-Entropy: Comparing Distributions*) so the sentence reads correctly in **print** without a hyperlink. In digital, link the section title to the correct heading. Use one short signpost per mention; avoid long parentheticals.
- **Cross-reference format:** Same pattern everywhere. Within chapter: "term (see *Section Title*)" with *Section Title* linked. Between chapters: "[Chapter N: Title](filename.md)" or "see [Chapter N: Title](filename.md)." No bare "see Section 2.5" if the book does not use section numbers; use section/chapter names.
- **Flow:** Where helpful, add a one-sentence chapter roadmap after the opening blurb. In "Connection to…" paragraphs, prefer one forward pointer at the end (e.g. "…see *Cross-Entropy: Comparing Distributions*") rather than multiple inline links.
- **Consistency:** Same cross-reference pattern in every chapter; same terminology; same structure for similar sections.

---

## Progress Tracking

Use this template to track completion:

### Chapter/Example: [Name]
- [ ] Section 1: [Title] - [Status]
- [ ] Section 2: [Title] - [Status]
- [ ] Section 3: [Title] - [Status]
- [ ] ...
- [ ] Review and polish
- [ ] Cross-references added
- [ ] Quality checklist completed
- [ ] Ready for publication

**Status Codes:**
- Not Started
- In Progress
- Draft Complete
- Review Needed
- Complete

---

## Final Notes

1. **Maintain Quality**: Don't rush. Better to have fewer, high-quality sections than many incomplete ones.

2. **Stay Consistent**: Follow existing patterns. If Chapters 1-2 are excellent, use them as templates.

3. **Verify Everything**: Every calculation should be verified. Every cross-reference should be checked.

4. **Think Like a Reader**: What would confuse a reader? What needs more explanation? What's missing?

5. **Iterate**: First draft → Review → Expand → Polish → Final

6. **Get Feedback**: Have others read sections. Fresh eyes catch issues.

7. **Take Breaks**: This is detailed work. Step away and return with fresh perspective.

---

*This prompt document should be your primary guide when expanding any section of the book. Refer to it frequently to maintain consistency and quality.*
