## Chapter 4: Training Neural Networks

In the previous chapters, we've learned about network architecture (perceptrons, layers, feedforward networks) and learning algorithms (loss functions, gradient descent, backpropagation). Now we'll see how these components combine into a complete training system. This chapter covers the training loop, batch processing, epochs, and the transition from neural networks to transformers.

We'll also introduce key transformer concepts that build on the neural network fundamentals we've established. This chapter serves as a bridge between the neural network foundations and the transformer-specific topics that follow.

**Navigation:**
- [← Previous: Learning Algorithms](03-learning-algorithms.md) | [Table of Contents](00-index.md) | [Next: The Matrix Core →](05-matrix-core.md)

---

## Training Loop: Putting It All Together

The training loop combines forward pass, loss computation, backward pass, and weight update into a complete learning cycle. This cycle repeats thousands or millions of times during training, gradually improving the model's ability to make correct predictions.

The complete cycle works as follows. First, the forward pass computes a prediction by passing input data through the network. Next, we compute the loss by comparing this prediction to the target. Then the backward pass computes gradients, telling us how much each parameter should change. Finally, we perform a weight update, changing the parameters using gradient descent. We then repeat this entire cycle with the next example.

![Training Loop](images/training/training-loop.svg)

To understand this intuitively, think of learning to play a musical instrument. You play a note (forward pass), hear if it's wrong (loss), figure out what to adjust (backward pass), and adjust your fingers (weight update). You repeat this process until you play correctly. Neural network training follows the same pattern, but instead of adjusting your fingers, the model adjusts its weights.

### Weight Update

Weight update is the process of changing matrix values (weights) based on gradients to improve predictions. Think of weight updates like tuning a radio. The current setting (weight) is like the current frequency. The static (loss) tells you how bad the signal is. The gradient tells you which direction to turn the dial, and the weight update is actually turning the dial. After many adjustments, you find the best frequency—the weight values that minimize loss.

We illustrate how weight updates work in practice using example values chosen to demonstrate the update process:

**Given:**
- Before training:
  - Weight matrix: $\mathbf{W}_{\text{old}} = \begin{bmatrix} 0.1 & 0.2 \\ 0.3 & 0.4 \end{bmatrix}$ (example initial weights)
  - Loss: $L = 2.5$ (example high loss value)
- After computing gradients:
  - Gradient: $\nabla_{\mathbf{W}} L = \begin{bmatrix} -0.5 & -0.3 \\ -0.2 & -0.1 \end{bmatrix}$ (example gradient values)
  - Learning rate: $\eta = 0.1$

**Equation to solve:**
$$\mathbf{W}_{\text{new}} = \mathbf{W}_{\text{old}} - \eta \cdot \nabla_{\mathbf{W}} L = \begin{bmatrix} 0.1 & 0.2 \\ 0.3 & 0.4 \end{bmatrix} - 0.1 \times \begin{bmatrix} -0.5 & -0.3 \\ -0.2 & -0.1 \end{bmatrix} = \begin{bmatrix} 0.15 & 0.23 \\ 0.32 & 0.41 \end{bmatrix}$$

**Computation:**

Applying the gradient descent update rule with learning rate $\eta = 0.1$:

$$\mathbf{W}_{\text{new}} = \mathbf{W}_{\text{old}} - \eta \cdot \nabla_{\mathbf{W}} L = \begin{bmatrix} 0.1 & 0.2 \\ 0.3 & 0.4 \end{bmatrix} - 0.1 \times \begin{bmatrix} -0.5 & -0.3 \\ -0.2 & -0.1 \end{bmatrix} = \begin{bmatrix} 0.15 & 0.23 \\ 0.32 & 0.41 \end{bmatrix}$$

After many such updates during training, the loss decreases from $L = 2.5$ to $L = 0.1$ (example low loss value after training).

The weights change based on the gradients, and after many such updates, the loss decreases dramatically. This is how the model learns patterns from examples—by repeatedly making small adjustments to its weights based on the errors it makes. The specific values are illustrative; in practice, these would be computed from the actual model, data, and loss function.

To see a complete training cycle, [Example 2: Single Training Step](10-example2-single-step.md) shows one complete iteration of the training loop.

---

## Batch Training: Processing Multiple Examples

A batch is a group of sequences processed together during training. Think of batch training like grading multiple papers at once. Instead of grading one paper, you grade 32 papers together. This is more efficient because it enables parallel processing, and you average the results across all papers to get a more stable estimate of how well the model is performing.

In practice, the batch size is the number of sequences processed together (typically 32, 64, or 128). All sequences in the batch are processed in parallel, which makes efficient use of GPU resources. The gradients computed for each sequence are averaged across the batch, and the loss is also averaged. This averaging provides a more stable estimate of the true gradient than processing sequences one at a time.

The following example illustrates how batching works. For clarity, we use simple text sequences as examples, though in practice these would be numerical vectors:

**Batch configuration:**
- Batch size: $B = 4$
- Batch contains 4 sequences:
  1. ["The", "cat", "sat"] (example sequence 1)
  2. ["The", "dog", "ran"] (example sequence 2)
  3. ["A", "bird", "flew"] (example sequence 3)
  4. ["A", "fish", "swam"] (example sequence 4)

All 4 sequences are processed together in parallel, and their gradients are averaged across the batch.

The sequences shown are illustrative examples. In practice, these would be converted to numerical vectors before processing (we'll explain how tokens become vectors in [Chapter 6: Embeddings](06-embeddings.md)). The key point is that all sequences in the batch are processed in parallel, and their gradients are averaged.

![Batch Training](images/training/batch-training.svg)

Batching matters because it enables efficient GPU utilization and provides stable gradient estimates. Larger batches give more stable gradients (since you're averaging over more examples) but require more memory. Smaller batches use less memory but may have noisier gradients.

To see batch training in action, [Example 4: Multiple Patterns](12-example4-multiple-patterns.md) demonstrates how multiple sequences are processed together.

### Epochs: Complete Passes Through Data

An epoch is one complete pass through the entire training dataset. Think of an epoch like reading an entire textbook once. You start at page 1 and read through to the end—that's one epoch. Multiple epochs means reading the book multiple times to learn better. Each time through, you notice different details and reinforce what you've learned.

In training, if you have a dataset with 10,000 sequences and a batch size of 32, you'll have 10,000 ÷ 32 = 313 batches per epoch. One epoch means processing all 313 batches. Training typically involves repeating this for multiple epochs (e.g., 10 epochs), giving the model multiple chances to see all the training data and improve.

We illustrate this with a numerical example using arbitrary values chosen to demonstrate the epoch concept:

**Given:**
- Dataset size: $N = 1000$ sequences (example dataset size)
- Batch size: $B = 32$ (example batch size)
- Number of batches per epoch: $\lceil N / B \rceil = \lceil 1000 / 32 \rceil \approx 32$ batches

**Epoch progression:**
- **Epoch 1**: Process batches 1 through 32 (all 1000 sequences)
- **Epoch 2**: Process batches 1 through 32 again (same sequences, typically in different order)
- **Epoch 3**: Process batches 1 through 32 again
- This process continues until the model converges

The specific numbers (1000 sequences, batch size 32) are chosen for illustration. In practice, these values would depend on the actual dataset size and available computational resources. The key concept is that one epoch means processing the entire dataset once, divided into batches.

Models typically need multiple epochs to learn effectively. Each epoch gives the model another opportunity to see all training data and improve its predictions. Early epochs show rapid improvement as the model learns basic patterns, while later epochs show slower, more refined improvements.

![Training Progress: Loss Decreases Over Epochs](images/training/training-loss-epochs.svg)

---

## From Neural Networks to Transformers

Transformers are a special type of neural network that uses attention to process sequences. Now that we understand the fundamentals of neural networks—perceptrons, layers, loss functions, gradient descent, and backpropagation—we can see how all these concepts come together in transformer architectures.

### The Transformer Architecture

A transformer is built from three main components. First, input processing handles the conversion of raw text into a form the network can work with. This involves tokenization (breaking text into tokens), token encoding (converting tokens to integer IDs), and embedding lookup (converting integer IDs to vectors).

Second, transformer blocks (of which there can be multiple) perform the core processing. Each block contains an attention layer that finds relevant information, a feed-forward network that adds non-linearity, layer normalization that stabilizes training, and residual connections that enable training of very deep networks.

Third, the output component converts the processed information into predictions. This includes an output projection that converts to vocabulary predictions, and a softmax function that converts those predictions into probabilities.

![Complete Transformer Architecture](images/network-structure/complete-transformer-architecture.svg)

To see the complete transformer architecture implemented with all components working together, [Example 6: Complete Transformer](14-example6-complete.md) demonstrates the full implementation including multiple transformer blocks, layer normalization, residual connections, and the complete training pipeline.

### Key Transformer Concepts

Now that we understand the basic architecture, let's define the key concepts that make transformers work. These terms will appear throughout the rest of this book, so it's important to understand them clearly. We'll organize them by theme, starting with how text enters the system and progressing through how it's processed.

#### Input Processing: From Text to Vectors

**Tokenization and Encoding**

The journey begins with **tokenization**, the process of breaking text into discrete units called tokens. Think of tokenization like cutting a sentence into individual words. For example, the input "The cat sat on the mat." becomes ["The", "cat", "sat", "on", "the", "mat", "."] after tokenization, with each piece now being a separate token. A **token** is the smallest unit of input that a transformer processes—think of it like a word on a Scrabble tile, where each tile represents one piece of information that the model can work with.

The **vocabulary** is the complete set of all possible tokens that a transformer can recognize. Think of vocabulary like a dictionary or Scrabble tile bag. It contains all possible words or tokens the model knows, has a limited size (e.g., 50,000 tokens for GPT models), and each token has a unique ID or index. Once we have tokens, **token encoding** converts them (strings) into integer IDs. Think of token encoding like assigning a student ID number to each student. The student name (token) maps to a student ID (integer), so "cat" might map to 2. This mapping uses the vocabulary we just defined.

**Embeddings**

**Embedding lookup** is the process of converting integer token IDs into vector representations using an embedding matrix. Think of embedding lookup like using a student ID to retrieve their file from a filing cabinet. The student ID (integer) gives you access to the student file (a vector of information), so ID 2 might retrieve the vector $\begin{bmatrix} 0.3 \\ 0.7 \\ -0.2 \end{bmatrix}$. An **embedding** converts a discrete token into a continuous vector (a list of numbers). Think of an embedding like a translator that converts a word (a discrete symbol) into a point on a map (continuous coordinates). The word "cat" is just a symbol, but its embedding $\begin{bmatrix} 0.3 \\ 0.7 \\ -0.2 \end{bmatrix}$ represents a location in meaning-space where similar words are close together.

#### Mathematical Foundations

**Vectors, Matrices, and Dimensions**

A **vector** is an ordered list of numbers, representing a point in space. Think of a vector like GPS coordinates. The vector $\begin{bmatrix} 0.3 \\ 0.7 \end{bmatrix}$ represents a point at (0.3, 0.7) on a 2D map, while $\begin{bmatrix} 0.3 \\ 0.7 \\ -0.2 \end{bmatrix}$ represents a point in 3D space. In basic algebra, you might add vectors component-wise: $\begin{bmatrix} 1 \\ 2 \end{bmatrix} + \begin{bmatrix} 3 \\ 4 \end{bmatrix} = \begin{bmatrix} 4 \\ 6 \end{bmatrix}$. This is just like adding coordinates—you add the x-components together and the y-components together.

A **matrix** is a rectangular grid of numbers, used to transform vectors. In algebra, matrix multiplication is like solving a system of linear equations. When you multiply a matrix by a vector, you're essentially computing multiple weighted sums simultaneously. For example, if you have equations $y_1 = 2x_1 + 3x_2$ and $y_2 = 4x_1 + 5x_2$, you can write this as a matrix multiplication where the matrix contains the coefficients.

Matrix multiplication $\mathbf{A}\mathbf{B}$ computes the dot product of each row of $\mathbf{A}$ with each column of $\mathbf{B}$. This is equivalent to computing multiple weighted sums simultaneously—exactly what happens when a layer processes an input vector. Think of a matrix like a machine that transforms objects. You feed in a vector (point A), the matrix applies transformation rules (rotate, scale, shift), and you get out a new vector (point B). **Dimension** refers to the size or length of a vector or matrix axis. Think of dimension like the number of measurements needed to describe something. In 2D, you need (x, y) coordinates on a map. In 3D, you need (x, y, z) coordinates in space. Higher dimensions require more measurements or features to fully describe the data.

**Model Parameters**

A **parameter** is a value in the model that gets learned (updated) during training. Think of parameters like adjustable knobs on a machine. Each knob controls some aspect of the machine's behavior, and during training, we adjust the knobs to make the machine work better. Weights and biases are both types of parameters. A **weight** is a parameter (number) in a matrix that determines how inputs are transformed. Think of weights like the strength of connections in a network. A high weight creates a strong connection, meaning the input has a big influence, while a low weight creates a weak connection with small influence. A **bias** is a parameter (number) that's added to a computation to shift the result. Think of bias like a baseline or offset, similar to setting a scale to zero before weighing something. It shifts the entire computation up or down.

A **hyperparameter** is a configuration setting that controls how the model is trained or structured, but is NOT learned during training. Think of hyperparameters like settings on a machine before you start it. The learning rate controls how fast the machine adjusts, the batch size determines how many items are processed at once, and the number of layers specifies how many processing stages the machine has.

#### Sequence Processing

**Sequences and Chunking**

A **sequence** is an ordered list of tokens that the transformer processes together. Think of a sequence like a sentence or paragraph. The order matters—"cat sat" is different from "sat cat"—and sequences have a fixed maximum length (e.g., 512 tokens for many models). When documents are too long, **chunking** splits them into smaller, manageable pieces called chunks. Think of chunking like dividing a long book into chapters. A long document gets split into multiple smaller chunks, with each chunk fitting within the model's sequence length limit.

#### Attention Mechanism

**Query, Key, and Value**

**Query (Q), Key (K), and Value (V)** are three different representations of the same token, each serving a specific purpose in attention. Think of attention like a library search system. The Query (Q) asks "What am I looking for?"—like typing a search query. The Key (K) says "What do I have to offer?"—like keywords on a book's spine. The Value (V) contains "What is my actual content?"—like the actual book content. **Q/K/V Maps** are three separate matrices that transform the same embedding into Query, Key, and Value vectors. Think of Q/K/V maps like three different lenses looking at the same object. You have the same token embedding (the object), but the $\mathbf{W}_Q$ lens gives you the Query view (what I'm looking for), the $\mathbf{W}_K$ lens gives you the Key view (what I'm advertising), and the $\mathbf{W}_V$ lens gives you the Value view (what I actually contain).

**Attention Dot Product**

The **attention dot product** is a mathematical operation that measures how similar two vectors are. In algebra, the dot product is computed by multiplying corresponding components and summing: for vectors $\begin{bmatrix} a \\ b \end{bmatrix}$ and $\begin{bmatrix} c \\ d \end{bmatrix}$, the dot product is $ac + bd$. This is essentially a weighted sum—you're multiplying each component of one vector by the corresponding component of the other, then adding everything together.

The dot product $\mathbf{Q} \cdot \mathbf{K}$ measures cosine similarity when vectors are normalized. For unnormalized vectors, it measures both magnitude and alignment. Think of the dot product like measuring alignment. A high dot product means the vectors point in the same direction, indicating they're similar and relevant to each other. A low dot product means the vectors point in different directions, indicating they're different and not relevant.

#### Output Generation

**Context Vectors and Output Projection**

A **context vector** is a weighted combination of all token values, where the weights come from attention. In basic algebra, this is like computing a weighted average. If you have values $\begin{bmatrix} a \\ b \\ c \end{bmatrix}$ with weights $\begin{bmatrix} 0.5 \\ 0.3 \\ 0.2 \end{bmatrix}$, the weighted average is $0.5a + 0.3b + 0.2c$. The context vector does exactly this: it multiplies each token value by its attention weight and sums them together. Think of a context vector like a blended smoothie. Each fruit (token value) contributes to the smoothie, and the amount of each fruit equals the attention weight. The final smoothie (the context vector) contains all the blended information. **Output Projection ($\mathbf{W}_O$)** is a matrix that transforms the context vector into vocabulary-space (predictions for each token). Think of $\mathbf{W}_O$ like a translator that converts context meaning (in semantic space) into the likelihood of each word (in vocabulary space). It's like converting "animal, four legs, meows" into "cat: 80%, dog: 15%, mat: 5%".

**Logits and Softmax**

**Logits** (pronounced "low-jits") are the raw, unnormalized scores output by the model before applying softmax. The term "logit" was coined by American statistician Joseph Berkson in 1944, derived from "logistic unit"—it represents the **log-odds transformation** (the logarithm of the odds ratio).

**Understanding the Original Meaning:**
To understand what "logarithm of odds ratio" means, we need to break it down:
- **Odds**: If the probability of an event is $p$, then the odds are $\frac{p}{1-p}$. For example, if the probability of rain is 0.75 (75%), the odds are $\frac{0.75}{1-0.75} = \frac{0.75}{0.25} = 3$ (often written as "3 to 1").
- **Odds ratio**: The ratio of two odds. For example, if Group A has odds of 3 and Group B has odds of 1, the odds ratio is $\frac{3}{1} = 3$.
- **Logarithm of odds ratio (logit)**: The natural logarithm of the odds: $\text{logit}(p) = \ln\left(\frac{p}{1-p}\right)$. This transformation converts probabilities (which range from 0 to 1) into real numbers (which can be any value from $-\infty$ to $+\infty$).

**Why This Matters for Neural Networks:**
In neural networks, we use "logit" more broadly to mean any raw score before it's converted to a probability, even though the original statistical meaning was more specific. The key insight is that logits are unconstrained real numbers, while probabilities are constrained to the range [0, 1]. This is why we need softmax to convert logits to probabilities.

Think of logits like raw test scores before grading on a curve. You might have raw scores [85, 90, 75, 80], and after applying the curve (softmax), you get probabilities [0.2, 0.5, 0.1, 0.2]. Logits can be any real numbers (positive, negative, large, small), while probabilities must be between 0 and 1 and sum to 1. **Softmax** is a function that converts numbers into probabilities (they sum to 1.0). In basic arithmetic, this is like converting numbers to percentages that add up to 100%. If you have test scores [85, 90, 75] out of 100, you might convert them to percentages, but softmax does something more sophisticated: it ensures the largest number gets the biggest share while all numbers sum to exactly 1.0. Think of softmax like dividing a pie. If you have scores [5, 2, 1], softmax converts them to probabilities [0.7, 0.2, 0.1]. The largest score gets the biggest slice, and all slices sum to 1.0 (the whole pie).

#### Training Techniques

**Layer Normalization**

**Layer normalization** is a technique that **normalizes** (standardizes) the inputs to a layer by adjusting the **mean** and **variance**. This technique serves two important purposes:

**1. What it does:**
Layer normalization transforms the inputs so they have consistent statistical properties. In basic algebra and statistics, this is exactly like computing **z-scores**: you subtract the **mean** (the average value) and divide by the **standard deviation** (a measure of how spread out the values are).

The formula is: $z = \frac{x - \mu}{\sigma}$

where:
- $\mu$ (pronounced "mu") is the **mean** (average value)
- $\sigma$ (pronounced "sigma") is the **standard deviation** (measure of spread)

This transforms any set of numbers so they have a mean of 0 and a standard deviation of 1.

**2. Why it matters:**
Think of layer normalization like standardizing test scores. Raw scores might vary widely (0-100), but after normalization (subtract mean, divide by standard deviation), the scores are centered around 0 with a consistent scale. This helps the network learn more effectively because:

- It prevents values from becoming too large or too small (which can cause numerical instability)
- It makes the training process more stable and faster
- It helps gradients flow better through deep networks

**3. Where it's used:**
To see layer normalization in a complete transformer implementation, [Example 6: Complete Transformer](14-example6-complete.md) includes layer normalization in all transformer blocks.

**Residual Connections**

A **residual connection** adds the input of a layer directly to its output. In algebra, this is like adding two functions together: if you have $f(x)$ and you compute $f(x) + x$, you're adding the original input to the transformed output. This is exactly what a residual connection does: Output = Layer(Input) + Input. Think of a residual connection like a shortcut or bypass. The main path goes Input → Layer → Output, but there's also a shortcut that goes Input → (directly to output). The final output is Layer(Input) + Input. Residual connections enable training of very deep networks by allowing gradients to flow directly through the shortcut, which helps prevent the vanishing gradient problem that can occur in very deep networks.

To see these concepts in action, [Example 5: Feed-Forward Layers](13-example5-feedforward.md) shows residual connections in action, and [Example 6: Complete Transformer](14-example6-complete.md) shows the full architecture with all components working together.

---

## The Complete Learning Cycle

Now that we've covered all the individual components, let's see how they work together in a complete transformer. The learning cycle consists of a forward pass (where the model makes predictions) and a backward pass (where the model learns from its mistakes).

During the forward pass, the model makes predictions. First, tokenization breaks the text "The cat sat" into tokens: ["The", "cat", "sat"]. Token encoding then converts these tokens to integer IDs: [1, 2, 3]. Embedding lookup converts the integer IDs into vectors: [[0.1, 0.2], [0.3, 0.7], [0.5, 0.1]]. The Q/K/V maps transform each embedding into three views (Query, Key, and Value). Attention computes Q·K (dot product) to find which tokens are relevant to each other. Softmax converts the attention scores to probabilities. The context vector is computed as a weighted sum of all Values, where the weights come from attention. Output projection (WO) transforms the context vector into vocabulary scores (logits). Finally, softmax converts the logits into prediction probabilities.

During the backward pass, the model learns from its mistakes. The loss function compares the prediction to the target. The backward pass then computes gradients, which flow backward through the network using the chain rule. Finally, weight updates change the parameters (weights and biases) to reduce the loss, using the gradient descent algorithm we learned earlier.

To see these concepts in action, we've prepared several examples that build from simple to complex. [Example 1: Minimal Forward Pass](09-example1-forward-pass.md) demonstrates the forward pass only, showing how predictions are made. [Example 2: Single Training Step](10-example2-single-step.md) shows one complete training cycle, combining forward pass, loss computation, and weight updates. [Example 3: Full Backpropagation](11-example3-full-backprop.md) traces the complete gradient flow through all components. [Example 4: Multiple Patterns](12-example4-multiple-patterns.md) demonstrates batch training with multiple sequences. [Example 5: Feed-Forward Layers](13-example5-feedforward.md) adds feed-forward networks and residual connections to the architecture. Finally, [Example 6: Complete Transformer](14-example6-complete.md) shows the full architecture with all components working together.

---

## Key Principles

As we conclude this chapter, let's summarize the key principles that underlie everything we've learned. First, everything in a transformer is a vector or matrix. Tokens become vectors, and all operations are performed using matrices. This mathematical foundation enables the parallel processing that makes transformers efficient.

Second, attention finds relevance. The Q·K dot product measures how relevant each token is to every other token, allowing the model to focus on the most important information when making predictions.

Third, softmax creates probabilities. It converts any scores into probabilities that sum to 1, which is essential for making predictions about which token comes next.

Fourth, the context vector combines information. It's a weighted sum of all token values, where the weights come from attention. This allows the model to blend information from multiple tokens based on their relevance.

Fifth, learning equals gradient descent. Gradients show how to update weights to reduce loss, and this is the mechanism that enables all neural network learning, from simple perceptrons to complex transformers.

---

## What's Next?

Now that you understand neural network fundamentals and how they apply to transformers, you're ready to dive deeper into the specific components that make transformers work. In [Chapter 5: The Matrix Core](05-matrix-core.md), we'll take a deep dive into matrix operations, which are the mathematical foundation of everything transformers do. [Chapter 6: Embeddings](06-embeddings.md) will show you exactly how tokens become vectors and why this representation is so powerful. [Chapter 7: Attention Intuition](07-attention-intuition.md) will help you develop a deep understanding of how attention finds relevant information. Finally, [Chapter 8: Why Transformers?](08-why-transformers.md) will explain the specific problems that transformers solve and why they've become so dominant in modern AI.

For quick reference as you continue reading, see [Appendix B: Terminology Reference](appendix-b-terminology-reference.md) for all definitions with their physical analogies.

Remember: Every concept in this chapter has a physical analogy. If you ever forget what something means, think about its physical analogy—that's what it's actually modeling. These analogies aren't just helpful mnemonics; they reflect the real-world processes that neural networks are designed to capture.

---

**Navigation:**
- [← Previous: Learning Algorithms](03-learning-algorithms.md) | [← Index](00-index.md) | [Next: The Matrix Core →](05-matrix-core.md)

