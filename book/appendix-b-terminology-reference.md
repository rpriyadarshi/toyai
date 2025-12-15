## Appendix B: Terminology Reference

**This appendix provides quick reference for all terminology used in this book. Each term includes its definition, physical analogy, and cross-reference to Chapter 1 where it's first introduced.**

**Note:** All definitions include physical analogies to connect abstract math to tangible reality. This breaks the cycle of memorizing math without understanding what it models.

---

### Activation Function

**What it is:** An activation function is a non-linear function applied to layer outputs to introduce non-linearity into the network.

**Physical Analogy:** Think of activation like a **filter that shapes the signal**:
- Without activation: network is just linear transformations (limited)
- With activation: network can learn complex, non-linear patterns
- Different activations = different "shapes" of transformation

**Common Activation Functions:**
- **ReLU**: $f(x) = \max(0, x)$ - Keeps positive, zeros negative
- **Sigmoid**: $f(x) = \frac{1}{1+e^{-x}}$ - Squashes to 0-1 range
- **Tanh**: $f(x) = \tanh(x)$ - Squashes to -1 to 1 range

**See:** [Chapter 1: Neural Network Fundamentals](01-neural-network-fundamentals.md) - "The Perceptron: The Basic Building Block"

---

### Attention Dot Product

**What it is:** A mathematical operation that measures how similar two vectors are.

**Physical Analogy:** Think of dot product like **measuring alignment**:
- High dot product = vectors point in same direction = similar = relevant
- Low dot product = vectors point different directions = different = not relevant
- Zero dot product = perpendicular = unrelated

**See:** [Chapter 1: Neural Network Fundamentals](01-neural-network-fundamentals.md) - "From Neural Networks to Transformers"

---

### Backpropagation

**What it is:** The algorithm that computes gradients by propagating the loss backward through the network.

**Physical Analogy:** Think of backward pass like **tracing back the cause of a mistake**:
- You made an error (high loss)
- Work backwards: "What caused this error?"
- Check each step: "Did this layer contribute to the error?"
- Calculate how much each parameter should change

**See:** [Chapter 1: Neural Network Fundamentals](01-neural-network-fundamentals.md) - "Backpropagation: Computing Gradients"

---

### Backward Pass

**What it is:** The process of computing gradients by propagating the loss backward through the network from output to input.

**Physical Analogy:** Think of backward pass like **tracing back the cause of a mistake**.

**See:** [Chapter 1: Neural Network Fundamentals](01-neural-network-fundamentals.md) - "Backpropagation: Computing Gradients"

---

### Batch

**What it is:** A batch is a group of sequences processed together during training.

**Physical Analogy:** Think of a batch like **grading multiple papers at once**:
- Instead of grading one paper, grade 32 papers together
- More efficient (parallel processing)
- Average the results across all papers

**See:** [Chapter 1: Neural Network Fundamentals](01-neural-network-fundamentals.md) - "Batch Training: Processing Multiple Examples"

---

### Bias

**What it is:** A bias is a parameter (number) that's added to a computation to shift the result.

**Physical Analogy:** Think of bias like a **baseline or offset**:
- Like setting a scale to zero before weighing
- Or adjusting a thermostat's baseline temperature
- Shifts the entire computation up or down

**See:** [Chapter 1: Neural Network Fundamentals](01-neural-network-fundamentals.md) - "The Perceptron: The Basic Building Block"

---

### Chunking

**What it is:** Chunking is the process of splitting long documents or sequences into smaller, manageable pieces (chunks).

**Physical Analogy:** Think of chunking like **dividing a long book into chapters**:
- Long document → Multiple smaller chunks
- Each chunk fits within model's sequence length limit
- Chunks can overlap to preserve context

**See:** [Chapter 1: Neural Network Fundamentals](01-neural-network-fundamentals.md) - "From Neural Networks to Transformers"

---

### Context Vector

**What it is:** A weighted combination of all token values, where weights come from attention.

**Physical Analogy:** Think of context vector like a **blended smoothie**:
- Each fruit (token value) contributes
- Amount of each fruit = attention weight
- Final smoothie = context vector (blended information)

**See:** [Chapter 1: Neural Network Fundamentals](01-neural-network-fundamentals.md) - "From Neural Networks to Transformers"

---

### Dimension

**What it is:** Dimension is the size or length of a vector or matrix axis.

**Physical Analogy:** Think of dimension like **the number of measurements** needed to describe something:
- 2D: (x, y) coordinates on a map
- 3D: (x, y, z) coordinates in space
- Higher dimensions: More measurements/features

**See:** [Chapter 1: Neural Network Fundamentals](01-neural-network-fundamentals.md) - "From Neural Networks to Transformers"

---

### Embedding

**What it is:** An embedding converts a discrete token into a continuous vector (list of numbers).

**Physical Analogy:** Think of an embedding like a **translator** that converts:
- A word (discrete symbol) → A point on a map (continuous coordinates)
- "cat" (just a symbol) → `[0.3, 0.7, -0.2]` (a location in meaning-space)

**See:** [Chapter 1: Neural Network Fundamentals](01-neural-network-fundamentals.md) - "From Neural Networks to Transformers"

---

### Embedding Lookup

**What it is:** Embedding lookup is the process of converting integer token IDs into vector representations using an embedding matrix.

**Physical Analogy:** Think of embedding lookup like **using a student ID to retrieve their file from a filing cabinet**:
- Student ID (integer) → Student file (vector of information)
- ID 12345 → File with [grades, attendance, address, ...]
- Each ID maps to a specific file location

**See:** [Chapter 1: Neural Network Fundamentals](01-neural-network-fundamentals.md) - "From Neural Networks to Transformers"

---

### Epoch

**What it is:** An epoch is one complete pass through the entire training dataset.

**Physical Analogy:** Think of an epoch like **reading an entire textbook once**:
- Start at page 1, read through to the end
- That's one epoch
- Multiple epochs = reading the book multiple times to learn better

**See:** [Chapter 1: Neural Network Fundamentals](01-neural-network-fundamentals.md) - "Batch Training: Processing Multiple Examples"

---

### Feed-Forward Network (FFN)

**What it is:** A feed-forward network is a layer that applies two linear transformations with an activation function in between.

**Physical Analogy:** Think of FFN like a **two-stage processing pipeline**:
- Stage 1: Expand (increase dimensions)
- Stage 2: Compress (reduce back to original dimensions)
- Activation in between adds non-linearity

**See:** [Chapter 1: Neural Network Fundamentals](01-neural-network-fundamentals.md) - "Feedforward Networks: Multi-Layer Perceptrons"

---

### Forward Pass

**What it is:** The forward pass is the process of computing predictions by passing input data through the network from input to output.

**Physical Analogy:** Think of forward pass like **following a recipe step-by-step**:
- Start with ingredients (input tokens)
- Process through each step (each layer)
- End with final dish (prediction)
- Data flows in one direction: Input → Layer 1 → Layer 2 → ... → Output

**See:** [Chapter 1: Neural Network Fundamentals](01-neural-network-fundamentals.md) - "Backpropagation: Computing Gradients"

---

### Gradient

**What it is:** A gradient shows how much each parameter should change to reduce the loss.

**Physical Analogy:** Think of gradient like a **compass pointing uphill**:
- Loss is like altitude (want to go down)
- Gradient points in the direction of steepest increase
- Negative gradient = direction to go DOWN (reduce loss)
- Magnitude = how steep the slope is

**See:** [Chapter 1: Neural Network Fundamentals](01-neural-network-fundamentals.md) - "Gradient Descent: How Networks Learn"

---

### Gradient Descent

**What it is:** Gradient descent is an optimization algorithm that uses gradients to iteratively update parameters to minimize loss.

**Physical Analogy:** Think of gradient descent like **walking downhill blindfolded**:
- You can't see the bottom, but you can feel which way is downhill (gradient)
- Take a step in that direction (weight update)
- Repeat until you reach the bottom (minimum loss)
- Step size = learning rate

**See:** [Chapter 1: Neural Network Fundamentals](01-neural-network-fundamentals.md) - "Gradient Descent: How Networks Learn"

---

### Hyperparameter

**What it is:** A hyperparameter is a configuration setting that controls how the model is trained or structured, but is NOT learned during training.

**Physical Analogy:** Think of hyperparameters like **settings on a machine before you start it**:
- Learning rate: How fast the machine adjusts
- Batch size: How many items processed at once
- Number of layers: How many processing stages
- These are set BEFORE training, not learned DURING training

**See:** [Chapter 1: Neural Network Fundamentals](01-neural-network-fundamentals.md) - "From Neural Networks to Transformers"

---

### Key (K)

**What it is:** One of three different representations of the same token. Key represents "What do I have to offer?"

**Physical Analogy:** Think of Key like **keywords on a book's spine**:
- "cat" offers: "I am a noun, I am the subject"
- Physical: An **advertisement** of what you contain

**See:** [Chapter 1: Neural Network Fundamentals](01-neural-network-fundamentals.md) - "From Neural Networks to Transformers"

---

### Layer

**What it is:** A layer is a computational unit in a neural network that transforms its input to produce output.

**Physical Analogy:** Think of a layer like a **factory assembly line station**:
- Input arrives → Layer processes it → Output goes to next layer
- Each layer does a specific job (embedding, attention, feed-forward)
- Multiple layers = multiple processing steps

**See:** [Chapter 1: Neural Network Fundamentals](01-neural-network-fundamentals.md) - "Layers: Stacking Neurons"

---

### Layer Normalization

**What it is:** Layer normalization is a technique that normalizes the inputs to a layer by adjusting the mean and variance.

**Physical Analogy:** Think of layer normalization like **standardizing test scores**:
- Raw scores vary widely (0-100)
- Normalize: subtract mean, divide by standard deviation
- Result: scores centered around 0 with consistent scale

**See:** [Chapter 1: Neural Network Fundamentals](01-neural-network-fundamentals.md) - "From Neural Networks to Transformers"

---

### Learning Rate

**What it is:** The learning rate is a hyperparameter that controls how large each weight update is during training.

**Physical Analogy:** Think of learning rate like **step size when walking downhill**:
- Large steps (high learning rate): Fast progress, but might overshoot the bottom
- Small steps (low learning rate): Slow progress, but more precise
- Too large: You jump over the valley (divergence)
- Too small: You take forever to reach the bottom (slow convergence)

**See:** [Chapter 1: Neural Network Fundamentals](01-neural-network-fundamentals.md) - "Gradient Descent: How Networks Learn"

---

### Logits

**What it is:** Logits are the raw, unnormalized scores output by the model before applying softmax.

**Physical Analogy:** Think of logits like **raw test scores before grading on a curve**:
- Raw scores: [85, 90, 75, 80] (logits)
- After curve (softmax): [0.2, 0.5, 0.1, 0.2] (probabilities)
- Logits can be any real numbers (positive, negative, large, small)
- Probabilities must be 0-1 and sum to 1

**See:** [Chapter 1: Neural Network Fundamentals](01-neural-network-fundamentals.md) - "From Neural Networks to Transformers"

---

### Loss (Softmax Loss / Cross-Entropy Loss)

**What it is:** A measure of how wrong the model's prediction is compared to the target.

**Physical Analogy:** Think of loss like a **score in a game**:
- Lower loss = better prediction = you're winning
- Higher loss = worse prediction = you're losing
- Goal: minimize loss (maximize accuracy)

**See:** [Chapter 1: Neural Network Fundamentals](01-neural-network-fundamentals.md) - "Loss Functions: Measuring Error"

---

### Matrix

**What it is:** A matrix is a rectangular grid of numbers, used to transform vectors.

**Physical Analogy:** Think of a matrix like a **machine that transforms objects**:
- Input: a vector (point A)
- Matrix: transformation rules (rotate, scale, shift)
- Output: a new vector (point B)

**See:** [Chapter 1: Neural Network Fundamentals](01-neural-network-fundamentals.md) - "From Neural Networks to Transformers"

---

### Output Projection (WO)

**What it is:** A matrix that transforms the context vector into vocabulary-space (predictions for each token).

**Physical Analogy:** Think of WO like a **translator** that converts:
- Context meaning (in semantic space) → Likelihood of each word (in vocabulary space)
- Like converting "animal, four legs, meows" → "cat: 80%, dog: 15%, mat: 5%"

**See:** [Chapter 1: Neural Network Fundamentals](01-neural-network-fundamentals.md) - "From Neural Networks to Transformers"

---

### Parameter

**What it is:** A parameter is a value in the model that gets learned (updated) during training.

**Physical Analogy:** Think of parameters like **adjustable knobs on a machine**:
- Each knob controls some aspect of the machine's behavior
- During training, we adjust knobs to make the machine work better
- Once trained, knobs are fixed at their learned values

**See:** [Chapter 1: Neural Network Fundamentals](01-neural-network-fundamentals.md) - "From Neural Networks to Transformers"

---

### Q/K/V Maps (Projection Matrices)

**What they are:** Three separate matrices that transform the same embedding into Query, Key, and Value vectors.

**Physical Analogy:** Think of Q/K/V maps like **three different lenses** looking at the same object:
- Same token embedding (the object)
- WQ lens → Query view (what I'm looking for)
- WK lens → Key view (what I'm advertising)
- WV lens → Value view (what I actually contain)

**See:** [Chapter 1: Neural Network Fundamentals](01-neural-network-fundamentals.md) - "From Neural Networks to Transformers"

---

### Query (Q)

**What it is:** One of three different representations of the same token. Query represents "What am I looking for?"

**Physical Analogy:** Think of Query like **typing a search query into Google**:
- "sat" asks: "What is the subject?"
- Physical: A **question** you ask

**See:** [Chapter 1: Neural Network Fundamentals](01-neural-network-fundamentals.md) - "From Neural Networks to Transformers"

---

### Residual Connection

**What it is:** A residual connection (also called skip connection) adds the input of a layer directly to its output.

**Physical Analogy:** Think of residual connection like a **shortcut or bypass**:
- Main path: Input → Layer → Output
- Shortcut: Input → (directly to output)
- Final: Output = Layer(Input) + Input

**See:** [Chapter 1: Neural Network Fundamentals](01-neural-network-fundamentals.md) - "From Neural Networks to Transformers"

---

### Sequence

**What it is:** A sequence is an ordered list of tokens that the transformer processes together.

**Physical Analogy:** Think of a sequence like a **sentence** or **paragraph**:
- Ordered: position matters ("cat sat" ≠ "sat cat")
- Fixed length: sequences have maximum length (e.g., 512 tokens)
- Context: tokens relate to each other based on position

**See:** [Chapter 1: Neural Network Fundamentals](01-neural-network-fundamentals.md) - "From Neural Networks to Transformers"

---

### Softmax

**What it is:** A function that converts numbers into probabilities (they sum to 1.0).

**Physical Analogy:** Think of softmax like **dividing a pie**:
- You have scores: [5, 2, 1]
- Softmax converts to probabilities: [0.7, 0.2, 0.1]
- The largest score gets the biggest slice
- All slices sum to 1.0 (the whole pie)

**See:** [Chapter 1: Neural Network Fundamentals](01-neural-network-fundamentals.md) - "From Neural Networks to Transformers"

---

### Token

**What it is:** A token is the smallest unit of input that a transformer processes. It's the result of tokenization.

**Physical Analogy:** Think of a token like a **word on a Scrabble tile**. Each tile represents one piece of information:
- "cat" = one token
- "the" = one token  
- "!" = one token (punctuation is also a token)

**See:** [Chapter 1: Neural Network Fundamentals](01-neural-network-fundamentals.md) - "From Neural Networks to Transformers"

---

### Token Encoding

**What it is:** Token encoding is the process of converting tokens (strings) into integer IDs that the model can process.

**Physical Analogy:** Think of token encoding like **assigning a student ID number to each student**:
- Student name (token) → Student ID (integer)
- "John Smith" → 12345
- Each student has a unique ID number
- The ID is used to look up student records

**See:** [Chapter 1: Neural Network Fundamentals](01-neural-network-fundamentals.md) - "From Neural Networks to Transformers"

---

### Tokenization

**What it is:** Tokenization is the process of breaking text into discrete units called tokens.

**Physical Analogy:** Think of tokenization like **cutting a sentence into individual words**:
- Input: "The cat sat on the mat."
- After tokenization: ["The", "cat", "sat", "on", "the", "mat", "."]
- Each piece is now a separate token

**See:** [Chapter 1: Neural Network Fundamentals](01-neural-network-fundamentals.md) - "From Neural Networks to Transformers"

---

### Value (V)

**What it is:** One of three different representations of the same token. Value represents "What is my actual content?"

**Physical Analogy:** Think of Value like **the actual book content once you find it**:
- Once "cat" is identified as relevant, retrieve its meaning
- Physical: The **actual information** you retrieve

**See:** [Chapter 1: Neural Network Fundamentals](01-neural-network-fundamentals.md) - "From Neural Networks to Transformers"

---

### Vector

**What it is:** A vector is an ordered list of numbers, representing a point in space.

**Physical Analogy:** Think of a vector like **GPS coordinates**:
- `[0.3, 0.7]` = point at (0.3, 0.7) on a 2D map
- `[0.3, 0.7, -0.2]` = point in 3D space

**See:** [Chapter 1: Neural Network Fundamentals](01-neural-network-fundamentals.md) - "From Neural Networks to Transformers"

---

### Vocabulary

**What it is:** The vocabulary is the complete set of all possible tokens that a transformer can recognize and process.

**Physical Analogy:** Think of vocabulary like a **dictionary** or **Scrabble tile bag**:
- Contains all possible words/tokens the model knows
- Limited size (e.g., 50,000 tokens for GPT models)
- Each token has a unique ID/index

**See:** [Chapter 1: Neural Network Fundamentals](01-neural-network-fundamentals.md) - "From Neural Networks to Transformers"

---

### Weight

**What it is:** A weight is a parameter (number) in a matrix that determines how inputs are transformed.

**Physical Analogy:** Think of weights like **the strength of connections** in a network:
- High weight = strong connection (input has big influence)
- Low weight = weak connection (input has small influence)
- Negative weight = inhibitory connection (opposes the input)

**See:** [Chapter 1: Neural Network Fundamentals](01-neural-network-fundamentals.md) - "The Perceptron: The Basic Building Block"

---

### Weight Update

**What it is:** The process of changing matrix values (weights) based on gradients to improve predictions.

**Physical Analogy:** Think of weight updates like **tuning a radio**:
- Current setting (weight) = current frequency
- Static (loss) = how bad the signal is
- Gradient = which direction to turn the dial
- Weight update = actually turning the dial
- After many adjustments, you find the best frequency

**See:** [Chapter 1: Neural Network Fundamentals](01-neural-network-fundamentals.md) - "Training Loop: Putting It All Together"

---

**Navigation:**
- [← Introduction](00-introduction.md) | [← Index](00-index.md) | [← Previous: Appendix A](appendix-a-matrix-calculus.md) | [Next: Appendix C →](appendix-c-hand-calculation-tips.md)

