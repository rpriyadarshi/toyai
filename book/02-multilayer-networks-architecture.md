## Chapter 2: Multilayer Networks and Architecture

In the previous chapter, we explored the perceptron—a single neuron that can make simple decisions. While a single perceptron is limited, stacking many perceptrons together creates powerful networks capable of learning complex patterns. This chapter shows how multiple perceptrons form layers, how layers combine to create networks, and how to design effective network architectures.

We'll see how layers enable hierarchical learning—simple patterns in early layers building into complex patterns in deeper layers. We'll also explore practical questions about network design: how many layers are needed, which activation functions to use, and how feedforward networks create the expansion-compression pattern that's fundamental to modern architectures.

**Navigation:**
- [← Previous: Neural Networks and the Perceptron](01-neural-networks-perceptron.md) | [Table of Contents](00b-toc.md) | [Next: Learning Algorithms →](03-learning-algorithms.md)

---

## Layers: Stacking Neurons

A layer is a collection of neurons (perceptrons) that process information together in parallel. Think of a layer as a factory assembly line station: input arrives, the layer processes it, and output goes to the next layer. Each layer performs a specific job, and having multiple layers means multiple processing steps, each building on what the previous layer learned.

To understand how multiple perceptrons form a layer, consider a layer with $n$ perceptrons, each processing the same input vector $\mathbf{x} \in \mathbb{R}^d$. Each perceptron has its own weight vector $\mathbf{w}_i \in \mathbb{R}^d$ and bias $b_i \in \mathbb{R}$. The layer computes $n$ outputs simultaneously: $y_i = f(\mathbf{w}_i \cdot \mathbf{x} + b_i)$ for $i = 1, \ldots, n$. We can stack these weight vectors into a matrix $\mathbf{W} \in \mathbb{R}^{n \times d}$ where each row is a perceptron's weight vector, and stack the biases into a vector $\mathbf{b} \in \mathbb{R}^n$. The entire layer computation becomes $\mathbf{y} = f(\mathbf{W}\mathbf{x} + \mathbf{b})$, where $f$ is applied element-wise. This matrix formulation shows that a layer is essentially multiple dot products computed simultaneously—exactly what matrix multiplication does.

Neural networks typically have three types of layers. The input layer receives raw data and passes it into the network. Hidden layers process the information, and you can have as many of these as needed for the complexity of your problem. Finally, the output layer produces the final prediction.

To visualize this, imagine a multi-stage factory. Stage 1 (the input layer) is where raw materials arrive. Stages 2 through 4 (hidden layers) each refine the product in some way, with each stage building on the work of the previous stage. Stage 5 (the output layer) is where the finished product exits.

We trace through an example to illustrate how data flows through multiple layers. The values shown are arbitrary examples chosen to demonstrate the transformation process. This example matches the network architecture shown in the diagram above, with 3 input neurons, 4 neurons in the first hidden layer, 3 neurons in the second hidden layer, and 4 output neurons.

**Note:** This is an illustrative example. The weight matrices ($\mathbf{W}_1$, $\mathbf{W}_2$, $\mathbf{W}_{\text{out}}$) and bias vectors ($\mathbf{b}_1$, $\mathbf{b}_2$, $\mathbf{b}_{\text{out}}$) are not shown explicitly here—only the final outputs after applying the full computation $\mathbf{y} = f(\mathbf{W}\mathbf{x} + \mathbf{b})$ are displayed. This keeps the example concise while demonstrating the data flow through layers.

**Equation to solve:**
For each layer: $\mathbf{y} = f(\mathbf{W}\mathbf{x} + \mathbf{b})$

- Hidden Layer 1: $\mathbf{y}_1 = f\left(\mathbf{W}_1 \begin{bmatrix} 0.5 \\ 0.3 \\ 0.2 \end{bmatrix} + \mathbf{b}_1\right) = \begin{bmatrix} 0.3 \\ 0.7 \\ 0.4 \\ -0.2 \end{bmatrix}$
- Hidden Layer 2: $\mathbf{y}_2 = f\left(\mathbf{W}_2 \begin{bmatrix} 0.3 \\ 0.7 \\ 0.4 \\ -0.2 \end{bmatrix} + \mathbf{b}_2\right) = \begin{bmatrix} 0.4 \\ 0.6 \\ -0.1 \end{bmatrix}$
- Output Layer: $\mathbf{y}_{\text{out}} = f\left(\mathbf{W}_{\text{out}} \begin{bmatrix} 0.4 \\ 0.6 \\ -0.1 \end{bmatrix} + \mathbf{b}_{\text{out}}\right) = \begin{bmatrix} 0.1 \\ 0.8 \\ 0.05 \\ 0.05 \end{bmatrix}$

**Computation:**

1. **Input Layer**: Receives input data $\mathbf{x} = \begin{bmatrix} 0.5 \\ 0.3 \\ 0.2 \end{bmatrix}$ (example 3-dimensional input)

2. **Hidden Layer 1**: Transforms to vector $\mathbf{y}_1 = \begin{bmatrix} 0.3 \\ 0.7 \\ 0.4 \\ -0.2 \end{bmatrix}$ (expands to 4 dimensions)

3. **Hidden Layer 2**: Transforms to $\mathbf{y}_2 = \begin{bmatrix} 0.4 \\ 0.6 \\ -0.1 \end{bmatrix}$ (compresses to 3 dimensions)

4. **Output Layer**: Produces $\mathbf{y}_{\text{out}} = \begin{bmatrix} 0.1 \\ 0.8 \\ 0.05 \\ 0.05 \end{bmatrix}$ (expands to 4-dimensional output values)

Each layer transforms the data in a specific way. The input layer receives the raw input data, the first hidden layer converts it into a numerical representation, the second hidden layer refines that representation, and the output layer produces the final output values. The specific numerical values are illustrative; in a trained network, these would result from the learned weight matrices and activation functions.


| |
|:---:|
| ![Multi-Layer Neural Network: Input → Hidden Layers → Output](images/network-structure/multi-layer-network.svg) |


Layers enable complex transformations that a single perceptron cannot achieve. In algebra, this is like composing functions. If you have two functions $f(x)$ and $g(x)$, composing them gives you $f(g(x))$—you apply $g$ first, then apply $f$ to the result. Multiple layers work the same way: Layer 2 processes the output of Layer 1, Layer 3 processes the output of Layer 2, and so on. By stacking layers, networks can learn hierarchical patterns—simple patterns in early layers, and increasingly complex patterns in deeper layers. This hierarchical learning is what makes deep neural networks so powerful.

To see layers working together in detail, [Example 5: Feed-Forward Layers](13-example5-feedforward.md) demonstrates how information flows through multiple layers.

---

## Designing Multilayer Networks: Activation Functions and Depth

Now that we understand how layers work, two critical practical questions arise: **Do we use different activation functions in different layers to detect different curvatures?** And **How do we know if we have enough layers for our problem?** These questions get to the heart of network architecture design.

### Activation Function Selection Across Layers

**The Common Misconception:**

You might think: "If ReLU creates sharp corners and Sigmoid creates smooth S-curves, shouldn't I use different activation functions in different layers to detect different types of curvatures in my data?" This is a natural intuition, but it's actually a **misconception** about how neural networks work.

**The Reality: Most Networks Use the Same Activation Throughout**

In practice, **most modern neural networks use the same activation function (typically ReLU) throughout all hidden layers**. Here's why this works and why you don't need different activations for different "curvatures":

**1. Composition Creates Complexity, Not Individual Shapes:**

The key insight is that **the composition of non-linear functions creates complexity**, not the individual shape of each activation function. When you stack multiple layers with the same activation function (say, ReLU), each layer applies a different weight matrix. The composition of these transformations—even with the same activation function—creates arbitrarily complex patterns.

Think of it like this: A single ReLU layer can create sharp corners. But when you compose multiple ReLU layers with different weight matrices, you're not just adding more sharp corners—you're creating complex, curved decision boundaries that can approximate any continuous function. The **weights determine what patterns are learned**, not the activation function's shape.

**2. The Weights Handle Different Curvatures:**

Different "curvatures" in your data are captured by **different weight matrices**, not different activation functions. A network with ReLU throughout can learn:
- Sharp, angular patterns (through specific weight configurations)
- Smooth, curved patterns (through other weight configurations)
- Complex combinations of both (through the composition of multiple layers)

The activation function provides the non-linearity that makes this learning possible, but the specific patterns learned depend on the weights, which are optimized during training.

**3. Simplicity and Empirical Effectiveness:**

Using the same activation function throughout:
- **Simplifies training**: You don't need to tune different activation functions for different layers
- **Is empirically effective**: Modern deep learning practice shows that ReLU (or variants like GELU, Swish) throughout hidden layers works extremely well
- **Reduces hyperparameter space**: Fewer choices to make means less chance of making poor architectural decisions

**When Different Activations Are Actually Used:**

There are legitimate cases where different activation functions appear in different parts of the network, but they're **not for detecting different curvatures**:

1. **Output layers**: Task-specific functions are used here:
   - **Sigmoid** for binary classification (outputs probabilities between 0 and 1)
   - **Softmax** for multi-class classification (outputs probability distributions over classes)
   - These are chosen for their mathematical properties (probabilistic outputs), not for "curvature detection"

2. **Historical architectures**: Some older networks (from the 1990s-2000s) used Tanh in early layers and ReLU in later layers, but this was largely due to:
   - Limited understanding of activation functions at the time
   - Hardware constraints (Tanh was easier to compute on older hardware)
   - Modern practice has largely converged on ReLU (or variants) throughout

3. **Specialized architectures**: Some modern architectures use different activations for specific reasons:
   - **GELU** (Gaussian Error Linear Unit, $f(x) = x \cdot \Phi(x)$ where $\Phi$ is the standard normal CDF) in some transformer models—a smooth approximation of ReLU
   - **Swish** ($f(x) = x \cdot \sigma(x)$ where $\sigma$ is sigmoid) in some vision models—a smooth, self-gated activation
   - But these are typically used **consistently** throughout the network, not mixed

**The Bottom Line:**

You don't need different activation functions to detect different curvatures. The same activation function (ReLU), when composed across multiple layers with different learned weights, can create arbitrarily complex patterns. The weights do the work of learning what patterns to detect; the activation function just provides the necessary non-linearity.

### Determining Sufficient Network Depth

**The Challenge:**

How do you know if your network has enough layers to solve your problem? Too few layers and the network **underfits** (can't learn the patterns). Too many layers and the network **overfits** (memorizes training data but fails to generalize). This is one of the most practical questions in neural network design.

**The Empirical Approach: Start Small and Grow**

The most practical method is **empirical**: start with a small network and add layers until you find the right balance. Here's the process:

1. **Start small**: Begin with 1-2 hidden layers and a moderate number of neurons per layer (e.g., 64-128 neurons)
2. **Train and evaluate**: Train on your training set and evaluate on a **validation set** (data held out from training)
3. **Monitor both losses**: Track both training loss and validation loss
4. **Add capacity if underfitting**: If both training and validation losses are high and plateau, add layers or neurons
5. **Stop when validation performance plateaus**: When adding layers stops improving validation performance, you've likely reached sufficient depth
6. **Watch for overfitting**: If training loss decreases but validation loss increases, you've gone too far—reduce capacity or add regularization

**Signs of Insufficient Capacity (Underfitting):**

Your network needs more layers or neurons if you see:
- **High training loss**: The network can't even learn the training data well
- **High validation loss**: The network can't generalize to new examples
- **Both losses plateau at high values**: Adding more training doesn't help
- **Simple patterns missed**: The network fails to capture obvious patterns in the data

**Example**: For digit recognition, if a 1-layer network consistently misclassifies digits that humans can easily distinguish, it likely needs more capacity.

**Signs of Sufficient Capacity:**

Your network has enough layers when:
- **Training loss decreases to low values**: The network can learn the training patterns
- **Validation loss decreases and stabilizes**: The network generalizes well to new examples
- **Good test performance**: Performance on held-out test data matches validation performance
- **Adding layers doesn't help**: Further increases in depth don't improve validation performance

**Signs of Excessive Capacity (Overfitting):**

Your network has too many layers if you see:
- **Training loss very low, validation loss much higher**: Large gap indicates memorization
- **Validation loss increases while training loss decreases**: Classic overfitting signature
- **Poor generalization**: Network performs well on training data but poorly on new examples
- **Large gap between training and validation metrics**: Accuracy, F1-score, or other metrics diverge significantly

**Theoretical Bounds: Mathematical Foundations**

The **universal approximation theorem** (proven by Cybenko in 1989 and Hornik et al. in 1991) establishes that a **single hidden layer with enough neurons** can approximate any continuous function to arbitrary accuracy. This mathematical result demonstrates that, from a representational capacity perspective, multiple layers are not strictly necessary—a single sufficiently wide layer is theoretically sufficient.

However, **deeper networks are often more efficient**:
- **Fewer total parameters**: A deep network with fewer neurons per layer can often achieve the same performance as a shallow network with many neurons per layer
- **Hierarchical learning**: Depth enables the natural hierarchical feature learning we discussed (simple patterns → complex patterns)
- **Better generalization**: In practice, deeper networks often generalize better than wide shallow networks

**Practical Guidelines: How Many Layers for Your Problem?**

While there's no universal formula, here are practical guidelines based on problem complexity:

- **Simple problems** (linear/logistic regression, simple classification):
  - **0-1 hidden layers** typically suffice
  - Example: Predicting house prices from square footage and number of bedrooms

- **Moderate complexity** (image classification, natural language processing, most practical ML tasks):
  - **2-10 hidden layers** is common
  - Example: Handwritten digit recognition (MNIST) typically needs 2-3 hidden layers
  - Example: Text classification often uses 2-5 layers

- **High complexity** (large-scale vision, language models, complex pattern recognition):
  - **10-100+ layers** may be needed
  - Example: ImageNet classification (ResNet uses 18-152 layers)
  - Example: Modern language models (GPT, BERT) use 12-100+ transformer layers

**Factors to Consider:**

When deciding on network depth, consider:

1. **Problem complexity**: More complex problems (more classes, more variation, more features) generally need deeper networks
2. **Dataset size**: Larger datasets can support deeper networks without overfitting
3. **Computational resources**: Deeper networks require more memory and training time
4. **Training time**: Deeper networks take longer to train
5. **Inference requirements**: Deeper networks are slower at inference (making predictions)

**The Iterative Process:**

In practice, network architecture design is **iterative**:
1. Start with a baseline (e.g., 2-3 layers for a moderate problem)
2. Train and evaluate
3. If underfitting: add layers or neurons, try again
4. If overfitting: reduce layers, add regularization (dropout, weight decay), or get more training data
5. Repeat until you find the sweet spot where validation performance is good and stable

**Concrete Example: Digit Recognition**

For handwritten digit recognition (10 classes, 28×28 pixel images):
- **Too shallow** (1 layer): Might achieve 85-90% accuracy, misses complex digit variations
- **Sufficient** (2-3 layers): Typically achieves 95-98% accuracy, good generalization
- **Too deep** (10+ layers): Might achieve 99%+ on training but only 96% on validation (overfitting)

The "right" depth depends on your specific dataset, but 2-3 hidden layers is a good starting point for this problem.

**Key Takeaway:**

There's no magic formula for network depth. The right number of layers depends on your problem, your data, and your computational resources. The empirical approach—starting small, monitoring training and validation performance, and iteratively adjusting—is the most reliable method. Mathematical analysis (specifically, the theory of function approximation and empirical studies of network efficiency) demonstrates that depth enables efficient hierarchical learning with fewer total parameters than wide shallow networks. Empirical evidence from practice shows that 2-10 layers work well for most problems, with deeper networks needed for highly complex tasks.

---

## Feedforward Networks: Multi-Layer Perceptrons

A **feed-forward network** (FFN) is a type of layer that applies two linear transformations with an activation function in between. This creates a two-stage processing pipeline: first expanding the dimensions to give the network more room to work, then compressing back to the original dimensions. The activation function in between adds crucial non-linearity that enables the network to learn complex patterns.

A feed-forward network is defined mathematically as:

$$\text{FFN}(\mathbf{x}) = f(\mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2$$

Let's break down each component of this equation:

**1. The input ($\mathbf{x}$):** The input is a $d$-dimensional vector ($\mathbf{x} \in \mathbb{R}^d$), where $d$ is the number of input features.

**2. The expansion phase ($\mathbf{x}\mathbf{W}_1 + \mathbf{b}_1$):** 
- $\mathbf{W}_1$ is a weight matrix of size $d \times d'$ (meaning it has $d$ rows and $d'$ columns), where $d' > d$ (typically $d' = 4d$ in practice). This matrix **expands** the input from dimension $d$ to dimension $d'$, giving the network more "room" to work with the information.
- $\mathbf{b}_1$ is a bias vector with $d'$ components ($\mathbf{b}_1 \in \mathbb{R}^{d'}$), which shifts the expanded representation.
- The result is a $d'$-dimensional vector (larger than the input).

**3. The activation function ($f()$):** A non-linear activation function (like ReLU) is applied **element-wise** (meaning it's applied to each component of the vector separately). This adds crucial **non-linearity** that enables the network to learn complex patterns. Without this non-linearity, the two linear transformations would collapse into a single linear transformation.

**4. The compression phase ($f(\mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2$):**
- $\mathbf{W}_2$ is a weight matrix of size $d' \times d$ (meaning it has $d'$ rows and $d$ columns), which **compresses** the representation back from dimension $d'$ to dimension $d$.
- $\mathbf{b}_2$ is a bias vector with $d$ components ($\mathbf{b}_2 \in \mathbb{R}^{d}$), which provides the final offset.
- The result is a $d$-dimensional vector (same size as the input, but transformed).

**Why the expansion-compression pattern works:** The expansion factor $d' > d$ (typically $d' = 4d$) gives the network more **capacity** (more neurons) to learn complex feature combinations in the higher-dimensional space. The compression phase then brings it back to the original dimension, but the information has been transformed in a meaningful way through the non-linear activation function.

In practice, ReLU is the most commonly used activation function for feed-forward networks in transformers (and is what we use in our examples), though some models use alternatives like GELU (Gaussian Error Linear Unit). The key requirement is that the activation function must be non-linear—without it, the two linear transformations would collapse into a single linear transformation, losing the network's ability to learn complex patterns.

Here, $\mathbf{W}_1$ expands the input from dimension $d$ to dimension $d'$, $\mathbf{b}_1$ shifts the expanded representation, the activation function $f()$ adds non-linearity, $\mathbf{W}_2$ compresses back to dimension $d$, and $\mathbf{b}_2$ provides the final offset. The expansion phase allows the network to learn complex feature combinations in the higher-dimensional space, while the compression ensures the output has the correct shape for the next layer.

To understand this intuitively, think of a feed-forward network as a two-stage transformation: expansion followed by compression. You start with a small package (a $d$-dimensional vector). In stage 1, you expand it into a large box (a $d'$-dimensional vector where $d' > d$), giving you more room to work with the information. The activation function then filters and processes the contents, introducing non-linearity. In stage 2, you compress it back down to a small package (back to $d$ dimensions), but now it's been transformed in a meaningful way. This expansion-compression pattern is fundamental to how many modern neural network architectures process information.

We trace through a numerical example using arbitrary values chosen to illustrate the expansion-compression pattern:

**Equation to solve (using ReLU activation):**
$$\text{FFN}\left(\begin{bmatrix} 0.3 \\ 0.7 \end{bmatrix}\right) = \text{ReLU}\left(\begin{bmatrix} 0.3 \\ 0.7 \end{bmatrix}\mathbf{W}_1 + \mathbf{b}_1\right)\mathbf{W}_2 + \mathbf{b}_2 = \begin{bmatrix} 0.4 \\ 0.6 \end{bmatrix}$$

**Computation:**

1. **Input**: $\mathbf{x} = \begin{bmatrix} 0.3 \\ 0.7 \end{bmatrix}$ (dimension 2, example input values)

2. **First transformation** (using $\mathbf{W}_1$, a $2 \times 4$ matrix):
   $$\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1 = \begin{bmatrix} 0.5 \\ 0.2 \\ 0.8 \\ 0.1 \end{bmatrix}$$
   
   This expands the input to 4 dimensions.

3. **Apply ReLU activation**:
   $$\text{ReLU}\left(\begin{bmatrix} 0.5 \\ 0.2 \\ 0.8 \\ 0.1 \end{bmatrix}\right) = \begin{bmatrix} 0.5 \\ 0.2 \\ 0.8 \\ 0.1 \end{bmatrix}$$
   
   Since all values are non-negative, ReLU preserves them unchanged.

4. **Second transformation** (using $\mathbf{W}_2$, a $4 \times 2$ matrix):
   $$\text{ReLU}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) \mathbf{W}_2 + \mathbf{b}_2 = \begin{bmatrix} 0.4 \\ 0.6 \end{bmatrix}$$
   
   This compresses the representation back to dimension 2.

The input starts as a 2-dimensional vector, expands to 4 dimensions in the middle (where the network can learn complex feature combinations), and then compresses back to 2 dimensions. The expansion gives the network capacity to learn, while the compression ensures the output has the correct shape for the next layer. The specific numerical values are illustrative; in practice, these would be determined by the learned weight matrices.


| |
|:---:|
| ![Feed-Forward Network Structure](images/network-structure/ffn-structure.svg) |


Feed-forward networks are crucial because they add both non-linearity and capacity to neural networks. The expansion phase allows the network to learn complex feature combinations that wouldn't be possible with just linear transformations. This is why FFNs are a core component of many modern neural network architectures.

To see feed-forward networks in action, [Example 5: Feed-Forward Layers](13-example5-feedforward.md) demonstrates the complete FFN computation step by step.

---

## Chapter Summary

In this chapter, we've seen how multiple perceptrons combine to form layers, and how layers stack to create powerful networks. The key insights are:

1. **Layers enable hierarchical learning**: Simple patterns in early layers build into complex patterns in deeper layers.

2. **Architecture design is practical**: Use the same activation function throughout (typically ReLU), and determine depth empirically by monitoring training and validation performance.

3. **Feedforward networks use expansion-compression**: The expansion phase gives the network capacity to learn complex feature combinations, while compression ensures the output has the correct shape.

Now that we understand network architecture, we need to understand how networks learn. In the next chapter, we'll explore the learning algorithms that enable networks to improve their predictions: loss functions, gradient descent, and backpropagation.

---

**Navigation:**
- [← Previous: Neural Networks and the Perceptron](01-neural-networks-perceptron.md) | [← Index](00b-toc.md) | [Next: Learning Algorithms →](03-learning-algorithms.md)

