## Chapter 1: Neural Network Fundamentals

This chapter establishes the foundation for understanding transformers by first explaining what neural networks are, how they work, and why they form the basis for modern language models. Throughout this chapter, we explain every concept using physical analogies that connect abstract mathematics to tangible reality, making the material more accessible and memorable.

**Navigation:**
- [← Introduction](00-introduction.md) | [Table of Contents](00-index.md) | [Next: The Matrix Core →](02-matrix-core.md)

### Why This Chapter First?

Most AI education jumps straight to transformers without explaining the fundamental system they're built on. This approach leaves students with a superficial understanding—they can describe what transformers do, but they don't understand how or why they work. 

This chapter takes a different approach. We build understanding from the ground up, starting with a single neuron and progressing through complete neural networks to transformers. Every concept connects to something you can visualize and understand. This foundation is essential because transformers are, at their core, sophisticated neural networks. To truly understand transformers, you must first understand the system they're built upon.

**Learning Principle:** Understand the system before you study its components.

---

## What Are Neural Networks?

Neural networks are computational systems inspired by how the brain processes information. To understand how they work, imagine a factory assembly line. Raw materials arrive at the input, pass through multiple assembly stations (called layers), and emerge as a finished product at the output. What makes this factory special is that it learns from examples. When you show it many examples—like "when I see 'cat', predict 'sat'"—the factory adjusts its machines (which we call weights) to get better at making predictions. After processing many examples, it learns the underlying patterns and can then make predictions on new inputs it has never seen before.

Neural networks matter because they can learn complex patterns from data without being explicitly programmed for each pattern. This ability to learn from examples makes them the foundation of modern AI, including the transformer models we'll study in later chapters. They excel at tasks where the rules are too complex to write down explicitly, but where we have many examples of the desired behavior.

![Neural Network Structure: Factory Assembly Line](images/network-structure/neural-network-structure.svg)

To see a neural network in action, [Example 1: Minimal Forward Pass](06-example1-forward-pass.md) demonstrates how a transformer makes predictions step by step through the forward pass.

---

## The Perceptron: The Basic Building Block

A perceptron is a single neuron—the simplest possible neural network. It functions as a decision-making unit that takes multiple inputs, weighs their importance, and produces a single output decision. While a single perceptron is quite limited in what it can learn, understanding how it works provides the foundation for understanding more complex networks.

### The Mathematical Story

Mathematically, a perceptron computes its output using the following formula:

$$y = f(\mathbf{w} \cdot \mathbf{x} + b)$$

where $\mathbf{x} \in \mathbb{R}^d$ is the input vector, $\mathbf{w} \in \mathbb{R}^d$ is the weight vector, $b \in \mathbb{R}$ is the bias scalar, $f: \mathbb{R} \to \mathbb{R}$ is the activation function, and $y \in \mathbb{R}$ is the output scalar.

In this equation, $\mathbf{x}$ represents the input, which is a vector of $d$ numbers. The variable $\mathbf{w}$ represents the weights, which determine how important each input is. The bias $b$ provides a baseline offset that shifts the entire computation. The function $f()$ is called the activation function, and it shapes the output in a specific way. Finally, $y$ is the output—the decision made by the perceptron.

**Notation Note:** When we write a vector like $\begin{bmatrix} 1 \\ 0 \end{bmatrix}$, this represents a single vector with two components (the first component is 1, the second is 0). This is different from a system of equations—it is one mathematical object (a vector), not multiple equations. The vertical arrangement is simply the standard mathematical notation for column vectors.

### Example: Computing Perceptron Output

We now illustrate the perceptron computation using a numerical example with a 2-dimensional input vector. For pedagogical clarity, we choose simple input values that make the arithmetic easy to follow: $\mathbf{x} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$, which is a unit basis vector where only the first component is non-zero. This choice simplifies the calculation while demonstrating the core computation.

**Given:**
- Input vector: $\mathbf{x} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$ (chosen for simplicity: only the first component is active, making it easy to trace how each component contributes)
- Weight vector: $\mathbf{w} = \begin{bmatrix} 0.1 \\ 0.2 \end{bmatrix}$ (arbitrary example values chosen to demonstrate the computation)
- Bias: $b = 0.05$ (arbitrary example value)
- Activation function: $f(x) = \text{ReLU}(x) = \max(0, x)$

**Equation to solve:**
$$y = f\left(\begin{bmatrix} 0.1 \\ 0.2 \end{bmatrix} \cdot \begin{bmatrix} 1 \\ 0 \end{bmatrix} + 0.05\right) = \text{ReLU}\left(\begin{bmatrix} 0.1 \\ 0.2 \end{bmatrix} \cdot \begin{bmatrix} 1 \\ 0 \end{bmatrix} + 0.05\right)$$

**Computation:**

1. **Compute the dot product** $\mathbf{w} \cdot \mathbf{x}$:
   $$\mathbf{w} \cdot \mathbf{x} = w_1 \cdot x_1 + w_2 \cdot x_2 = 0.1 \times 1 + 0.2 \times 0 = 0.1 + 0 = 0.1$$

   **Important Note on Dot Products:** The dot product of two vectors always produces a **scalar** (a single number), not a vector. This is because we multiply corresponding components and then **sum** them together. The formula $\mathbf{w} \cdot \mathbf{x} = \sum_{i=1}^d w_i x_i$ shows this: we multiply each $w_i$ by $x_i$, then add all the products together, resulting in a single number. This weighted sum is exactly what we need for a perceptron, which produces a single output value.

2. **Add the bias term**:
   $$\mathbf{w} \cdot \mathbf{x} + b = 0.1 + 0.05 = 0.15$$

3. **Apply the activation function**:
   $$y = f(0.15) = \text{ReLU}(0.15) = \max(0, 0.15) = 0.15$$

**Result:** The perceptron produces output $y = 0.15$ (a scalar, single number).

**Interpretation:** We chose the input vector $\begin{bmatrix} 1 \\ 0 \end{bmatrix}$ because it simplifies the calculation—only the first component is non-zero, making it easy to see how that component contributes to the output. The first component (value 1) is multiplied by weight 0.1, contributing 0.1 to the weighted sum. The second component (value 0) is multiplied by weight 0.2, contributing 0 to the weighted sum (demonstrating that zero inputs produce zero contribution regardless of the weight value). The bias term adds 0.05 to the result. After applying the ReLU activation function, which preserves non-negative values, the final output is 0.15.

### Example: Alternative Input

To demonstrate how different inputs produce different outputs, we now consider the same perceptron with input $\mathbf{x} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$, which is the complementary unit basis vector (only the second component is non-zero). This choice allows us to compare how the same perceptron responds to different input patterns while keeping the arithmetic simple.

**Equation to solve:**
$$y = f\left(\begin{bmatrix} 0.1 \\ 0.2 \end{bmatrix} \cdot \begin{bmatrix} 0 \\ 1 \end{bmatrix} + 0.05\right) = \text{ReLU}\left(\begin{bmatrix} 0.1 \\ 0.2 \end{bmatrix} \cdot \begin{bmatrix} 0 \\ 1 \end{bmatrix} + 0.05\right)$$

**Computation:**

1. Dot product: $\mathbf{w} \cdot \mathbf{x} = 0.1 \times 0 + 0.2 \times 1 = 0.2$
2. Add bias: $0.2 + 0.05 = 0.25$
3. Apply ReLU: $y = \max(0, 0.25) = 0.25$

This example demonstrates that the same perceptron produces different outputs for different input vectors. By using the unit basis vectors $\begin{bmatrix} 1 \\ 0 \end{bmatrix}$ and $\begin{bmatrix} 0 \\ 1 \end{bmatrix}$, we can clearly see how each input component contributes independently to the output, enabling the network to distinguish between different input patterns based on their component values.

You might notice that the core computation $\mathbf{w} \cdot \mathbf{x} + b$ looks very familiar—it's closely related to the equation of a straight line! In algebra, we write a line as $y = mx + c$, where $m$ is the slope and $c$ is the y-intercept. For a perceptron with a single input ($d=1$), $\mathbf{w} \cdot \mathbf{x} + b$ becomes $wx + b$, which is exactly $y = mx + c$ (where $w$ is the slope and $b$ is the intercept). For multiple inputs ($d > 1$), $\mathbf{w} \cdot \mathbf{x} = \sum_{i=1}^d w_i x_i$ is the dot product (weighted sum), which generalizes the line equation to multiple dimensions. The bias $b$ still shifts the entire computation up or down, just like the y-intercept shifts a line.

**Understanding Dot Products vs. Matrix Multiplication:**

It's important to understand the difference between a dot product and matrix multiplication, as this distinction is crucial for understanding how neural networks work:

- **Dot product** ($\mathbf{w} \cdot \mathbf{x}$): Takes two vectors of the same dimension and produces a **scalar** (single number). The formula is $\mathbf{w} \cdot \mathbf{x} = \sum_{i=1}^d w_i x_i = w_1 x_1 + w_2 x_2 + \cdots + w_d x_d$. This is a weighted sum—we multiply each component of $\mathbf{w}$ by the corresponding component of $\mathbf{x}$, then add all the products together. The result is always a single number.

- **Matrix multiplication** ($\mathbf{W}\mathbf{x}$): Takes a matrix and a vector and produces a **vector**. If $\mathbf{W}$ is an $n \times d$ matrix and $\mathbf{x}$ is a $d$-dimensional vector, then $\mathbf{W}\mathbf{x}$ is an $n$-dimensional vector. Each element of the result is computed as a dot product: the $i$-th element of $\mathbf{W}\mathbf{x}$ is the dot product of the $i$-th row of $\mathbf{W}$ with $\mathbf{x}$.

**Why We Use Dot Products in Perceptrons:**

A single perceptron produces one output value (a scalar), so we need a dot product, not matrix multiplication. The dot product $\mathbf{w} \cdot \mathbf{x}$ takes the $d$-dimensional input vector $\mathbf{x}$ and the $d$-dimensional weight vector $\mathbf{w}$, and produces a single number. This single number is then passed through the activation function to produce the final output.

**When We Use Matrix Multiplication:**

When we have multiple perceptrons (a layer), we use matrix multiplication. If we have $n$ perceptrons, each with its own weight vector $\mathbf{w}_i$, we can stack these weight vectors into a matrix $\mathbf{W}$ where each row is a weight vector. Then $\mathbf{W}\mathbf{x}$ computes all $n$ dot products simultaneously, producing an $n$-dimensional output vector (one value per perceptron). This is exactly what happens in a layer: multiple dot products computed in parallel via matrix multiplication.

**Concrete Example:**

- **Single perceptron**: $\mathbf{w} = \begin{bmatrix} 0.1 \\ 0.2 \end{bmatrix}$, $\mathbf{x} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$
  - Dot product: $\mathbf{w} \cdot \mathbf{x} = 0.1 \times 1 + 0.2 \times 0 = 0.1$ (scalar)
  
- **Layer with 2 perceptrons**: $\mathbf{W} = \begin{bmatrix} 0.1 & 0.2 \\ 0.3 & 0.4 \end{bmatrix}$, $\mathbf{x} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$
  - Matrix multiplication: $\mathbf{W}\mathbf{x} = \begin{bmatrix} 0.1 \times 1 + 0.2 \times 0 \\ 0.3 \times 1 + 0.4 \times 0 \end{bmatrix} = \begin{bmatrix} 0.1 \\ 0.3 \end{bmatrix}$ (vector)
  - Notice: Each row of $\mathbf{W}$ is dotted with $\mathbf{x}$, producing one element of the result vector

To understand what this generalization means geometrically, think of it this way: $y = mx + c$ defines a straight line in 2D space (the x-y plane). When we move to multiple dimensions with $\mathbf{w} \cdot \mathbf{x} + b$, we're not creating "a bunch of lines"—we're creating a single geometric object called a **hyperplane**. The equation $\mathbf{w} \cdot \mathbf{x} + b = 0$ defines a decision boundary that divides the input space:

- **1D input ($d=1$)**: $w_1 x_1 + b = 0$ defines a point on a number line (dividing positive from negative)
- **2D input ($d=2$)**: $w_1 x_1 + w_2 x_2 + b = 0$ defines a straight line in the $(x_1, x_2)$ plane (dividing the plane into two regions)
- **3D input ($d=3$)**: $w_1 x_1 + w_2 x_2 + w_3 x_3 + b = 0$ defines a plane in 3D space (dividing space into two regions)
- **nD input ($d=n$)**: $\mathbf{w} \cdot \mathbf{x} + b = 0$ defines a hyperplane in $n$-dimensional space

All points on one side of the hyperplane satisfy $\mathbf{w} \cdot \mathbf{x} + b > 0$, while all points on the other side satisfy $\mathbf{w} \cdot \mathbf{x} + b < 0$. This is why the vector form is so powerful: it's the same mathematical structure (a hyperplane) regardless of dimension, just like how a line in 2D, a plane in 3D, and a hyperplane in higher dimensions are all the same type of geometric object—they're all flat surfaces that divide space.

| | | |
|:---:|:---:|:---:|
| ![1D Hyperplane: Decision Point](images/other/hyperplane-1d.svg) | ![2D Hyperplane: Decision Line](images/other/hyperplane-2d.svg) | ![3D Hyperplane: Decision Plane](images/other/hyperplane-3d.svg) |

The key difference is that the perceptron then applies an activation function $f()$ to this linear combination. If $f()$ is the identity function (just returns its input unchanged), then the perceptron is computing a linear function—a straight line (or hyperplane in higher dimensions). But with other activation functions, we get non-linear transformations that enable the network to learn complex, curved patterns that a simple straight line cannot represent.

To see this transformation in action, consider what happens when we apply different activation functions to straight lines. The graphs below show four different linear functions (y = 2x + 1, y = -x + 2, y = 0.5x - 1, and y = -1.5x + 0.5) and how they are transformed by three common activation functions: ReLU, Sigmoid, and Tanh. (We'll define these functions precisely in a moment, but for now, notice their visual effects.) ReLU zeros out all negative values, creating sharp corners where lines cross zero. Sigmoid squashes everything into the 0-1 range, creating smooth S-shaped curves. Tanh does something similar but squashes to the -1 to 1 range, preserving the sign of the original values. These transformations are what allow neural networks to learn non-linear patterns—without them, the network would only be able to compute straight lines.

| | |
|:---:|:---:|
| ![Linear Functions](images/activation-functions/examples/linear-function.svg) | ![After ReLU](images/activation-functions/examples/relu-applied.svg) |
| ![After Sigmoid](images/activation-functions/examples/sigmoid-applied.svg) | ![After Tanh](images/activation-functions/examples/tanh-applied.svg) |

To understand this intuitively, think of a perceptron as a simple voting system. The inputs are like votes: [vote1, vote2, vote3]. The weights determine the strength of each vote: [0.8, 0.2, 0.5] means vote1 is most important. The bias adds a baseline value, like always adding +0.1 regardless of the votes. The activation function then shapes the result, perhaps by saying "if the total exceeds 0.5, output YES."

We illustrate this with a numerical example using arbitrary values chosen to demonstrate the computation:

**Equation to solve:**
$$y = f\left(\begin{bmatrix} 0.8 \\ 0.2 \\ 0.5 \end{bmatrix} \cdot \begin{bmatrix} 1.0 \\ 0.5 \\ 0.3 \end{bmatrix} + 0.1\right) = \text{ReLU}\left(\begin{bmatrix} 0.8 \\ 0.2 \\ 0.5 \end{bmatrix} \cdot \begin{bmatrix} 1.0 \\ 0.5 \\ 0.3 \end{bmatrix} + 0.1\right)$$

**Given:**
- Input vector: $\mathbf{x} = \begin{bmatrix} 1.0 \\ 0.5 \\ 0.3 \end{bmatrix}$ (example 3-dimensional input vector)
- Weight vector: $\mathbf{w} = \begin{bmatrix} 0.8 \\ 0.2 \\ 0.5 \end{bmatrix}$ (example weight values)
- Bias: $b = 0.1$ (example bias value)
- Activation function: $f(x) = \text{ReLU}(x) = \max(0, x)$

**Computation:**

1. **Compute the weighted sum** $\mathbf{w} \cdot \mathbf{x} + b$:
   $$\mathbf{w} \cdot \mathbf{x} + b = w_1 x_1 + w_2 x_2 + w_3 x_3 + b = 0.8 \times 1.0 + 0.2 \times 0.5 + 0.5 \times 0.3 + 0.1 = 0.8 + 0.1 + 0.15 + 0.1 = 1.15$$

2. **Apply the activation function**:
   $$y = f(1.15) = \text{ReLU}(1.15) = \max(0, 1.15) = 1.15$$

**Result:** The perceptron produces output $y = 1.15$.

This example demonstrates that each input is multiplied by its corresponding weight, the products are summed together, the bias is added, and then the activation function is applied to produce the final output. The specific values are chosen arbitrarily for illustration; in practice, these would be learned during training.

### Understanding the Components

We now examine each component of the perceptron in detail, as understanding these building blocks is essential for grasping how more complex networks operate.

**Weight ($\mathbf{w}$):** Weights determine the strength of connections in a network. A high weight creates a strong connection, meaning that input has a large influence on the output. A low weight creates a weak connection, giving that input only a small influence. When a weight is negative, it creates an inhibitory connection that opposes the input rather than supporting it. During training, the network learns which weights to assign to each input based on how well those weights help it make correct predictions.

**Numerical Example:** Consider a weight vector $\mathbf{w} = \begin{bmatrix} 0.1 \\ 0.2 \end{bmatrix}$:
- The first input component is scaled by weight 0.1 (moderate influence)
- The second input component is scaled by weight 0.2 (stronger influence, contributing twice as much per unit input)
- For comparison, a weight vector $\mathbf{w} = \begin{bmatrix} 0.5 \\ -0.3 \end{bmatrix}$ would give the first component strong positive influence (0.5), while the second component would have inhibitory influence (-0.3), reducing the output when that input is positive

**Computation Example:** Using the same simple input vector $\mathbf{x} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$ and weights $\mathbf{w} = \begin{bmatrix} 0.1 \\ 0.2 \end{bmatrix}$:
- First component contribution: $w_1 \times x_1 = 0.1 \times 1 = 0.1$
- Second component contribution: $w_2 \times x_2 = 0.2 \times 0 = 0$
- Total weighted input: $0.1 + 0 = 0.1$

This demonstrates that weights only affect the output when their corresponding input components are non-zero. The weight value 0.2 has no effect in this case because the corresponding input component is zero. We chose $\begin{bmatrix} 1 \\ 0 \end{bmatrix}$ specifically to make this property clear—by having one component be zero, we can isolate the effect of the other component.

Weights are typically initialized to small random values (e.g., sampled from a normal distribution with mean 0 and standard deviation 0.01) to break symmetry. If all weights start at zero, all neurons in a layer would compute identical outputs and learn identical features, which would waste capacity. Random initialization ensures each neuron starts with different weights, enabling the network to learn diverse features.

**Bias ($b$):** The bias acts as a baseline or offset that shifts the entire computation up or down. In algebra, this is like translating a graph: if you have $y = f(x)$ and you add a constant $c$ to get $y = f(x) + c$, the entire graph shifts up or down by $c$ units. The bias does the same thing—it shifts the entire function up or down. Think of it like setting a scale to zero before weighing something, or adjusting a thermostat's baseline temperature. The bias allows the perceptron to make decisions even when all inputs are zero, and it provides flexibility in how the decision boundary is positioned. Mathematically, without bias, a perceptron with all-zero inputs always outputs $f(0) = f(\mathbf{w} \cdot \mathbf{0}) = f(0)$, which severely limits expressivity. The bias term enables the network to learn decision boundaries that don't pass through the origin.

**Numerical Example:** Consider bias $b = 0.05$:
- Given weighted sum 0.1, the value before activation becomes $0.1 + 0.05 = 0.15$
- Given weighted sum 0.0, the value before activation becomes $0.0 + 0.05 = 0.05$ (positive output even with zero weighted input)
- With bias $b = -0.1$, the same weighted sum of 0.1 becomes $0.1 + (-0.1) = 0.0$

**Comparison Example:** To illustrate how bias affects the output, consider two perceptrons with identical weights $\mathbf{w} = \begin{bmatrix} 0.1 \\ 0.2 \end{bmatrix}$ and the same simple input $\mathbf{x} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$ (chosen to keep the calculation straightforward):
- Perceptron A with $b = 0.05$: output = ReLU(0.1 + 0.05) = ReLU(0.15) = 0.15
- Perceptron B with $b = -0.1$: output = ReLU(0.1 + (-0.1)) = ReLU(0.0) = 0.0

The bias parameter shifts the decision threshold, controlling the perceptron's sensitivity to input variations.

**Activation Function ($f()$):** The activation function acts as a filter that shapes the signal. Without an activation function, a network can only perform linear transformations, which severely limits what it can learn. With an activation function, the network gains the ability to learn complex, non-linear patterns. Different activation functions create different "shapes" of transformation, each suited to different types of problems.

**Numerical Example:** Consider the pre-activation value 0.15 (weighted sum + bias):
- **ReLU**: $f(0.15) = \max(0, 0.15) = 0.15$ (preserves non-negative values)
- **ReLU with negative input**: For input $-0.1$, ReLU produces $\max(0, -0.1) = 0$ (rectifies negative values to zero)
- **Sigmoid**: $f(0.15) = \frac{1}{1+e^{-0.15}} \approx 0.537$ (maps to the interval $(0, 1)$)
- **Tanh**: $f(0.15) = \tanh(0.15) \approx 0.149$ (maps to the interval $(-1, 1)$, preserving the sign of the input)

**Comparison Example:** To compare different activation functions, we use the same input $\mathbf{x} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$ (chosen for simplicity), weights $\mathbf{w} = \begin{bmatrix} 0.1 \\ 0.2 \end{bmatrix}$, and bias $b = 0.05$:
- Pre-activation value: $0.1 + 0.05 = 0.15$
- **With ReLU activation**: output = $\max(0, 0.15) = 0.15$
- **With Sigmoid activation**: output = $\frac{1}{1+e^{-0.15}} \approx 0.537$
- **With identity (no activation)**: output = $0.15$ (linear transformation only)

This comparison demonstrates that different activation functions produce distinct outputs from the same pre-activation value, enabling networks to learn different types of non-linear patterns.

Activation functions must be (at least piecewise) differentiable for gradient descent to work. They introduce non-linearity, which is mathematically necessary for learning complex patterns. It can be proven (via the universal approximation theorem) that neural networks with non-linear activation functions can approximate any continuous function, given sufficient capacity. Without activation functions, multiple layers would collapse into a single linear layer, losing the hierarchical learning capability that makes deep networks powerful.

### How Activation Functions Enable Complex Pattern Learning

To understand why activation functions are essential for learning complex patterns, we need to see what happens when we stack multiple layers—both with and without activation functions.

**Why Linear Layers Collapse:**

Consider a network with two layers, both using the identity function (no activation). The first layer computes $\mathbf{y}_1 = \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1$, and the second layer computes $\mathbf{y}_2 = \mathbf{W}_2 \mathbf{y}_1 + \mathbf{b}_2$. If we substitute the first equation into the second:

$$\mathbf{y}_2 = \mathbf{W}_2 (\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 = \mathbf{W}_2 \mathbf{W}_1 \mathbf{x} + \mathbf{W}_2 \mathbf{b}_1 + \mathbf{b}_2$$

This simplifies to $\mathbf{y}_2 = \mathbf{W}_{\text{combined}} \mathbf{x} + \mathbf{b}_{\text{combined}}$, where $\mathbf{W}_{\text{combined}} = \mathbf{W}_2 \mathbf{W}_1$ and $\mathbf{b}_{\text{combined}} = \mathbf{W}_2 \mathbf{b}_1 + \mathbf{b}_2$. This is just a single linear transformation! No matter how many linear layers you stack, the result is always equivalent to a single linear layer. This is why linear layers "collapse"—they can't create any complexity beyond what a single layer can do.

**How Non-Linear Activation Functions Prevent Collapse:**

Now consider the same two-layer network, but with a non-linear activation function $f()$ applied after each layer. The first layer computes $\mathbf{y}_1 = f(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)$, and the second layer computes $\mathbf{y}_2 = f(\mathbf{W}_2 \mathbf{y}_1 + \mathbf{b}_2)$. When we substitute:

$$\mathbf{y}_2 = f(\mathbf{W}_2 f(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2)$$

Because $f()$ is non-linear, we cannot simplify this to a single linear transformation. The non-linearity "breaks" the composition, preventing collapse. Each layer now contributes something unique that cannot be replicated by a single layer.

**How Non-Linear Composition Creates Complex Patterns:**

The power of non-linear activation functions comes from composition—applying one non-linear function to the output of another. Each layer transforms the input in a non-linear way, and when you stack multiple such transformations, the complexity compounds.

Think of it like this: A single non-linear function can create simple curves (like ReLU creating sharp corners, or sigmoid creating S-curves). But when you compose multiple non-linear functions, each layer can:
1. **Detect simple patterns** in early layers (e.g., "is this pixel bright?" or "does this edge exist?")
2. **Combine simple patterns** in middle layers (e.g., "is this a corner?" or "does this look like part of a face?")
3. **Recognize complex patterns** in later layers (e.g., "is this a complete face?" or "does this sentence make sense?")

**Concrete Example: Hierarchical Pattern Learning**

Imagine training a network to recognize handwritten digits. Without activation functions, the network could only learn linear decision boundaries—it could separate "digit 0" from "digit 1" with a straight line, but couldn't learn the complex curved shapes that distinguish digits.

With ReLU activation functions, here's what happens:

- **Layer 1** learns to detect simple features: "Is there a vertical line here?" "Is there a horizontal line there?" Each neuron might activate when it sees a specific edge orientation. The ReLU function allows neurons to "turn on" (output positive values) when they detect their feature, and "turn off" (output zero) otherwise.

- **Layer 2** receives these simple feature detections and learns to combine them: "If I see a vertical line on the left AND a curve on the right, that might be part of a '6'." The ReLU function again allows neurons to activate only when multiple conditions are met.

- **Layer 3** receives these combined features and learns even more complex patterns: "If I see the pattern from Layer 2 that looks like the top of a '6', AND the pattern that looks like the bottom of a '6', then this is likely a '6'."

Each layer builds on the previous layer's output, and the non-linear activation function is what makes this hierarchical composition possible. Without it, Layer 2 would just be a linear combination of Layer 1's outputs, which could be replicated by a single layer. With non-linear activation, Layer 2 creates genuinely new patterns that Layer 1 couldn't represent.

**Mathematical Intuition:**

In algebra, composing linear functions always gives you a linear function: if $f(x) = ax + b$ and $g(x) = cx + d$, then $g(f(x)) = c(ax + b) + d = (ca)x + (cb + d)$, which is still linear. But composing non-linear functions creates new functions with different shapes. For example, if $f(x) = \max(0, x)$ (ReLU) and $g(x) = \max(0, x)$, then $g(f(x)) = \max(0, \max(0, x)) = \max(0, x)$—but when you have different weight matrices between the activations, the composition creates genuinely new non-linear patterns.

This is why deep networks with activation functions can learn complex patterns: each layer applies a non-linear transformation, and stacking many such transformations creates a function that can approximate arbitrarily complex relationships between inputs and outputs.

**How Patterns Translate to Classification Decisions:**

Understanding that networks learn hierarchical patterns is only half the story. The crucial question is: how do these patterns actually help the network make correct classifications? The answer lies in how the final output layer uses these learned patterns.

Consider our handwritten digit recognition example. The network doesn't just learn patterns for the sake of learning them—it learns patterns that are **useful for distinguishing between different digits**. Here's how the patterns contribute to the final classification:

1. **Pattern Detection Creates Feature Vectors**: Each layer's output is a vector of numbers, where each number represents how strongly a particular pattern is detected. For example, after Layer 1, you might have a vector like $[0.8, 0.2, 0.0, 0.5]$, where:
   - $0.8$ means "strong vertical line detected"
   - $0.2$ means "weak horizontal line detected"
   - $0.0$ means "no diagonal line detected"
   - $0.5$ means "moderate curve detected"

2. **Later Layers Combine Features**: Layer 2 receives these feature vectors and learns to combine them. A neuron in Layer 2 might learn: "If I see a strong vertical line (0.8) AND a moderate curve (0.5), this combination suggests the top part of a '6'." This neuron outputs a high value (say, 0.9) when it sees this combination, and a low value (say, 0.1) otherwise.

3. **Output Layer Makes the Decision**: The final output layer receives the combined features from the last hidden layer. It has one neuron per class (e.g., one for "digit 0", one for "digit 1", etc.). Each output neuron learns which combinations of features indicate its class. For example:
   - The "digit 1" neuron might have high weights for "vertical line" features and low weights for "curve" features
   - The "digit 6" neuron might have high weights for "vertical line + curve" combinations
   - The "digit 0" neuron might have high weights for "closed loop" features

4. **The Final Computation**: When the network sees an input image, it computes:
   - Layer 1 detects simple patterns → outputs feature vector $\mathbf{h}_1$
   - Layer 2 combines patterns → outputs feature vector $\mathbf{h}_2$
   - Layer 3 recognizes complex patterns → outputs feature vector $\mathbf{h}_3$
   - Output layer computes: $\text{logit}_i = \mathbf{w}_i \cdot \mathbf{h}_3 + b_i$ for each class $i$
   - The class with the highest logit becomes the prediction

**Concrete Example: Recognizing the Digit "6"**

Let's trace through what happens when the network sees an image of the digit "6":

- **Input**: Pixel values representing a "6" shape
- **Layer 1 Output**: $[0.8, 0.1, 0.0, 0.7]$ 
  - High value (0.8) for "vertical line on left" detector
  - Low value (0.1) for "horizontal line" detector
  - Zero (0.0) for "diagonal line" detector
  - High value (0.7) for "curve" detector

- **Layer 2 Output**: $[0.2, 0.9, 0.1, 0.3]$
  - Low value (0.2) for "straight line combination" detector
  - **High value (0.9) for "vertical line + curve" combination** (this is the key pattern for "6")
  - Low values for other combinations

- **Layer 3 Output**: $[0.1, 0.05, 0.95, 0.2]$
  - Very low values for "digit 0", "digit 1", "digit 2" patterns
  - **Very high value (0.95) for "digit 6" pattern** (recognizes the complete "6" shape)

- **Output Layer**: Computes logits for each class
  - Logit for "digit 0": $\mathbf{w}_0 \cdot [0.1, 0.05, 0.95, 0.2] + b_0 = 0.3$
  - Logit for "digit 1": $\mathbf{w}_1 \cdot [0.1, 0.05, 0.95, 0.2] + b_1 = 0.1$
  - Logit for "digit 2": $\mathbf{w}_2 \cdot [0.1, 0.05, 0.95, 0.2] + b_2 = 0.2$
  - **Logit for "digit 6": $\mathbf{w}_6 \cdot [0.1, 0.05, 0.95, 0.2] + b_6 = 2.5$** (highest!)

- **Softmax**: Converts logits to probabilities
  - $P(\text{digit 0}) = 0.05$
  - $P(\text{digit 1}) = 0.02$
  - $P(\text{digit 2}) = 0.03$
  - **$P(\text{digit 6}) = 0.90$** (prediction!)

**Why This Works: Training Guides Pattern Learning**

The key insight is that the network doesn't learn arbitrary patterns—it learns patterns that are **useful for reducing the loss function**. During training:

1. The network makes a prediction (e.g., "digit 6" with 90% confidence)
2. The loss function compares this to the true label (e.g., "digit 6")
3. Backpropagation computes gradients that tell each layer: "Your patterns helped (or hurt) the prediction"
4. Weight updates adjust the patterns to be more useful

For example, if the network incorrectly predicts "digit 1" when it sees a "6", the gradients will:
- **Increase** weights in neurons that detect "vertical line + curve" (the pattern that distinguishes "6" from "1")
- **Decrease** weights in neurons that detect "just vertical line" (the pattern that led to the wrong prediction)

Over many training examples, the network learns that certain pattern combinations are highly predictive of specific classes. The hierarchical structure means:
- Early layers learn patterns that are **reusable** across many classes (edges, curves, lines)
- Later layers learn patterns that are **specific** to particular classes (the exact combination that means "6" vs "0")

This is why the patterns have value: they're not just abstract features—they're features that have been optimized through training to maximize classification accuracy. Each pattern contributes to the final decision through the weighted combination in the output layer, and the training process ensures these weights are set to make the most useful patterns have the most influence.

**Neural Networks vs. Hashing: A Critical Distinction**

You might wonder: is this process like hashing, where ideally each data item gets a unique identifier? The answer is **no**—and understanding why reveals something fundamental about how neural networks work.

**Hashing creates unique identifiers:**
- Same input → same hash (deterministic)
- Different inputs → different hashes (ideally unique)
- Goal: Create a one-to-one mapping for lookup/identification
- Example: Hash function might map "cat.jpg" → `0x3A7F2B1C` (unique identifier)

**Neural networks learn shared representations:**
- Similar inputs → similar representations (not unique!)
- Goal: Learn features that help with classification/prediction
- Different inputs that belong to the same class should produce similar patterns
- Example: All images of "digit 6" should activate similar neurons, regardless of handwriting style

**Why We Want Shared Patterns, Not Unique Hashes:**

If neural networks worked like hashing (creating unique identifiers for each input), they would be useless for classification. Here's why:

1. **Generalization requires similarity**: To recognize a new "6" that the network has never seen before, it must share features with the "6"s it trained on. If each training example got a unique hash, the network couldn't recognize new examples.

2. **Pattern reuse is the goal**: The whole point is that many different "6"s share the same pattern (vertical line + curve). This shared pattern is what makes classification possible.

3. **The output layer needs shared features**: The output layer learns to say "if I see pattern X (vertical line + curve), predict class '6'". This only works if many different "6" images produce pattern X. If each image produced a unique pattern, the output layer couldn't learn useful mappings.

**Concrete Example:**

Imagine you have 1000 training images of the digit "6", each handwritten differently:
- Image 1: Slightly slanted "6"
- Image 2: Bold "6"
- Image 3: Thin "6"
- ... (997 more variations)

**With hashing (unique identifiers):**
- Image 1 → hash `0x3A7F2B1C`
- Image 2 → hash `0x9D4E8A2F`
- Image 3 → hash `0x1B6C3D9E`
- ... (all different, no relationship)

When you show a new "6" image, it gets a completely new hash. The network can't recognize it because it has no connection to the training examples.

**With neural network pattern learning (shared features):**
- Image 1 → pattern `[0.8, 0.1, 0.0, 0.7]` (vertical line + curve detected)
- Image 2 → pattern `[0.9, 0.0, 0.0, 0.8]` (stronger vertical line + curve)
- Image 3 → pattern `[0.7, 0.2, 0.0, 0.6]` (weaker but same pattern)
- ... (all similar, sharing the "vertical line + curve" pattern)

When you show a new "6" image, it produces a similar pattern (e.g., `[0.85, 0.1, 0.0, 0.75]`). The output layer recognizes this pattern and correctly predicts "digit 6" because it learned that this pattern combination indicates class "6".

**The Key Insight:**

Neural networks are doing the **opposite** of hashing:
- **Hashing**: "Make each input unique"
- **Neural networks**: "Make similar inputs produce similar representations"

This is called **representation learning** or **feature learning**. The network learns to map inputs to a space where:
- Similar inputs (same class) are close together
- Different inputs (different classes) are far apart

This is more like **clustering** than hashing—grouping similar things together rather than giving each thing a unique identifier.

**Why This Matters:**

If you tried to design a neural network that created unique hashes for each input, it would:
1. Fail to generalize to new examples
2. Require storing every training example (defeating the purpose of learning)
3. Be unable to classify anything it hasn't seen before

The shared patterns are what make neural networks powerful: they learn to recognize the **essence** of what makes something a "6" (the pattern), not just memorize each individual "6" (a unique hash).

**Similarity-Preserving Hashing: A Closer Analogy**

You might be thinking: "What about a hash function that generates identical hashes for similar things?" This is actually much closer to what neural networks do! This concept is called **locality-sensitive hashing (LSH)** or **similarity-preserving hashing**.

**Locality-Sensitive Hashing (LSH):**
- Similar inputs → similar (or identical) hashes
- Different inputs → different hashes
- Goal: Preserve similarity relationships in the hash space
- Example: Two similar images might hash to `0x3A7F2B1C` and `0x3A7F2B1D` (very similar hashes)

**How Neural Networks Compare:**

Neural networks are doing something similar to LSH, but with crucial differences:

1. **Neural networks learn what "similar" means**: Unlike traditional LSH (which uses a fixed similarity metric like Euclidean distance), neural networks learn from data what features make inputs similar for the specific task. For digit recognition, "similar" means "same digit class", not just "pixel-wise similar". A bold "6" and a thin "6" are very different pixel-wise, but the network learns they're similar because they share the pattern (vertical line + curve).

2. **Continuous representations, not discrete hashes**: Neural networks produce continuous vectors (e.g., `[0.8, 0.1, 0.0, 0.7]`), not discrete hash codes. This allows for:
   - Fine-grained similarity (not just "same" or "different")
   - Gradient-based learning (can't take gradients of discrete hashes)
   - Smooth interpolation between patterns

3. **Task-specific similarity**: The network learns similarity that's optimized for the task. For digit recognition, two "6"s with different handwriting styles are similar. For a different task (like detecting forgeries), those same two images might need to be treated as different.

**Concrete Comparison:**

**Traditional LSH (fixed similarity metric):**
- Uses a predefined distance function (e.g., Hamming distance, cosine similarity)
- Two images are "similar" if they're close in pixel space
- Hash function: `hash(image) = f(pixels)`, where `f` is fixed
- Problem: Pixel-wise similarity doesn't match semantic similarity (a bold "6" and thin "6" are far apart in pixel space but semantically identical)

**Neural Network (learned similarity):**
- Learns what makes inputs similar for the task
- Two images are "similar" if they activate similar patterns
- Representation function: `representation(image) = learned_patterns(image)`, where `learned_patterns` is optimized through training
- Solution: Learns that bold "6" and thin "6" are similar because they share the "vertical line + curve" pattern

**The Key Difference:**

Both LSH and neural networks create similarity-preserving mappings, but:
- **LSH**: Uses a fixed similarity metric → "These inputs are similar because they're close in this predefined space"
- **Neural networks**: Learn the similarity metric → "These inputs are similar because they share patterns that help with the task"

**Why This Matters:**

Neural networks can learn task-specific similarity. For example:
- **Digit recognition**: A bold "6" and thin "6" are similar (same digit)
- **Handwriting analysis**: A bold "6" and thin "6" might be different (different writing styles)
- **Forgery detection**: Two identical-looking "6"s might be different (one is a forgery)

The same network architecture can learn different notions of similarity depending on the training data and loss function. This flexibility is what makes neural networks powerful—they don't just preserve similarity, they learn what similarity means for the task at hand.

**The Bottom Line:**

Your intuition is correct: neural networks are like similarity-preserving hash functions. But they go further—they don't just preserve a predefined notion of similarity; they learn what "similar" means from the data, optimized for the specific task. This learned similarity is what enables them to generalize to new examples and perform well on classification tasks.

The most common activation functions you'll encounter are:

- **ReLU** (Rectified Linear Unit): $f(x) = \max(0, x)$ - This function keeps positive values unchanged and zeros out any negative values. It's the most commonly used activation in modern neural networks because it's simple, efficient, and works well in practice. Note that ReLU is non-differentiable at $x=0$, but in practice this rarely causes issues since the probability of exactly hitting zero is negligible.

- **Sigmoid**: $f(x) = \frac{1}{1+e^{-x}}$ - This function squashes any input into the range $(0, 1)$, making it useful when you need outputs that represent probabilities. The sigmoid function is smooth and differentiable everywhere, but it saturates (gradient approaches zero) for very large positive or negative inputs, which can slow down learning. While sigmoid can represent probabilities, modern networks typically use softmax for multi-class classification (see [Example 7: Character Recognition](12-example7-character-recognition.md)).

- **Tanh**: $f(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ - Similar to sigmoid but squashes inputs into the range $(-1, 1)$, providing a symmetric output around zero. Like sigmoid, tanh is smooth and differentiable everywhere, but also saturates at the extremes. The hyperbolic tangent function can also be written as $\tanh(x) = \frac{\sinh(x)}{\cosh(x)}$, where $\sinh(x) = \frac{e^x - e^{-x}}{2}$ and $\cosh(x) = \frac{e^x + e^{-x}}{2}$ are the hyperbolic sine and cosine functions, respectively.

**Perceptron Diagram:**

![The Perceptron: Single Neuron Decision-Making Unit](images/network-structure/perceptron-diagram.svg)

**Activation Function Behavior:**

| | | |
|:---:|:---:|:---:|
| ![ReLU Activation Function](images/activation-functions/activation-relu.svg) | ![Sigmoid Activation Function](images/activation-functions/activation-sigmoid.svg) | ![Tanh Activation Function](images/activation-functions/activation-tanh.svg) |

**Activation Function Graphs:**

| | | |
|:---:|:---:|:---:|
| ![ReLU Graph](images/activation-functions/activation-graph-relu.svg) | ![Sigmoid Graph](images/activation-functions/activation-graph-sigmoid.svg) | ![Tanh Graph](images/activation-functions/activation-graph-tanh.svg) |

To see ReLU activation in action within a feed-forward network, [Example 5: Feed-Forward Layers](10-example5-feedforward.md) demonstrates how ReLU is applied in the expansion phase of a feed-forward network.

While a single perceptron is quite limited in what it can learn, stacking many perceptrons together creates powerful networks capable of learning complex patterns. This is why we build networks with multiple layers, as we'll see in the next section.

---

## Layers: Stacking Neurons

A layer is a collection of neurons (perceptrons) that process information together in parallel. Think of a layer as a factory assembly line station: input arrives, the layer processes it, and output goes to the next layer. Each layer performs a specific job, and having multiple layers means multiple processing steps, each building on what the previous layer learned.

To understand how multiple perceptrons form a layer, consider a layer with $n$ perceptrons, each processing the same input vector $\mathbf{x} \in \mathbb{R}^d$. Each perceptron has its own weight vector $\mathbf{w}_i \in \mathbb{R}^d$ and bias $b_i \in \mathbb{R}$. The layer computes $n$ outputs simultaneously: $y_i = f(\mathbf{w}_i \cdot \mathbf{x} + b_i)$ for $i = 1, \ldots, n$. We can stack these weight vectors into a matrix $\mathbf{W} \in \mathbb{R}^{n \times d}$ where each row is a perceptron's weight vector, and stack the biases into a vector $\mathbf{b} \in \mathbb{R}^n$. The entire layer computation becomes $\mathbf{y} = f(\mathbf{W}\mathbf{x} + \mathbf{b})$, where $f$ is applied element-wise. This matrix formulation shows that a layer is essentially multiple dot products computed simultaneously—exactly what matrix multiplication does.

Neural networks typically have three types of layers. The input layer receives raw data and passes it into the network. Hidden layers process the information, and you can have as many of these as needed for the complexity of your problem. Finally, the output layer produces the final prediction.

To visualize this, imagine a multi-stage factory. Stage 1 (the input layer) is where raw materials arrive. Stages 2 through 4 (hidden layers) each refine the product in some way, with each stage building on the work of the previous stage. Stage 5 (the output layer) is where the finished product exits.

We trace through an example to illustrate how data flows through multiple layers. The values shown are arbitrary examples chosen to demonstrate the transformation process. This example matches the network architecture shown in the diagram above, with 3 input neurons, 4 neurons in the first hidden layer, 3 neurons in the second hidden layer, and 4 output neurons.

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

![Multi-Layer Neural Network: Input → Hidden Layers → Output](images/network-structure/multi-layer-network.svg)

Layers enable complex transformations that a single perceptron cannot achieve. In algebra, this is like composing functions. If you have two functions $f(x)$ and $g(x)$, composing them gives you $f(g(x))$—you apply $g$ first, then apply $f$ to the result. Multiple layers work the same way: Layer 2 processes the output of Layer 1, Layer 3 processes the output of Layer 2, and so on. By stacking layers, networks can learn hierarchical patterns—simple patterns in early layers, and increasingly complex patterns in deeper layers. This hierarchical learning is what makes deep neural networks so powerful.

To see layers working together in detail, [Example 5: Feed-Forward Layers](10-example5-feedforward.md) demonstrates how information flows through multiple layers.

---

## Feedforward Networks: Multi-Layer Perceptrons

A **feed-forward network** (FFN) is a type of layer that applies two linear transformations with an activation function in between. This creates a two-stage processing pipeline: first expanding the dimensions to give the network more room to work, then compressing back to the original dimensions. The activation function in between adds crucial non-linearity that enables the network to learn complex patterns.

Mathematically, a feed-forward network is defined as:

$$\text{FFN}(\mathbf{x}) = f(\mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2$$

where $\mathbf{x} \in \mathbb{R}^d$ is the input vector, $\mathbf{W}_1 \in \mathbb{R}^{d \times d'}$ is the first weight matrix that expands dimensions, $\mathbf{b}_1 \in \mathbb{R}^{d'}$ is the first bias vector, $\mathbf{W}_2 \in \mathbb{R}^{d' \times d}$ is the second weight matrix that compresses dimensions back, $\mathbf{b}_2 \in \mathbb{R}^{d}$ is the second bias vector, and $f()$ is a non-linear activation function applied element-wise. The expansion factor $d' > d$ (typically $d' = 4d$ in practice) gives the network more capacity to learn complex feature combinations.

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

![Feed-Forward Network Structure](images/network-structure/ffn-structure.svg)

Feed-forward networks are crucial because they add both non-linearity and capacity to neural networks. The expansion phase allows the network to learn complex feature combinations that wouldn't be possible with just linear transformations. This is why FFNs are a core component of many modern neural network architectures.

To see feed-forward networks in action, [Example 5: Feed-Forward Layers](10-example5-feedforward.md) demonstrates the complete FFN computation step by step.

---

## Loss Functions: Measuring Error

A **loss function** (also called **cross-entropy loss**) measures how wrong the model's prediction is compared to the target. In basic algebra, this is like measuring the distance between two points. If you have a target point and a predicted point, you can compute how far apart they are. The loss function does something similar: it measures how far the model's prediction is from the correct answer. Think of loss like a score in a game: lower loss means better predictions, while higher loss means worse predictions. Our goal during training is to minimize the loss, which maximizes the model's accuracy.

Mathematically, the cross-entropy loss is defined as:

$$L = -\log P_{\text{model}}(y_{\text{target}})$$

where $P_{\text{model}}(y_{\text{target}})$ is the probability assigned by the model to the correct class $y_{\text{target}}$. Notice that when the model assigns high probability to the correct answer, the loss is low (since the logarithm of a number close to 1 is close to 0). When the model assigns low probability to the correct answer, the loss is high (since the logarithm of a small number is a large negative number, and we negate it).

We use log probabilities rather than raw probabilities to avoid numerical underflow when probabilities are very small (e.g., $10^{-10}$). Direct probability multiplication would cause the result to underflow to zero in floating-point arithmetic, making gradient computation impossible. By working in log space, we maintain numerical stability while preserving the mathematical relationship between probabilities and loss.

To understand this intuitively, think of loss like a golf score: lower is better. A perfect prediction gives you a loss near 0, while a wrong prediction gives you a high loss. Just as in golf, you want to minimize your score.

We illustrate this with a numerical example. For clarity, we use a simple 4-class classification problem with arbitrary class labels:

**Given:**
- Target: Class C (one-hot encoding: $\mathbf{y}_{\text{target}} = \begin{bmatrix} 0 \\ 0 \\ 1 \\ 0 \end{bmatrix}$)
- Prediction probabilities: $\mathbf{p} = \begin{bmatrix} 0.1 \\ 0.2 \\ 0.6 \\ 0.1 \end{bmatrix}$

**Equation to solve:**
$$L = -\log(0.6) = 0.51$$

**Computation:**

$$L = -\log(0.6) \approx 0.51$$

The model assigned 60% probability to the correct class, resulting in a loss of approximately 0.51.

In this case, the model correctly identified class C as the most likely answer, assigning it 60% probability. The loss of 0.51 reflects that this is a reasonable prediction, though not perfect. Now consider what happens when the model makes a wrong prediction:

**Given:**
- Prediction probabilities: $\mathbf{p} = \begin{bmatrix} 0.8 \\ 0.1 \\ 0.05 \\ 0.05 \end{bmatrix}$ (model assigned only 5% to correct class)

**Equation to solve:**
$$L = -\log(0.05) \approx 3.0$$

**Computation:**

$$L = -\log(0.05) \approx 3.0$$

The model assigned only 5% probability to the correct class, resulting in a much higher loss of approximately 3.0.

Here, the model assigned only 5% probability to the correct answer (class C), while assigning 80% to an incorrect class. The loss of 3.0 is much higher, correctly penalizing the model for its poor prediction. The specific class labels and probabilities are chosen for illustration; in practice, these would represent actual predictions from a trained model.

| | |
|:---:|:---:|
| ![Correct Prediction](images/training/loss-correct.svg) | ![Wrong Prediction](images/training/loss-wrong.svg) |

![Cross-Entropy Loss: L = -log(P(target))](images/training/cross-entropy-loss.svg)

Loss functions are essential because they tell us how well the model is learning. During training, we compute the loss after each prediction and use it to guide how we update the model's parameters. The entire training process is essentially a search for parameter values that minimize this loss function.

To see loss computation in action, [Example 2: Single Training Step](07-example2-single-step.md) shows how loss is computed and used to update the model. To see a complete classification example with hand calculations, [Example 7: Character Recognition](12-example7-character-recognition.md) demonstrates digit classification from pixel input to final prediction.

---

## Gradient Descent: How Networks Learn

Gradient descent is an optimization algorithm that uses gradients to iteratively update parameters in order to minimize loss. To understand how it works, imagine walking downhill blindfolded. You can't see the bottom of the valley, but you can feel which way is downhill—that feeling is the gradient. You take a step in that direction (a weight update), and you repeat this process until you reach the bottom (minimum loss). The size of each step is controlled by the learning rate, which we'll discuss shortly.

### The Gradient

A gradient shows how much each parameter should change to reduce the loss. In basic calculus and algebra, the gradient is the derivative—it tells you the slope of a function at a given point. If you remember from algebra that the slope of a line $y = mx + b$ is $m$, the gradient generalizes this concept to multi-dimensional functions. The gradient tells you the slope in each direction.

The gradient $\nabla_{\mathbf{W}} L$ is a vector (or matrix) of partial derivatives: $\nabla_{\mathbf{W}} L = \begin{bmatrix} \frac{\partial L}{\partial w_1} \\ \frac{\partial L}{\partial w_2} \\ \vdots \end{bmatrix}$. Each component tells us how much the loss changes when we change that specific parameter. Think of the gradient as a compass pointing uphill. Since loss is like altitude (and we want to go down), the gradient points in the direction of steepest increase. This means the negative gradient points in the direction we want to go—downhill, toward lower loss. The magnitude of the gradient tells us how steep the slope is: a large gradient means a steep slope, while a small gradient means a gentle slope.

We illustrate this with a numerical example using arbitrary values chosen to demonstrate the gradient descent update:

**Given:**
- Parameter $W$ (a weight in a matrix): Current value $W = 0.5$ (example initial weight)
- Loss at $W=0.5$: $L = 2.0$ (example loss value)
- Gradient: $\frac{\partial L}{\partial W} = -0.3$ (example gradient value)
- Learning rate: $\eta = 0.1$

**Equation to solve:**
$$W_{\text{new}} = W_{\text{old}} - \eta \cdot \frac{\partial L}{\partial W} = 0.5 - 0.1 \times (-0.3) = 0.5 + 0.03 = 0.53$$

**Computation:**

The negative gradient indicates that increasing $W$ will decrease the loss. The magnitude of 0.3 indicates a moderate slope. Applying the gradient descent update rule:

$$W_{\text{new}} = W_{\text{old}} - \eta \cdot \frac{\partial L}{\partial W} = 0.5 - 0.1 \times (-0.3) = 0.5 + 0.03 = 0.53$$

In this example, the negative gradient tells us that increasing the weight will decrease the loss. The magnitude of 0.3 indicates a moderate slope. We then update the weight by moving in the direction opposite to the gradient (since we want to minimize loss), scaled by the learning rate. The specific values are chosen for illustration; in practice, these would be computed from the actual loss function and current parameter values.

**Gradient Visualization:**

![Gradient Visualization](images/training/gradient-visualization.svg)

### The Learning Rate

The learning rate is a hyperparameter that controls how large each weight update is during training. Think of the learning rate as your step size when walking downhill. If you take large steps (high learning rate), you make fast progress but might overshoot the bottom of the valley. If you take small steps (low learning rate), your progress is slower but more precise. If your steps are too large, you might jump right over the valley and never find the minimum (this is called divergence). If your steps are too small, it takes forever to reach the bottom (slow convergence).

In neural networks, the learning rate (denoted as $\eta$ or `lr`) is typically set between 0.0001 and 0.01. It's used in the gradient descent update rule: $W_{\text{new}} = W_{\text{old}} - \eta \times \text{gradient}$. Often, the learning rate is scheduled to start high (for fast initial learning) and decrease over time (for fine-tuning as training progresses).

To see how the learning rate affects updates, consider this example using arbitrary values:

**Given:**
- Gradient: $\frac{\partial L}{\partial W} = -0.5$ (example gradient value indicating weight should increase)
- Learning rate: $\eta = 0.1$ (small step size)

**Equation to solve (with learning rate 0.1):**
$$W_{\text{new}} = W_{\text{old}} - \eta \cdot \frac{\partial L}{\partial W} = W_{\text{old}} - 0.1 \times (-0.5) = W_{\text{old}} + 0.05$$

**Computation:**

$$W_{\text{new}} = W_{\text{old}} - 0.1 \times (-0.5) = W_{\text{old}} + 0.05$$

This results in a small increase in the weight value.

With a learning rate of 0.1, we make a modest update. But if we used a learning rate of 1.0:

**Equation to solve (with learning rate 1.0):**
$$W_{\text{new}} = W_{\text{old}} - \eta \cdot \frac{\partial L}{\partial W} = W_{\text{old}} - 1.0 \times (-0.5) = W_{\text{old}} + 0.5$$

**Computation with larger learning rate:**

If the learning rate was $\eta = 1.0$:
$$W_{\text{new}} = W_{\text{old}} - 1.0 \times (-0.5) = W_{\text{old}} + 0.5$$

This results in a much larger increase in the weight value, which might cause the algorithm to overshoot the optimal value.

This much larger update could cause us to overshoot the optimal weight value, potentially making the loss worse rather than better. The gradient value of -0.5 is chosen arbitrarily for illustration; in practice, it would be computed from the actual loss function.

### Gradient Descent Algorithm

The gradient descent algorithm is mathematically defined as:

$$\mathbf{W}_{\text{new}} = \mathbf{W}_{\text{old}} - \eta \cdot \nabla_{\mathbf{W}} L$$

where $\mathbf{W} \in \mathbb{R}^{m \times n}$ is the weight matrix, $\eta \in \mathbb{R}^+$ is the learning rate, and $\nabla_{\mathbf{W}} L \in \mathbb{R}^{m \times n}$ is the gradient of the loss with respect to the weights.

Gradient descent converges to a local minimum under certain conditions: the loss function must be differentiable (or at least have subgradients), the learning rate must be sufficiently small (typically $\eta < \frac{2}{\lambda_{\max}}$ where $\lambda_{\max}$ is the largest eigenvalue of the Hessian), and the initialization must be reasonable. In practice, neural network loss landscapes are non-convex, so gradient descent finds local minima rather than global minima. However, for overparameterized networks (which includes most modern architectures), local minima are often good enough for practical purposes.

The gradient descent process follows these steps: First, we compute the loss by comparing the model's prediction to the target. Next, we compute the gradients, which tell us which direction to move each weight. Then we update the weights using the formula $W_{\text{new}} = W_{\text{old}} - \eta \times \frac{\partial L}{\partial W}$. This is like solving an equation iteratively. In algebra, if you're trying to find where a function equals zero, you might start with a guess, compute the slope at that point, and move in the direction that reduces the function value. Gradient descent does exactly this: it iteratively refines the weights until it finds values that minimize the loss. We repeat this process for many iterations, gradually moving the weights toward values that minimize the loss.

| | |
|:---:|:---:|
| ![Gradient Descent Algorithm](images/training/gradient-descent-algorithm.svg) | ![Gradient Descent: Loss Decreases Over Time](images/training/gradient-descent-path.svg) |

We trace through a complete iteration to illustrate how this works, using example values chosen to demonstrate the process:

**Given:**
- Initial weight: $W = 0.5$ (arbitrary starting value)
- Loss: $L = 2.0$ (example initial loss)
- Gradient: $\frac{\partial L}{\partial W} = -0.3$ (negative indicates increasing $W$ will decrease loss)
- Learning rate: $\eta = 0.1$ (example learning rate)

**Equation to solve:**
$$W_{\text{new}} = W_{\text{old}} - \eta \cdot \frac{\partial L}{\partial W} = 0.5 - 0.1 \times (-0.3) = 0.5 + 0.03 = 0.53$$

**Computation:**

$$W_{\text{new}} = 0.5 - 0.1 \times (-0.3) = 0.5 + 0.03 = 0.53$$

After this update, the loss decreases from $L = 2.0$ to $L = 1.8$, confirming that we moved in the correct direction.

After this update, the loss decreased from 2.0 to 1.8, confirming that we moved in the right direction. We would continue this process, computing new gradients and making new updates, until the loss stops decreasing significantly. The specific values are illustrative; in practice, these would be computed from the actual model and data.

Gradient descent is the fundamental algorithm that enables neural networks to learn. Without it, we couldn't systematically update parameters to reduce loss. It's the engine that drives all neural network training, from simple perceptrons to complex multi-layer networks.

To see gradient descent in action, [Example 2: Single Training Step](07-example2-single-step.md) demonstrates a complete gradient descent update with a single weight.

---

## Backpropagation: Computing Gradients

Backpropagation is the algorithm that computes gradients by propagating the loss backward through the network. To understand this, think of tracing back the cause of a mistake. You made an error (high loss), and now you need to work backwards to understand what caused it. You check each step, asking "Did this layer contribute to the error?" and calculate how much each parameter should change to fix it.

Backpropagation operates on a computational graph, which represents the forward pass as a directed acyclic graph (DAG). Each node represents an operation (addition, multiplication, activation function, etc.), and edges represent data flow. The computational graph enables automatic differentiation: by traversing the graph backward from the loss node, we can compute gradients for all parameters using the chain rule of calculus.

### The Forward Pass

The forward pass is the process of computing predictions by passing input data through the network from input to output. Think of the forward pass as following a recipe step-by-step. You start with ingredients (input data), process them through each step (each layer), and end with the final dish (the prediction). Data flows in one direction: Input → Layer 1 → Layer 2 → ... → Output.

In a neural network, the forward pass follows a specific sequence of transformations. First, the input data is received by the input layer. Then each hidden layer processes the data, applying weight matrices, adding biases, and applying activation functions. The output layer produces the final predictions. For classification tasks, the output is typically converted to probabilities using a softmax function. For a concrete classification example, see [Example 7: Character Recognition](12-example7-character-recognition.md), which shows how pixel inputs become class predictions through the same process.

![Forward Pass Flow](images/flow-diagrams/forward-pass-flow.svg)

We trace through a complete forward pass example using arbitrary values chosen to illustrate the transformation process:

**Equations to solve:**
- Hidden Layer 1: $\mathbf{y}_1 = f\left(\mathbf{W}_1 \begin{bmatrix} 0.5 \\ 0.3 \end{bmatrix} + \mathbf{b}_1\right) = \begin{bmatrix} 0.1 \\ 0.2 \\ 0.4 \end{bmatrix}$
- Hidden Layer 2: $\mathbf{y}_2 = f\left(\mathbf{W}_2 \begin{bmatrix} 0.1 \\ 0.2 \\ 0.4 \end{bmatrix} + \mathbf{b}_2\right) = \begin{bmatrix} 0.3 \\ 0.7 \end{bmatrix}$
- Output Layer: $\mathbf{y}_{\text{out}} = f\left(\mathbf{W}_{\text{out}} \begin{bmatrix} 0.3 \\ 0.7 \end{bmatrix} + \mathbf{b}_{\text{out}}\right) = \begin{bmatrix} 1.0 \\ 2.0 \\ 0.5 \\ 0.3 \end{bmatrix}$
- Softmax: $\text{softmax}\left(\begin{bmatrix} 1.0 \\ 2.0 \\ 0.5 \\ 0.3 \end{bmatrix}\right) = \begin{bmatrix} 0.1 \\ 0.8 \\ 0.05 \\ 0.05 \end{bmatrix}$

**Computation:**

1. **Input**: $\mathbf{x} = \begin{bmatrix} 0.5 \\ 0.3 \end{bmatrix}$ (example 2-dimensional input)

2. **Input Layer**: $\mathbf{x}_{\text{in}} = \begin{bmatrix} 0.5 \\ 0.3 \end{bmatrix}$

3. **Hidden Layer 1**: $\mathbf{y}_1 = \begin{bmatrix} 0.1 \\ 0.2 \\ 0.4 \end{bmatrix}$ (expands to 3 dimensions)

4. **Hidden Layer 2**: $\mathbf{y}_2 = \begin{bmatrix} 0.3 \\ 0.7 \end{bmatrix}$ (compresses to 2 dimensions)

5. **Output Layer**: $\mathbf{z} = \begin{bmatrix} 1.0 \\ 2.0 \\ 0.5 \\ 0.3 \end{bmatrix}$ (raw scores for 4 classes)

6. **Softmax**: $\text{softmax}(\mathbf{z}) = \begin{bmatrix} 0.1 \\ 0.8 \\ 0.05 \\ 0.05 \end{bmatrix}$ (probabilities for 4 classes)

Each step transforms the data, building up a richer representation until we finally have probabilities for each possible output class. The forward pass is how the model makes predictions, and it's the first step in both inference (making predictions) and training (where we'll then compute loss and gradients). The specific numerical values are illustrative; in a trained network, these would result from the learned weight matrices.

To see the forward pass in detail, [Example 1: Minimal Forward Pass](06-example1-forward-pass.md) shows the complete forward pass computation step by step.

### The Backward Pass

The backward pass (also called backpropagation) is the process of computing gradients by propagating the loss backward through the network from output to input. This is where the magic of learning happens—we figure out how each parameter contributed to the error and how much it should change.

In a neural network, the backward pass follows this sequence. First, we compute the loss at the output by comparing the prediction to the target. Then we compute the gradient with respect to the output. Next, we propagate this gradient backward through each layer, using the chain rule of calculus. The gradient flows from the loss through the output layer, then through each hidden layer in reverse order, and finally to the input layer. At each step, we compute the gradient for each parameter (weight and bias). Once we have all the gradients, we use them to update the parameters using gradient descent.

The chain rule enables this backward propagation. For a two-layer network with output $y = f_2(f_1(\mathbf{x}; \mathbf{W}_1); \mathbf{W}_2)$, the gradient with respect to $\mathbf{W}_1$ is computed as:

$$\frac{\partial L}{\partial \mathbf{W}_1} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial f_1} \frac{\partial f_1}{\partial \mathbf{W}_1}$$

Each term in this product is computed during the backward pass, with gradients flowing from the output back to the input. This is exactly what backpropagation does: it applies the chain rule at each layer, computing gradients layer by layer from output to input.

![Backward Pass Flow](images/flow-diagrams/backward-pass-flow.svg)

We trace through a backward pass example using arbitrary gradient values chosen to illustrate the propagation process:

**Given:**
- Loss: $L = 0.5$ (example loss value)

**Equations to solve (using chain rule):**

1. Gradient with respect to output:
   $$\frac{\partial L}{\partial \mathbf{y}_{\text{out}}} = \begin{bmatrix} -0.2 \\ 0.8 \\ -0.3 \\ -0.3 \end{bmatrix}$$

2. Gradient with respect to hidden layer 2 (computed via chain rule):
   $$\frac{\partial L}{\partial \mathbf{y}_2} = \begin{bmatrix} 0.1 \\ 0.2 \end{bmatrix}$$

3. Gradient with respect to hidden layer 1 (computed via chain rule):
   $$\frac{\partial L}{\partial \mathbf{y}_1} = \begin{bmatrix} 0.05 \\ 0.15 \\ 0.1 \end{bmatrix}$$

4. For each weight matrix (computed via chain rule):
   $$\frac{\partial L}{\partial \mathbf{W}_i} = \frac{\partial L}{\partial \mathbf{y}_i} \frac{\partial \mathbf{y}_i}{\partial \mathbf{W}_i}$$

**Computation:**

The backward pass propagates gradients from the output layer back to the input layer:

1. **Gradient with respect to output**: $\frac{\partial L}{\partial \mathbf{y}_{\text{out}}} = \begin{bmatrix} -0.2 \\ 0.8 \\ -0.3 \\ -0.3 \end{bmatrix}$ (example gradients)

2. **Gradient with respect to hidden layer 2**: $\frac{\partial L}{\partial \mathbf{y}_2} = \begin{bmatrix} 0.1 \\ 0.2 \end{bmatrix}$ (propagated backward via chain rule)

3. **Gradient with respect to hidden layer 1**: $\frac{\partial L}{\partial \mathbf{y}_1} = \begin{bmatrix} 0.05 \\ 0.15 \\ 0.1 \end{bmatrix}$ (propagated backward via chain rule)

4. **Gradient with respect to input layer**: Computed via chain rule

5. **Gradient with respect to all weight matrices**: Computed using $\frac{\partial L}{\partial \mathbf{W}_i} = \frac{\partial L}{\partial \mathbf{y}_i} \frac{\partial \mathbf{y}_i}{\partial \mathbf{W}_i}$

6. **Weight update**: $\mathbf{W}_{\text{new}} = \mathbf{W}_{\text{old}} - \eta \cdot \frac{\partial L}{\partial \mathbf{W}_i}$

The gradient values shown are illustrative examples. In practice, these would be computed exactly using the chain rule of calculus based on the actual loss function and network architecture.

The gradient flows backward, with each layer receiving information about how much it contributed to the error. The backward pass enables learning because without it, we couldn't compute how to update parameters to reduce loss. It's the mechanism that makes gradient descent possible in multi-layer networks.

To see the complete backward pass in action, [Example 3: Full Backpropagation](08-example3-full-backprop.md) shows the complete gradient flow through all components of a multi-layer network.

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

To see a complete training cycle, [Example 2: Single Training Step](07-example2-single-step.md) shows one complete iteration of the training loop.

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

The sequences shown are illustrative examples. In practice, these would be converted to numerical vectors (embeddings) before processing. The key point is that all sequences in the batch are processed in parallel, and their gradients are averaged.

![Batch Training](images/training/batch-training.svg)

Batching matters because it enables efficient GPU utilization and provides stable gradient estimates. Larger batches give more stable gradients (since you're averaging over more examples) but require more memory. Smaller batches use less memory but may have noisier gradients.

To see batch training in action, [Example 4: Multiple Patterns](09-example4-multiple-patterns.md) demonstrates how multiple sequences are processed together.

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

To see the complete transformer architecture implemented with all components working together, [Example 6: Complete Transformer](11-example6-complete.md) demonstrates the full implementation including multiple transformer blocks, layer normalization, residual connections, and the complete training pipeline.

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

**Logits** are the raw, unnormalized scores output by the model before applying softmax. Think of logits like raw test scores before grading on a curve. You might have raw scores [85, 90, 75, 80], and after applying the curve (softmax), you get probabilities [0.2, 0.5, 0.1, 0.2]. Logits can be any real numbers (positive, negative, large, small), while probabilities must be between 0 and 1 and sum to 1. **Softmax** is a function that converts numbers into probabilities (they sum to 1.0). In basic arithmetic, this is like converting numbers to percentages that add up to 100%. If you have test scores [85, 90, 75] out of 100, you might convert them to percentages, but softmax does something more sophisticated: it ensures the largest number gets the biggest share while all numbers sum to exactly 1.0. Think of softmax like dividing a pie. If you have scores [5, 2, 1], softmax converts them to probabilities [0.7, 0.2, 0.1]. The largest score gets the biggest slice, and all slices sum to 1.0 (the whole pie).

#### Training Techniques

**Layer Normalization**

**Layer normalization** is a technique that normalizes the inputs to a layer by adjusting the mean and variance. In basic algebra and statistics, this is exactly like computing z-scores: you subtract the mean and divide by the standard deviation. The formula is $z = \frac{x - \mu}{\sigma}$, where $\mu$ is the mean and $\sigma$ is the standard deviation. This transforms any set of numbers so they have a mean of 0 and a standard deviation of 1. Think of layer normalization like standardizing test scores. Raw scores might vary widely (0-100), but after normalization (subtract mean, divide by standard deviation), the scores are centered around 0 with a consistent scale. To see layer normalization in a complete transformer implementation, [Example 6: Complete Transformer](11-example6-complete.md) includes layer normalization in all transformer blocks.

**Residual Connections**

A **residual connection** adds the input of a layer directly to its output. In algebra, this is like adding two functions together: if you have $f(x)$ and you compute $f(x) + x$, you're adding the original input to the transformed output. This is exactly what a residual connection does: Output = Layer(Input) + Input. Think of a residual connection like a shortcut or bypass. The main path goes Input → Layer → Output, but there's also a shortcut that goes Input → (directly to output). The final output is Layer(Input) + Input. Residual connections enable training of very deep networks by allowing gradients to flow directly through the shortcut, which helps prevent the vanishing gradient problem that can occur in very deep networks.

To see these concepts in action, [Example 5: Feed-Forward Layers](10-example5-feedforward.md) shows residual connections in action, and [Example 6: Complete Transformer](11-example6-complete.md) shows the full architecture with all components working together.

---

## The Complete Learning Cycle

Now that we've covered all the individual components, let's see how they work together in a complete transformer. The learning cycle consists of a forward pass (where the model makes predictions) and a backward pass (where the model learns from its mistakes).

During the forward pass, the model makes predictions. First, tokenization breaks the text "The cat sat" into tokens: ["The", "cat", "sat"]. Token encoding then converts these tokens to integer IDs: [1, 2, 3]. Embedding lookup converts the integer IDs into vectors: [[0.1, 0.2], [0.3, 0.7], [0.5, 0.1]]. The Q/K/V maps transform each embedding into three views (Query, Key, and Value). Attention computes Q·K (dot product) to find which tokens are relevant to each other. Softmax converts the attention scores to probabilities. The context vector is computed as a weighted sum of all Values, where the weights come from attention. Output projection (WO) transforms the context vector into vocabulary scores (logits). Finally, softmax converts the logits into prediction probabilities.

During the backward pass, the model learns from its mistakes. The loss function compares the prediction to the target. The backward pass then computes gradients, which flow backward through the network using the chain rule. Finally, weight updates change the parameters (weights and biases) to reduce the loss, using the gradient descent algorithm we learned earlier.

To see these concepts in action, we've prepared several examples that build from simple to complex. [Example 1: Minimal Forward Pass](06-example1-forward-pass.md) demonstrates the forward pass only, showing how predictions are made. [Example 2: Single Training Step](07-example2-single-step.md) shows one complete training cycle, combining forward pass, loss computation, and weight updates. [Example 3: Full Backpropagation](08-example3-full-backprop.md) traces the complete gradient flow through all components. [Example 4: Multiple Patterns](09-example4-multiple-patterns.md) demonstrates batch training with multiple sequences. [Example 5: Feed-Forward Layers](10-example5-feedforward.md) adds feed-forward networks and residual connections to the architecture. Finally, [Example 6: Complete Transformer](11-example6-complete.md) shows the full architecture with all components working together.

---

## Key Principles

As we conclude this chapter, let's summarize the key principles that underlie everything we've learned. First, everything in a transformer is a vector or matrix. Tokens become vectors, and all operations are performed using matrices. This mathematical foundation enables the parallel processing that makes transformers efficient.

Second, attention finds relevance. The Q·K dot product measures how relevant each token is to every other token, allowing the model to focus on the most important information when making predictions.

Third, softmax creates probabilities. It converts any scores into probabilities that sum to 1, which is essential for making predictions about which token comes next.

Fourth, the context vector combines information. It's a weighted sum of all token values, where the weights come from attention. This allows the model to blend information from multiple tokens based on their relevance.

Fifth, learning equals gradient descent. Gradients show how to update weights to reduce loss, and this is the mechanism that enables all neural network learning, from simple perceptrons to complex transformers.

---

## What's Next?

Now that you understand neural network fundamentals, you're ready to dive deeper into the specific components that make transformers work. In Chapter 2: The Matrix Core, we'll take a deep dive into matrix operations, which are the mathematical foundation of everything transformers do. Chapter 3: Embeddings will show you exactly how tokens become vectors and why this representation is so powerful. Chapter 4: Attention Intuition will help you develop a deep understanding of how attention finds relevant information. Finally, Chapter 5: Why Transformers? will explain the specific problems that transformers solve and why they've become so dominant in modern AI.

For quick reference as you continue reading, see [Appendix B: Terminology Reference](appendix-b-terminology-reference.md) for all definitions with their physical analogies.

Remember: Every concept in this chapter has a physical analogy. If you ever forget what something means, think about its physical analogy—that's what it's actually modeling. These analogies aren't just helpful mnemonics; they reflect the real-world processes that neural networks are designed to capture.

---

**Navigation:**
- [← Introduction](00-introduction.md) | [← Index](00-index.md) | [Next: The Matrix Core →](02-matrix-core.md)

