## Chapter 3: Learning Algorithms

Now that we understand network architecture—how perceptrons form layers and how layers combine to create networks—we need to understand how networks learn. This chapter covers the three fundamental algorithms that enable learning: loss functions (which measure error), gradient descent (which finds better parameters), and backpropagation (which computes the gradients needed for gradient descent).

These algorithms work together: the loss function tells us how wrong we are, backpropagation computes how to fix it, and gradient descent actually makes the fix. Understanding these three components is essential for understanding how neural networks improve their predictions through training.

**Navigation:**
- [← Previous: Multilayer Networks and Architecture](02-multilayer-networks-architecture.md) | [Table of Contents](00-index.md) | [Next: Training Neural Networks →](04-training-neural-networks.md)

---

## Loss Functions: Measuring Error

A **loss function** (also called **cross-entropy loss**) measures how wrong the model's prediction is compared to the target. In basic algebra, this is like measuring the distance between two points. If you have a target point and a predicted point, you can compute how far apart they are. The loss function does something similar: it measures how far the model's prediction is from the correct answer. Think of loss like a score in a game: lower loss means better predictions, while higher loss means worse predictions. Our goal during training is to minimize the loss, which maximizes the model's accuracy.

The cross-entropy loss is defined mathematically as:

$$L = -\log P_{\text{model}}(y_{\text{target}})$$

Let's break down what this formula means:

**1. The components:**
- $P_{\text{model}}(y_{\text{target}})$ is the **probability** assigned by the model to the correct class $y_{\text{target}}$ (the true answer). This is a number between 0 and 1.
- $\log$ is the **natural logarithm** (base $e$), which transforms probabilities into a different scale.
- The negative sign ($-$) flips the sign so that lower loss means better predictions.

**2. How it works:**
- When the model assigns **high probability** to the correct answer (e.g., $P = 0.9$), the loss is **low** (since $\log(0.9) \approx -0.1$, and $-\log(0.9) \approx 0.1$).
- When the model assigns **low probability** to the correct answer (e.g., $P = 0.1$), the loss is **high** (since $\log(0.1) \approx -2.3$, and $-\log(0.1) \approx 2.3$).

**3. Why we use logarithms:**
We use **log probabilities** rather than raw probabilities for two important reasons:

- **Numerical stability:** When probabilities are very small (e.g., $10^{-10}$), direct probability multiplication would cause **numerical underflow** (the result becomes zero in floating-point arithmetic), making **gradient computation** impossible. By working in **log space**, we maintain **numerical stability** while preserving the mathematical relationship between probabilities and loss.

- **Computational efficiency:** Logarithms convert multiplication into addition, which is computationally faster and more stable for gradient calculations.

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

| |
|:---:|
| ![Cross-Entropy Loss: L = -log(P(target))](images/training/cross-entropy-loss.svg) |

Loss functions are essential because they tell us how well the model is learning. During training, we compute the loss after each prediction and use it to guide how we update the model's parameters. The entire training process is essentially a search for parameter values that minimize this loss function.

To see loss computation in action, [Example 2: Single Training Step](10-example2-single-step.md) shows how loss is computed and used to update the model. To see a complete classification example with hand calculations, [Example 7: Character Recognition](15-example7-character-recognition.md) demonstrates digit classification from pixel input to final prediction.

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


| |
|:---:|
| ![Gradient Visualization](images/training/gradient-visualization.svg) |


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

Let's break down each component:

**1. The components:**
- $\mathbf{W} \in \mathbb{R}^{m \times n}$ is the **weight matrix** (a matrix with $m$ rows and $n$ columns, where each element is a real number)
- $\eta \in \mathbb{R}^+$ is the **learning rate** (a positive real number that controls step size)
- $\nabla_{\mathbf{W}} L \in \mathbb{R}^{m \times n}$ is the **gradient** of the loss with respect to the weights (a matrix of the same size as $\mathbf{W}$, where each element tells us how much the loss changes when we change that weight)

**2. How it works:**
The formula says: "Take the old weights, subtract the gradient (scaled by the learning rate), and that gives us the new weights." The negative sign means we move in the direction opposite to the gradient (since the gradient points uphill, we want to go downhill to reduce loss).

**3. Convergence conditions:**
Gradient descent converges to a **local minimum** (a point where the loss is lower than all nearby points) under certain conditions:

- The **loss function** must be **differentiable** (or at least have **subgradients**), meaning we can compute how the loss changes as we change the weights
- The learning rate must be sufficiently small (typically $\eta < \frac{2}{\lambda_{\max}}$ where $\lambda_{\max}$ is the largest **eigenvalue** of the **Hessian** matrix, which measures the curvature of the loss function)
- The **initialization** (starting values for the weights) must be reasonable

**4. Local vs. global minima:**
In practice, neural network **loss landscapes** (the shape of the loss function across all possible weight values) are **non-convex** (they have many hills and valleys, not just one smooth bowl). This means gradient descent finds **local minima** rather than **global minima** (the absolute lowest point). However, for **overparameterized networks** (networks with more parameters than training examples, which includes most modern architectures), local minima are often good enough for practical purposes—they typically have loss values very close to the global minimum.

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

To see gradient descent in action, [Example 2: Single Training Step](10-example2-single-step.md) demonstrates a complete gradient descent update with a single weight.

---

## Backpropagation: Computing Gradients

Backpropagation is the algorithm that computes gradients by propagating the loss backward through the network. To understand this, think of tracing back the cause of a mistake. You made an error (high loss), and now you need to work backwards to understand what caused it. You check each step, asking "Did this layer contribute to the error?" and calculate how much each parameter should change to fix it.

Backpropagation operates on a computational graph, which represents the forward pass as a directed acyclic graph (DAG). Each node represents an operation (addition, multiplication, activation function, etc.), and edges represent data flow. The computational graph enables automatic differentiation: by traversing the graph backward from the loss node, we can compute gradients for all parameters using the chain rule of calculus.

### The Forward Pass

The forward pass is the process of computing predictions by passing input data through the network from input to output. Think of the forward pass as following a recipe step-by-step. You start with ingredients (input data), process them through each step (each layer), and end with the final dish (the prediction). Data flows in one direction: Input → Layer 1 → Layer 2 → ... → Output.

In a neural network, the forward pass follows a specific sequence of transformations. First, the input data is received by the input layer. Then each hidden layer processes the data, applying weight matrices, adding biases, and applying activation functions. The output layer produces the final predictions. For classification tasks, the output is typically converted to probabilities using a softmax function. For a concrete classification example, see [Example 7: Character Recognition](15-example7-character-recognition.md), which shows how pixel inputs become class predictions through the same process.


| |
|:---:|
| ![Forward Pass Flow](images/flow-diagrams/forward-pass-flow.svg) |


We trace through a complete forward pass example using arbitrary values chosen to illustrate the transformation process:

**Note:** This is an illustrative example. The weight matrices ($\mathbf{W}_1$, $\mathbf{W}_2$, $\mathbf{W}_{\text{out}}$) and bias vectors ($\mathbf{b}_1$, $\mathbf{b}_2$, $\mathbf{b}_{\text{out}}$) are not shown explicitly here—only the final outputs after applying the full computation $\mathbf{y} = f(\mathbf{W}\mathbf{x} + \mathbf{b})$ are displayed. This keeps the example concise while demonstrating the forward pass flow.

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

To see the forward pass in detail, [Example 1: Minimal Forward Pass](09-example1-forward-pass.md) shows the complete forward pass computation step by step.

### The Backward Pass

The backward pass (also called backpropagation) is the process of computing gradients by propagating the loss backward through the network from output to input. This is where the magic of learning happens—we figure out how each parameter contributed to the error and how much it should change.

In a neural network, the backward pass follows this sequence. First, we compute the loss at the output by comparing the prediction to the target. Then we compute the gradient with respect to the output. Next, we propagate this gradient backward through each layer, using the chain rule of calculus. The gradient flows from the loss through the output layer, then through each hidden layer in reverse order, and finally to the input layer. At each step, we compute the gradient for each parameter (weight and bias). Once we have all the gradients, we use them to update the parameters using gradient descent.

The chain rule enables this backward propagation. For a two-layer network with output $y = f_2(f_1(\mathbf{x}; \mathbf{W}_1); \mathbf{W}_2)$, the gradient with respect to $\mathbf{W}_1$ is computed as:

$$\frac{\partial L}{\partial \mathbf{W}_1} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial f_1} \frac{\partial f_1}{\partial \mathbf{W}_1}$$

Each term in this product is computed during the backward pass, with gradients flowing from the output back to the input. This is exactly what backpropagation does: it applies the chain rule at each layer, computing gradients layer by layer from output to input.


| |
|:---:|
| ![Backward Pass Flow](images/flow-diagrams/backward-pass-flow.svg) |


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

To see the complete backward pass in action, [Example 3: Full Backpropagation](11-example3-full-backprop.md) shows the complete gradient flow through all components of a multi-layer network.

---

## Chapter Summary

In this chapter, we've covered the three fundamental learning algorithms:

1. **Loss functions** measure how wrong the model's predictions are, providing the signal that guides learning.

2. **Gradient descent** uses gradients to iteratively update parameters, moving toward better predictions.

3. **Backpropagation** computes the gradients needed for gradient descent by propagating the loss backward through the network using the chain rule.

These three algorithms work together: loss functions tell us what's wrong, backpropagation tells us how to fix it, and gradient descent actually makes the fix. In the next chapter, we'll see how these algorithms combine into a complete training loop, including batch processing and the transition from neural networks to transformers.

---

**Navigation:**
- [← Previous: Multilayer Networks and Architecture](02-multilayer-networks-architecture.md) | [← Index](00-index.md) | [Next: Training Neural Networks →](04-training-neural-networks.md)

