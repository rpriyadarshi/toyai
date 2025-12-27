# Example 7: Character Recognition

## Goal

Understand how neural networks perform classification tasks using continuous activations. This example demonstrates that the same fundamental principles (ReLU, softmax, classification) learned in transformer examples apply to other neural network architectures.

## What You'll Learn

- Feed-forward neural network architecture
- ReLU activation function in hidden layers
- How continuous outputs become discrete predictions via softmax
- The complete forward pass for image classification
- Connection to transformer concepts from Examples 1-6

## Model Architecture

```
Input (4 pixels) → Hidden Layer (2 neurons, ReLU) → Output Layer (4 logits) → Softmax → Probabilities → Prediction
```

**Components:**
- **Input Layer**: 4 pixels from a 2×2 image
- **Hidden Layer**: 2 neurons with ReLU activation
- **Output Layer**: 4 logits (one per digit class: 0, 1, 2, 3)
- **Softmax**: Converts logits to probabilities
- **Decision**: Select class with highest probability (argmax)

## Key Concepts

### 1. Continuous Activations

ReLU outputs continuous values, not binary decisions:
- ReLU(0.39) = 0.39 (continuous)
- ReLU(-0.5) = 0.0 (continuous)
- **No "firing"** - neurons output continuous values throughout

### 2. Softmax Classification

Same function used in transformers:
- Converts logits to probabilities
- All probabilities sum to 1.0
- Largest logit gets highest probability

### 3. Discrete Decision

Made via argmax on continuous probabilities:
- Probabilities: [0.25, 0.27, 0.24, 0.24] (continuous)
- Prediction: Digit 1 (discrete selection)

### 4. Universal Principles

The same fundamentals apply:
- **Transformers**: Continuous activations → softmax → discrete token selection
- **Classification**: Continuous activations → softmax → discrete class selection

## Running the Example

```bash
cd build
make example7
./examples/example7_character_recognition/example7
```

## Hand Calculation

See `../../worksheets/example7_worksheet.md` for step-by-step hand calculation guide.

## Expected Output

The model will:
1. Process a 2×2 pixel image (vertical line pattern)
2. Compute hidden layer with ReLU activation
3. Compute output logits for 4 digit classes
4. Apply softmax to get probabilities
5. Predict the digit with highest probability

**Key Output:**
- All intermediate values are continuous
- No binary "firing" decisions
- Softmax produces probability distribution
- Argmax selects discrete class

## Connection to Transformers

| Concept | Transformers (Examples 1-6) | Classification (Example 7) |
|---------|----------------------------|---------------------------|
| **Activation** | ReLU in FFN layers | ReLU in hidden layer |
| **Softmax** | Attention weights, output probabilities | Class probabilities |
| **Continuous** | All activations are continuous | All activations are continuous |
| **Discrete Decision** | Select token with highest probability | Select class with highest probability |
| **No "Firing"** | Neurons output continuous values | Neurons output continuous values |

## Next Steps

- **Review Examples 1-6**: See how the same principles apply to transformers
- **Try different patterns**: Modify input image and trace through forward pass
- **Understand continuity**: Explain why all values are continuous (no "firing")

## Key Takeaways

1. **Continuous Activations**: ReLU outputs continuous values, not binary decisions
2. **Softmax Classification**: Same function used in transformers and classification
3. **Universal Principles**: Neural network fundamentals apply across architectures
4. **No "Firing"**: Modern networks use continuous activations throughout
5. **Discrete Decisions**: Made via argmax on continuous probabilities

