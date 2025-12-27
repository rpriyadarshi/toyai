/**
 * ============================================================================
 * Example 7: Character Recognition
 * ============================================================================
 * 
 * Goal: Understand how neural networks perform classification tasks
 * 
 * This example demonstrates:
 * - Feed-forward network architecture
 * - ReLU activation in hidden layers
 * - Softmax for classification
 * - How continuous activations lead to discrete predictions
 * 
 * Architecture: 4 inputs → 2 hidden (ReLU) → 4 outputs (softmax)
 * All computations use 2×2 matrices that can be verified by hand.
 */

#include "../../src/core/Matrix.hpp"
#include "../../src/core/Softmax.hpp"
#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <cmath>

/**
 * Apply ReLU activation element-wise: max(0, x)
 */
Matrix applyReLU(const Matrix& m) {
    Matrix result;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            result.data[i][j] = std::max(0.0, m.data[i][j]);
        }
    }
    return result;
}

/**
 * Print vector with label
 */
void printVector(const std::vector<double>& vec, const std::string& name) {
    std::cout << name << " = [";
    for (size_t i = 0; i < vec.size(); i++) {
        std::cout << std::fixed << std::setprecision(3) << vec[i];
        if (i < vec.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}

int main() {
    std::cout << std::string(70, '=') << "\n";
    std::cout << "Example 7: Character Recognition\n";
    std::cout << "Goal: Understand classification with continuous activations\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    // ========================================================================
    // Setup: Define the model
    // ========================================================================
    
    std::cout << "MODEL SETUP\n";
    std::cout << std::string(70, '-') << "\n\n";
    
    std::cout << "Architecture:\n";
    std::cout << "  Input: 4 pixels (2×2 image)\n";
    std::cout << "  Hidden: 2 neurons with ReLU activation\n";
    std::cout << "  Output: 4 logits (digits 0, 1, 2, 3)\n";
    std::cout << "  Softmax: Converts logits to probabilities\n\n";
    
    // Input: 2×2 image representing digit "1" (vertical line)
    // We'll represent 4 pixels as: [pixel0, pixel1; pixel2, pixel3]
    // Pattern: vertical line = [0.1, 0.9; 0.1, 0.9]
    Matrix input(0.1, 0.9, 0.1, 0.9);
    std::cout << "Input Image (2×2 pixels, vertical line pattern):\n";
    input.print("Input");
    std::cout << "  Interpretation: [0.1, 0.9; 0.1, 0.9] = vertical line (digit '1')\n";
    std::cout << "  Pixel values: 0.1 = light, 0.9 = dark\n\n";
    
    // Weight matrices (simplified to fit 2×2)
    // W1: Maps 4 inputs to 2 hidden neurons
    // We'll use: W1 maps input matrix to hidden vector
    // For simplicity, we'll compute: hidden[i] = sum(W1[i][j] * input[j])
    Matrix W1(0.2, 0.3, 0.1, 0.4);  // Weights for hidden layer
    Matrix b1(0.1, 0.05, 0.0, 0.0);  // Bias (only first 2 elements used)
    
    std::cout << "Hidden Layer Weights:\n";
    W1.print("W1");
    b1.print("b1 (bias, first 2 elements used)");
    std::cout << "\n";
    
    // W2: Maps 2 hidden neurons to 4 output logits
    // We'll use two 2×2 operations or adapt
    Matrix W2(0.3, 0.2, 0.1, 0.4);  // Weights for output layer (adapted)
    Matrix b2(0.05, 0.1, 0.0, 0.0);  // Bias (first 2 elements used, then replicated)
    
    std::cout << "Output Layer Weights:\n";
    W2.print("W2");
    b2.print("b2 (bias, adapted for 4 outputs)");
    std::cout << "\n";
    
    Softmax softmax;
    
    // ========================================================================
    // Forward Pass: Classify digit
    // ========================================================================
    
    std::cout << std::string(70, '=') << "\n";
    std::cout << "FORWARD PASS: Classify 2×2 image\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    // Step 1: Input
    std::cout << "Step 1: Input Pixels\n";
    std::cout << std::string(70, '-') << "\n";
    std::cout << "Input image (2×2):\n";
    input.print("Input");
    std::cout << "  Flattened: [0.1, 0.9, 0.1, 0.9]\n";
    std::cout << "  Pattern: Vertical line (represents digit '1')\n\n";
    
    // Step 2: Hidden layer
    // Compute: hidden = W1 × input_flat + b1
    // Since we have 4 inputs and 2 hidden, we'll compute manually
    std::cout << "Step 2: Hidden Layer (W1 × input + b1)\n";
    std::cout << std::string(70, '-') << "\n";
    
    // Extract input as vector: [0.1, 0.9, 0.1, 0.9]
    std::vector<double> input_vec = {0.1, 0.9, 0.1, 0.9};
    
    // Compute hidden layer (2 neurons)
    // hidden[0] = W1[0][0]*input[0] + W1[0][1]*input[1] + W1[1][0]*input[2] + W1[1][1]*input[3] + b1[0]
    // hidden[1] = W1[0][0]*input[0] + W1[0][1]*input[1] + W1[1][0]*input[2] + W1[1][1]*input[3] + b1[1]
    // Actually, let's use a simpler mapping:
    // hidden[0] = W1[0][0]*input[0] + W1[0][1]*input[1] + b1[0]
    // hidden[1] = W1[1][0]*input[2] + W1[1][1]*input[3] + b1[1]
    
    std::vector<double> hidden(2);
    hidden[0] = W1.get(0, 0) * input_vec[0] + W1.get(0, 1) * input_vec[1] + b1.get(0, 0);
    hidden[1] = W1.get(1, 0) * input_vec[2] + W1.get(1, 1) * input_vec[3] + b1.get(0, 1);
    
    std::cout << "Computation:\n";
    std::cout << "  hidden[0] = W1[0][0]×input[0] + W1[0][1]×input[1] + b1[0]\n";
    std::cout << "            = " << W1.get(0, 0) << "×" << input_vec[0] 
              << " + " << W1.get(0, 1) << "×" << input_vec[1] 
              << " + " << b1.get(0, 0) << "\n";
    std::cout << "            = " << (W1.get(0, 0) * input_vec[0]) << " + " 
              << (W1.get(0, 1) * input_vec[1]) << " + " << b1.get(0, 0) << "\n";
    std::cout << "            = " << hidden[0] << "\n\n";
    
    std::cout << "  hidden[1] = W1[1][0]×input[2] + W1[1][1]×input[3] + b1[1]\n";
    std::cout << "            = " << W1.get(1, 0) << "×" << input_vec[2] 
              << " + " << W1.get(1, 1) << "×" << input_vec[3] 
              << " + " << b1.get(0, 1) << "\n";
    std::cout << "            = " << (W1.get(1, 0) * input_vec[2]) << " + " 
              << (W1.get(1, 1) * input_vec[3]) << " + " << b1.get(0, 1) << "\n";
    std::cout << "            = " << hidden[1] << "\n\n";
    
    printVector(hidden, "Hidden (before ReLU)");
    std::cout << "\n";
    
    // Step 3: ReLU activation
    std::cout << "Step 3: ReLU Activation (max(0, x))\n";
    std::cout << std::string(70, '-') << "\n";
    
    std::vector<double> hidden_relu(2);
    hidden_relu[0] = std::max(0.0, hidden[0]);
    hidden_relu[1] = std::max(0.0, hidden[1]);
    
    std::cout << "ReLU applied element-wise:\n";
    std::cout << "  hidden_relu[0] = max(0, " << hidden[0] << ") = " << hidden_relu[0] << "\n";
    std::cout << "  hidden_relu[1] = max(0, " << hidden[1] << ") = " << hidden_relu[1] << "\n\n";
    
    printVector(hidden_relu, "Hidden (after ReLU)");
    std::cout << "  Note: All values are continuous (not binary 'firing')\n\n";
    
    // Step 4: Output layer
    std::cout << "Step 4: Output Layer (W2 × hidden + b2)\n";
    std::cout << std::string(70, '-') << "\n";
    
    // Compute logits for 4 classes
    // We'll use W2 to map 2 hidden to 4 outputs
    // logit[0] = W2[0][0]*hidden[0] + W2[0][1]*hidden[1] + b2[0]
    // logit[1] = W2[1][0]*hidden[0] + W2[1][1]*hidden[1] + b2[1]
    // logit[2] = W2[0][0]*hidden[0] + W2[0][1]*hidden[1] + b2[0] (reuse, simplified)
    // logit[3] = W2[1][0]*hidden[0] + W2[1][1]*hidden[1] + b2[1] (reuse, simplified)
    
    std::vector<double> logits(4);
    logits[0] = W2.get(0, 0) * hidden_relu[0] + W2.get(0, 1) * hidden_relu[1] + b2.get(0, 0);
    logits[1] = W2.get(1, 0) * hidden_relu[0] + W2.get(1, 1) * hidden_relu[1] + b2.get(0, 1);
    logits[2] = W2.get(0, 0) * hidden_relu[0] + W2.get(0, 1) * hidden_relu[1] + b2.get(1, 0);
    logits[3] = W2.get(1, 0) * hidden_relu[0] + W2.get(1, 1) * hidden_relu[1] + b2.get(1, 1);
    
    std::cout << "Computation:\n";
    std::cout << "  logit[0] = W2[0][0]×hidden[0] + W2[0][1]×hidden[1] + b2[0]\n";
    std::cout << "           = " << W2.get(0, 0) << "×" << hidden_relu[0] 
              << " + " << W2.get(0, 1) << "×" << hidden_relu[1] 
              << " + " << b2.get(0, 0) << " = " << logits[0] << "\n";
    std::cout << "  logit[1] = W2[1][0]×hidden[0] + W2[1][1]×hidden[1] + b2[1]\n";
    std::cout << "           = " << W2.get(1, 0) << "×" << hidden_relu[0] 
              << " + " << W2.get(1, 1) << "×" << hidden_relu[1] 
              << " + " << b2.get(0, 1) << " = " << logits[1] << "\n";
    std::cout << "  logit[2] = W2[0][0]×hidden[0] + W2[0][1]×hidden[1] + b2[2]\n";
    std::cout << "           = " << W2.get(0, 0) << "×" << hidden_relu[0] 
              << " + " << W2.get(0, 1) << "×" << hidden_relu[1] 
              << " + " << b2.get(1, 0) << " = " << logits[2] << "\n";
    std::cout << "  logit[3] = W2[1][0]×hidden[0] + W2[1][1]×hidden[1] + b2[3]\n";
    std::cout << "           = " << W2.get(1, 0) << "×" << hidden_relu[0] 
              << " + " << W2.get(1, 1) << "×" << hidden_relu[1] 
              << " + " << b2.get(1, 1) << " = " << logits[3] << "\n\n";
    
    const char* class_names[] = {"Digit 0", "Digit 1", "Digit 2", "Digit 3"};
    std::cout << "Logits (raw scores):\n";
    for (int i = 0; i < 4; i++) {
        std::cout << "  " << class_names[i] << ": " << std::fixed << std::setprecision(4) << logits[i] << "\n";
    }
    std::cout << "\n";
    
    // Step 5: Softmax
    std::cout << "Step 5: Softmax (Logits → Probabilities)\n";
    std::cout << std::string(70, '-') << "\n";
    std::vector<double> probs = softmax.forward(logits);
    softmax.printSteps(logits, probs);
    
    // Step 6: Prediction
    std::cout << "Step 6: Prediction (Argmax)\n";
    std::cout << std::string(70, '-') << "\n";
    
    int predicted_class = std::distance(probs.begin(), 
                                       std::max_element(probs.begin(), probs.end()));
    
    std::cout << "Select class with highest probability:\n";
    for (int i = 0; i < 4; i++) {
        std::cout << "  " << class_names[i] << ": " 
                  << std::fixed << std::setprecision(4) << probs[i] 
                  << " (" << (probs[i] * 100) << "%)";
        if (i == predicted_class) std::cout << " ← PREDICTED";
        std::cout << "\n";
    }
    std::cout << "\n";
    std::cout << "Prediction: " << class_names[predicted_class] 
              << " (confidence: " << std::fixed << std::setprecision(1) 
              << (probs[predicted_class] * 100) << "%)\n\n";
    
    // ========================================================================
    // Summary
    // ========================================================================
    
    std::cout << std::string(70, '=') << "\n";
    std::cout << "SUMMARY\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    std::cout << "Input: 2×2 image (vertical line pattern)\n";
    std::cout << "Predicted class: " << class_names[predicted_class] << "\n";
    std::cout << "Confidence: " << std::fixed << std::setprecision(1) 
              << (probs[predicted_class] * 100) << "%\n\n";
    
    std::cout << "Key Insights:\n";
    std::cout << "1. All activations are continuous (ReLU outputs continuous values)\n";
    std::cout << "2. No 'firing' - neurons don't output binary 0/1\n";
    std::cout << "3. Softmax converts continuous logits to probabilities\n";
    std::cout << "4. Discrete decision made via argmax on continuous probabilities\n";
    std::cout << "5. Same principles apply to transformers (Examples 1-6)\n\n";
    
    std::cout << "This demonstrates that neural network fundamentals are universal,\n";
    std::cout << "not transformer-specific. The same continuous activations and\n";
    std::cout << "softmax classification work the same way in both contexts.\n\n";
    
    std::cout << std::string(70, '=') << "\n";
    
    return 0;
}

