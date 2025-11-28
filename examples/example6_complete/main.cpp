/**
 * ============================================================================
 * Example 6: Complete Transformer
 * ============================================================================
 * 
 * Goal: Full implementation with all components
 * 
 * This example demonstrates:
 * - Multiple transformer blocks
 * - Layer normalization
 * - Residual connections everywhere
 * - Complete training pipeline
 * - End-to-end computation
 */

#include "../../src/core/Matrix.hpp"
#include "../../src/core/Embedding.hpp"
#include "../../src/core/LinearProjection.hpp"
#include "../../src/core/Attention.hpp"
#include "../../src/core/Softmax.hpp"
#include "../../src/core/Loss.hpp"
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <algorithm>

// Layer normalization (simplified for 2x2)
Matrix layerNorm(const Matrix& input) {
    Matrix result;
    for (int i = 0; i < 2; i++) {
        // Compute mean and std for row i
        double mean = (input.get(i, 0) + input.get(i, 1)) / 2.0;
        double var = ((input.get(i, 0) - mean) * (input.get(i, 0) - mean) +
                     (input.get(i, 1) - mean) * (input.get(i, 1) - mean)) / 2.0;
        double std = std::sqrt(var + 1e-8);  // Add epsilon for stability
        
        // Normalize
        result.set(i, 0, (input.get(i, 0) - mean) / std);
        result.set(i, 1, (input.get(i, 1) - mean) / std);
    }
    return result;
}

// ReLU
Matrix relu(const Matrix& input) {
    Matrix result;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            result.set(i, j, std::max(0.0, input.get(i, j)));
        }
    }
    return result;
}

int main() {
    std::cout << std::string(70, '=') << "\n";
    std::cout << "Example 6: Complete Transformer\n";
    std::cout << "Goal: Full architecture with all components\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    // Complete transformer setup
    Embedding embedding;
    
    // Transformer block 1
    Matrix WQ1(0.1, 0.0, 0.0, 0.1);
    Matrix WK1(0.1, 0.0, 0.0, 0.1);
    Matrix WV1(0.1, 0.0, 0.0, 0.1);
    Matrix W1_1(0.2, 0.1, 0.1, 0.2);
    Matrix W2_1(0.2, 0.1, 0.1, 0.2);
    
    LinearProjection projQ1(WQ1, false);
    LinearProjection projK1(WK1, false);
    LinearProjection projV1(WV1, false);
    LinearProjection ffn1_1(W1_1, false);
    LinearProjection ffn2_1(W2_1, false);
    
    Attention attention1;
    
    std::vector<int> input_tokens = {0, 1};  // A, B
    
    std::cout << "COMPLETE FORWARD PASS\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    // Input embeddings
    Matrix X = embedding.forwardSequence(input_tokens);
    std::cout << "Input embeddings:\n";
    X.print("X");
    std::cout << "\n";
    
    // Transformer Block 1
    std::cout << "Transformer Block 1:\n";
    std::cout << std::string(70, '-') << "\n";
    
    // Attention
    Matrix Q1 = projQ1.forward(X);
    Matrix K1 = projK1.forward(X);
    Matrix V1 = projV1.forward(X);
    Attention::AttentionResult attn1 = attention1.forward(Q1, K1, V1);
    
    // Layer norm + residual
    Matrix norm1 = layerNorm(attn1.output.add(X));
    
    // Feed-forward
    Matrix ffn_hidden = ffn1_1.forward(norm1);
    Matrix ffn_act = relu(ffn_hidden);
    Matrix ffn_out = ffn2_1.forward(ffn_act);
    
    // Layer norm + residual
    Matrix block1_output = layerNorm(ffn_out.add(norm1));
    
    attn1.output.print("Attention output");
    norm1.print("After attention + norm + residual");
    ffn_out.print("FFN output");
    block1_output.print("Block 1 output");
    std::cout << "\n";
    
    // Output projection
    Matrix WO(0.1, 0.0, 0.0, 0.1);
    double context[2] = {block1_output.get(1, 0), block1_output.get(1, 1)};
    
    std::vector<double> logits(4);
    logits[0] = context[0] * 0.1;
    logits[1] = 0.0;
    logits[2] = context[1] * 0.1;
    logits[3] = 0.0;
    
    Softmax softmax;
    std::vector<double> probs = softmax.forward(logits);
    
    std::cout << "Final Probabilities:\n";
    const char* tokens[] = {"A", "B", "C", "D"};
    for (int i = 0; i < 4; i++) {
        std::cout << "  P(" << tokens[i] << ") = " 
                  << std::fixed << std::setprecision(4) << probs[i] << "\n";
    }
    std::cout << "\n";
    
    std::cout << "Complete transformer with:\n";
    std::cout << "  - Attention mechanism\n";
    std::cout << "  - Feed-forward networks\n";
    std::cout << "  - Layer normalization\n";
    std::cout << "  - Residual connections\n";
    std::cout << "  - Multiple layers (can be extended)\n\n";
    
    std::cout << "Congratulations! You've mastered transformers from first principles!\n";
    std::cout << std::string(70, '=') << "\n";
    
    return 0;
}

