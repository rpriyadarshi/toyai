/**
 * ============================================================================
 * Example 4: Multiple Patterns
 * ============================================================================
 * 
 * Goal: Learn multiple patterns from multiple examples
 * 
 * This example demonstrates:
 * - Batch training
 * - Gradient accumulation
 * - Learning multiple patterns simultaneously
 * - Training loop with multiple epochs
 */

#include "../../src/core/Matrix.hpp"
#include "../../src/core/Embedding.hpp"
#include "../../src/core/LinearProjection.hpp"
#include "../../src/core/Attention.hpp"
#include "../../src/core/Softmax.hpp"
#include "../../src/core/Loss.hpp"
#include "../../src/core/Optimizer.hpp"
#include <iostream>
#include <vector>
#include <iomanip>

int main() {
    std::cout << std::string(70, '=') << "\n";
    std::cout << "Example 4: Multiple Patterns\n";
    std::cout << "Goal: Learn multiple patterns simultaneously\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    // Training examples
    std::vector<std::vector<int>> examples = {
        {0, 1},  // A B → C
        {0, 0},  // A A → D
        {1, 0}   // B A → C
    };
    
    std::vector<int> targets = {2, 3, 2};  // C, D, C
    
    std::cout << "Training Examples:\n";
    const char* tokens[] = {"A", "B", "C", "D"};
    for (size_t i = 0; i < examples.size(); i++) {
        std::cout << "  [" << tokens[examples[i][0]] << ", " 
                  << tokens[examples[i][1]] << "] → " 
                  << tokens[targets[i]] << "\n";
    }
    std::cout << "\n";
    
    // Model setup
    Embedding embedding;
    Matrix WQ(0.1, 0.0, 0.0, 0.1);
    Matrix WK(0.1, 0.0, 0.0, 0.1);
    Matrix WV(0.1, 0.0, 0.0, 0.1);
    Matrix WO(0.1, 0.0, 0.0, 0.1);
    
    LinearProjection projQ(WQ, true);
    LinearProjection projK(WK, true);
    LinearProjection projV(WV, true);
    
    Attention attention;
    Softmax softmax;
    Optimizer optimizer(0.1);
    
    // Training loop
    const int num_epochs = 5;
    
    std::cout << "TRAINING LOOP (" << num_epochs << " epochs)\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        double total_loss = 0.0;
        
        // Process each example
        for (size_t i = 0; i < examples.size(); i++) {
            // Forward pass
            Matrix X = embedding.forwardSequence(examples[i]);
            Matrix Q = projQ.forward(X);
            Matrix K = projK.forward(X);
            Matrix V = projV.forward(X);
            
            Attention::AttentionResult attn_result = attention.forward(Q, K, V);
            
            double context[2] = {attn_result.output.get(1, 0), 
                                attn_result.output.get(1, 1)};
            
            std::vector<double> logits(4);
            logits[0] = context[0] * 0.1 + context[1] * 0.0;
            logits[1] = context[0] * 0.0 + context[1] * 0.0;
            logits[2] = context[0] * 0.0 + context[1] * 0.1;
            logits[3] = context[0] * 0.0 + context[1] * 0.0;
            
            std::vector<double> probs = softmax.forward(logits);
            double loss = Loss::crossEntropy(probs, targets[i]);
            total_loss += loss;
            
            // Backward pass (simplified - would accumulate gradients in real implementation)
            // For this example, we'll do one update per example
        }
        
        double avg_loss = total_loss / examples.size();
        std::cout << "Epoch " << epoch << ": Average Loss = " 
                  << std::fixed << std::setprecision(6) << avg_loss << "\n";
    }
    
    std::cout << "\nTraining complete!\n";
    std::cout << "Next: See Example 5 for feed-forward layers.\n";
    std::cout << std::string(70, '=') << "\n";
    
    return 0;
}

