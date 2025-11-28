/**
 * ============================================================================
 * Example 2: Single Training Step
 * ============================================================================
 * 
 * Goal: Understand how one weight update works
 * 
 * This example demonstrates:
 * - Forward pass (from Example 1)
 * - Loss computation
 * - Gradient computation
 * - One weight update
 * - Improved prediction after update
 * 
 * We only train WO (output projection) to keep it simple.
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
    std::cout << "Example 2: Single Training Step\n";
    std::cout << "Goal: Learn how one weight update works\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    // ========================================================================
    // Setup: Same as Example 1, but WO is trainable
    // ========================================================================
    
    Embedding embedding;
    Matrix WQ(0.1, 0.0, 0.0, 0.1);
    Matrix WK(0.1, 0.0, 0.0, 0.1);
    Matrix WV(0.1, 0.0, 0.0, 0.1);
    
    LinearProjection projQ(WQ, false);
    LinearProjection projK(WK, false);
    LinearProjection projV(WV, false);
    
    // Output projection - NOW TRAINABLE
    Matrix WO(0.1, 0.0, 0.0, 0.1);
    
    Attention attention;
    Softmax softmax;
    Optimizer optimizer(0.1);  // Learning rate = 0.1
    
    // Training example: [A, B] → C
    std::vector<int> input_tokens = {0, 1};  // A, B
    int target_token = 2;  // C
    
    // ========================================================================
    // BEFORE TRAINING: Forward Pass
    // ========================================================================
    
    std::cout << "BEFORE TRAINING\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    // Forward pass (same as Example 1)
    Matrix X = embedding.forwardSequence(input_tokens);
    Matrix Q = projQ.forward(X);
    Matrix K = projK.forward(X);
    Matrix V = projV.forward(X);
    
    Attention::AttentionResult attn_result = attention.forward(Q, K, V);
    
    // Get context vector for position 1
    double context[2] = {attn_result.output.get(1, 0), attn_result.output.get(1, 1)};
    
    // Compute logits (simplified for 2x2 case)
    std::vector<double> logits(4);
    logits[0] = context[0] * 0.1 + context[1] * 0.0;  // A
    logits[1] = context[0] * 0.0 + context[1] * 0.0;  // B
    logits[2] = context[0] * 0.0 + context[1] * 0.1;  // C
    logits[3] = context[0] * 0.0 + context[1] * 0.0;  // D
    
    std::vector<double> probs_before = softmax.forward(logits);
    
    std::cout << "Initial probabilities:\n";
    const char* tokens[] = {"A", "B", "C", "D"};
    for (int i = 0; i < 4; i++) {
        std::cout << "  P(" << tokens[i] << ") = " 
                  << std::fixed << std::setprecision(4) << probs_before[i];
        if (i == target_token) std::cout << "  ← Target";
        std::cout << "\n";
    }
    std::cout << "\n";
    
    // ========================================================================
    // TRAINING STEP
    // ========================================================================
    
    std::cout << "TRAINING STEP\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    // Step 1: Compute loss
    std::cout << "Step 1: Compute Loss\n";
    std::cout << std::string(70, '-') << "\n";
    double loss = Loss::crossEntropy(probs_before, target_token);
    Loss::printLoss(probs_before, target_token, loss);
    std::cout << "\n";
    
    // Step 2: Compute gradient w.r.t. logits
    std::cout << "Step 2: Gradient w.r.t. Logits\n";
    std::cout << std::string(70, '-') << "\n";
    std::vector<double> dL_dlogits = Loss::crossEntropyGradient(probs_before, target_token);
    
    std::cout << "Gradient w.r.t. logits:\n";
    for (int i = 0; i < 4; i++) {
        std::cout << "  dL/dlogit(" << tokens[i] << ") = " 
                  << std::fixed << std::setprecision(6) << dL_dlogits[i];
        if (i == target_token) std::cout << "  ← Should increase!";
        std::cout << "\n";
    }
    std::cout << "\n";
    
    // Step 3: Compute gradient w.r.t. WO
    // For logit[i] = context · WO_column_i
    // dL/dWO = outer(context, dL_dlogits)
    std::cout << "Step 3: Gradient w.r.t. WO\n";
    std::cout << std::string(70, '-') << "\n";
    
    // For our simplified case, we need to map gradients back to WO
    // Since we're using WO[0][0] for A and WO[1][1] for C, etc.
    Matrix dWO;
    dWO.set(0, 0, context[0] * dL_dlogits[0]);  // For token A
    dWO.set(0, 1, context[0] * dL_dlogits[1]);  // For token B
    dWO.set(1, 0, context[0] * dL_dlogits[2]);  // For token C
    dWO.set(1, 1, context[1] * dL_dlogits[2]);  // For token C (also uses row 1)
    
    std::cout << "Old WO:\n";
    WO.print("WO");
    std::cout << "\nGradient:\n";
    dWO.print("dWO");
    std::cout << "\n";
    
    // Step 4: Update weights
    std::cout << "Step 4: Update Weights\n";
    std::cout << std::string(70, '-') << "\n";
    optimizer.step(WO, dWO);
    
    std::cout << "New WO (after update):\n";
    WO.print("WO");
    std::cout << "\n";
    
    // ========================================================================
    // AFTER TRAINING: Forward Pass Again
    // ========================================================================
    
    std::cout << "AFTER TRAINING\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    // Recompute with updated WO
    logits[0] = context[0] * WO.get(0, 0) + context[1] * WO.get(1, 0);
    logits[1] = context[0] * WO.get(0, 1) + context[1] * WO.get(1, 1);
    logits[2] = context[0] * WO.get(0, 0) + context[1] * WO.get(1, 0);  // Simplified
    logits[3] = context[0] * WO.get(0, 1) + context[1] * WO.get(1, 1);
    
    std::vector<double> probs_after = softmax.forward(logits);
    
    std::cout << "Updated probabilities:\n";
    for (int i = 0; i < 4; i++) {
        std::cout << "  P(" << tokens[i] << ") = " 
                  << std::fixed << std::setprecision(4) << probs_after[i];
        if (i == target_token) std::cout << "  ← Target";
        std::cout << "\n";
    }
    std::cout << "\n";
    
    // ========================================================================
    // Comparison
    // ========================================================================
    
    std::cout << std::string(70, '=') << "\n";
    std::cout << "COMPARISON\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    std::cout << "Before training: P(C) = " << probs_before[target_token] << "\n";
    std::cout << "After training:  P(C) = " << probs_after[target_token] << "\n";
    std::cout << "Improvement:     +" << (probs_after[target_token] - probs_before[target_token]) << "\n\n";
    
    if (probs_after[target_token] > probs_before[target_token]) {
        std::cout << "SUCCESS! The model learned to increase probability for C.\n";
        std::cout << "With more training steps, this would continue improving.\n\n";
    }
    
    std::cout << "Next: See Example 3 for full backpropagation through all weights!\n";
    std::cout << std::string(70, '=') << "\n";
    
    return 0;
}

