/**
 * ============================================================================
 * TRANSFORMER ATTENTION: THE MATRIX CORE OF GENERATIVE AI
 * ============================================================================
 * 
 * This file demonstrates the fundamental matrix operations at the heart of 
 * transformer models (GPT, BERT, etc.) using 2x2 matrices that can be 
 * verified by hand on paper.
 * 
 * Author: toyai educational project
 * Purpose: Demystify WHY transformers work, not just HOW
 * 
 * ============================================================================
 * THE BIG PICTURE: WHAT IS GENERATIVE AI'S MATRIX CORE?
 * ============================================================================
 * 
 * Generative AI models like GPT are fundamentally about:
 * 1. REPRESENTING information as vectors (embeddings)
 * 2. RELATING information via attention (weighted similarity)
 * 3. TRANSFORMING information via linear projections (matrix multiply)
 * 
 * The "matrix core" refers to the dense matrix multiplications that:
 * - Project inputs to Query (Q), Key (K), Value (V) spaces
 * - Compute attention scores via Q×K^T
 * - Aggregate information via scores×V
 * - Transform outputs through feedforward layers
 * 
 * WHY matrices? Because:
 * - Linear transformations preserve structure while changing representation
 * - Matrix multiply is parallelizable (GPUs excel at this)
 * - Gradients flow cleanly through linear operations
 * 
 * ============================================================================
 * HOW IS IT ORGANIZED? THE ATTENTION MECHANISM
 * ============================================================================
 * 
 * Self-attention computes: Attention(Q,K,V) = softmax(Q×K^T / √d_k) × V
 * 
 * Organization:
 * 1. INPUT: Token embeddings (each token is a vector)
 * 2. PROJECTION: Linear layers create Q, K, V from input
 *    - Q (Query): "What am I looking for?"
 *    - K (Key): "What do I have to offer?"
 *    - V (Value): "What information do I contain?"
 * 3. ATTENTION SCORES: Q×K^T measures compatibility
 *    - High score = tokens should pay attention to each other
 * 4. SCALING: Divide by √d_k to prevent extreme softmax values
 * 5. SOFTMAX: Convert scores to probabilities (sum to 1)
 * 6. WEIGHTED SUM: Attention × V aggregates relevant information
 * 
 * ============================================================================
 */

#include <iostream>
#include <cmath>
#include <iomanip>
#include <string>
#include <sstream>

/**
 * ============================================================================
 * MATRIX CLASS: The Foundation
 * ============================================================================
 * 
 * WHY a custom Matrix class?
 * - Educational clarity: See exactly what happens in each operation
 * - Hand-calculable: 2x2 matrices let you verify on paper
 * - Gradient tracking: Store derivatives for backpropagation
 */
class Matrix2x2 {
public:
    double data[2][2];
    double grad[2][2];  // Gradient storage for backpropagation
    
    Matrix2x2() {
        // WHY initialize to zero?
        // Uninitialized memory causes undefined behavior
        // Zero is a neutral starting point
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                data[i][j] = 0.0;
                grad[i][j] = 0.0;
            }
        }
    }
    
    Matrix2x2(double a, double b, double c, double d) {
        // Row-major layout: [a b; c d]
        // WHY this layout? Matches mathematical notation
        data[0][0] = a; data[0][1] = b;
        data[1][0] = c; data[1][1] = d;
        zeroGrad();
    }
    
    void zeroGrad() {
        // WHY zero gradients?
        // Gradients ACCUMULATE during backprop
        // Must reset before each training step
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                grad[i][j] = 0.0;
    }
    
    /**
     * Matrix Multiplication: C = A × B
     * 
     * WHY this operation is central to AI:
     * - Combines features: Each output element depends on ALL inputs
     * - Learns patterns: Weight matrices encode learned relationships
     * - Parallel: GPU tensor cores do this in hardware
     * 
     * Hand calculation for 2x2:
     * [a b]   [e f]   [ae+bg  af+bh]
     * [c d] × [g h] = [ce+dg  cf+dh]
     */
    Matrix2x2 matmul(const Matrix2x2& other) const {
        Matrix2x2 result;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                result.data[i][j] = 0.0;
                for (int k = 0; k < 2; k++) {
                    result.data[i][j] += data[i][k] * other.data[k][j];
                }
            }
        }
        return result;
    }
    
    /**
     * Transpose: Swap rows and columns
     * 
     * WHY transpose in attention?
     * - Q×K^T computes all pairwise dot products at once
     * - Dot product = similarity measure in vector space
     * - Without transpose, dimensions wouldn't align for matmul
     */
    Matrix2x2 transpose() const {
        Matrix2x2 result;
        result.data[0][0] = data[0][0];
        result.data[0][1] = data[1][0];  // swap
        result.data[1][0] = data[0][1];  // swap
        result.data[1][1] = data[1][1];
        return result;
    }
    
    /**
     * Scalar multiplication
     * 
     * WHY scale in attention?
     * - Prevents dot products from growing too large with dimension
     * - Large values → softmax saturates → vanishing gradients
     * - Scale factor √d_k keeps variance stable
     */
    Matrix2x2 scale(double s) const {
        Matrix2x2 result;
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                result.data[i][j] = data[i][j] * s;
        return result;
    }
    
    /**
     * Element-wise addition
     * 
     * WHY addition in neural nets?
     * - Residual connections: x + f(x) allows gradient flow
     * - Bias terms: Shift activation function
     */
    Matrix2x2 add(const Matrix2x2& other) const {
        Matrix2x2 result;
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                result.data[i][j] = data[i][j] + other.data[i][j];
        return result;
    }
    
    Matrix2x2 subtract(const Matrix2x2& other) const {
        Matrix2x2 result;
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                result.data[i][j] = data[i][j] - other.data[i][j];
        return result;
    }
    
    /**
     * Hadamard (element-wise) product
     * 
     * WHY element-wise multiply?
     * - Gating mechanisms (LSTM, GRU)
     * - Masking: Zero out certain positions
     * - Gradient computation uses this
     */
    Matrix2x2 hadamard(const Matrix2x2& other) const {
        Matrix2x2 result;
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                result.data[i][j] = data[i][j] * other.data[i][j];
        return result;
    }
    
    /**
     * Frobenius norm squared: sum of all elements squared
     * 
     * WHY this as loss?
     * - Mean Squared Error (MSE) is sum of squared differences
     * - Differentiable everywhere (unlike absolute value)
     * - Penalizes large errors more than small ones
     */
    double frobeniusNormSquared() const {
        double sum = 0.0;
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                sum += data[i][j] * data[i][j];
        return sum;
    }
    
    double sum() const {
        double s = 0.0;
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                s += data[i][j];
        return s;
    }
    
    void print(const std::string& name) const {
        std::cout << name << ":\n";
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  [" << data[0][0] << ", " << data[0][1] << "]\n";
        std::cout << "  [" << data[1][0] << ", " << data[1][1] << "]\n";
    }
    
    void printGrad(const std::string& name) const {
        std::cout << "d" << name << "/dL (gradient):\n";
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  [" << grad[0][0] << ", " << grad[0][1] << "]\n";
        std::cout << "  [" << grad[1][0] << ", " << grad[1][1] << "]\n";
    }
};

/**
 * ============================================================================
 * SOFTMAX: Converting Scores to Probabilities
 * ============================================================================
 * 
 * WHY softmax?
 * - Attention weights must sum to 1 (probability distribution)
 * - Exponential ensures all values are positive
 * - Differentiable (unlike argmax)
 * - Emphasizes larger values while keeping all non-zero
 * 
 * softmax(x_i) = exp(x_i) / sum(exp(x_j))
 * 
 * Applied row-wise: each row sums to 1
 * WHY row-wise? Each query (row) has its own attention distribution
 */
Matrix2x2 softmaxRows(const Matrix2x2& x) {
    Matrix2x2 result;
    
    for (int i = 0; i < 2; i++) {
        // Step 1: Find max for numerical stability
        // WHY subtract max? exp(large) overflows, but exp(x-max) is safe
        double maxVal = std::max(x.data[i][0], x.data[i][1]);
        
        // Step 2: Compute exponentials
        double exp0 = std::exp(x.data[i][0] - maxVal);
        double exp1 = std::exp(x.data[i][1] - maxVal);
        
        // Step 3: Normalize
        double sum = exp0 + exp1;
        result.data[i][0] = exp0 / sum;
        result.data[i][1] = exp1 / sum;
    }
    return result;
}

/**
 * ============================================================================
 * SOFTMAX BACKWARD: The Jacobian
 * ============================================================================
 * 
 * WHY is softmax gradient complex?
 * - Each output depends on ALL inputs (through the sum)
 * - The Jacobian has form: ds_i/dx_j = s_i(δ_ij - s_j)
 *   where δ_ij is Kronecker delta (1 if i=j, else 0)
 * 
 * For gradient dL/dx given dL/ds:
 * dL/dx_i = sum_j(dL/ds_j * ds_j/dx_i)
 *         = sum_j(dL/ds_j * s_j * (δ_ij - s_i))
 *         = dL/ds_i * s_i - s_i * sum_j(dL/ds_j * s_j)
 *         = s_i * (dL/ds_i - dot(dL/ds, s))
 */
Matrix2x2 softmaxBackward(const Matrix2x2& softmaxOut, const Matrix2x2& gradOut) {
    Matrix2x2 gradIn;
    
    for (int i = 0; i < 2; i++) {
        // Compute dot product of gradient and softmax output for this row
        double dot = gradOut.data[i][0] * softmaxOut.data[i][0] + 
                     gradOut.data[i][1] * softmaxOut.data[i][1];
        
        for (int j = 0; j < 2; j++) {
            gradIn.data[i][j] = softmaxOut.data[i][j] * (gradOut.data[i][j] - dot);
        }
    }
    return gradIn;
}

/**
 * ============================================================================
 * FORWARD PASS: Scaled Dot-Product Attention
 * ============================================================================
 * 
 * Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V
 * 
 * WHY this formula?
 * 
 * 1. Q × K^T: Computes similarity between every query-key pair
 *    - Q has shape (seq_len, d_k): each row is a query
 *    - K^T has shape (d_k, seq_len): each column is a key
 *    - Result: (seq_len, seq_len) similarity matrix
 *    
 * 2. / √d_k: Scaling factor
 *    - Dot products grow with dimension d_k
 *    - Variance of dot product ≈ d_k (when inputs have unit variance)
 *    - Dividing by √d_k keeps variance ≈ 1
 *    - WHY care about variance? Softmax is sensitive to scale!
 *      Large inputs → probabilities near 0 or 1 → vanishing gradients
 *    
 * 3. softmax: Convert scores to attention weights
 *    - Each row sums to 1 (probability distribution)
 *    - Query i's attention weights over all keys
 *    
 * 4. × V: Weighted combination of values
 *    - If attention[i][j] is high, query i pays attention to value j
 *    - Output row i is weighted average of value rows
 */
struct AttentionResult {
    Matrix2x2 scores;      // Q × K^T / √d_k
    Matrix2x2 weights;     // softmax(scores)
    Matrix2x2 output;      // weights × V
};

AttentionResult scaledDotProductAttention(
    const Matrix2x2& Q,
    const Matrix2x2& K, 
    const Matrix2x2& V,
    double scale
) {
    AttentionResult result;
    
    // Step 1: Q × K^T (similarity scores)
    Matrix2x2 Kt = K.transpose();
    Matrix2x2 rawScores = Q.matmul(Kt);
    
    // Step 2: Scale by 1/√d_k
    result.scores = rawScores.scale(scale);
    
    // Step 3: Softmax (row-wise to get attention distribution)
    result.weights = softmaxRows(result.scores);
    
    // Step 4: Weighted sum of values
    result.output = result.weights.matmul(V);
    
    return result;
}

/**
 * ============================================================================
 * BACKWARD PASS: Backpropagation Through Attention
 * ============================================================================
 * 
 * WHY backpropagation?
 * - To learn, we need gradients of loss w.r.t. parameters
 * - Chain rule: Break complex function into simple steps
 * - Each step has a known local gradient
 * - Multiply local gradients together for end-to-end gradient
 * 
 * For attention: Output = softmax(Q×K^T/√d) × V
 * 
 * Backward steps (reverse order of forward):
 * 1. dL/dWeights, dL/dV from output = Weights × V
 * 2. dL/dScores from scores → softmax → weights  
 * 3. dL/dQ, dL/dK from Q × K^T → raw_scores
 */
struct AttentionGradients {
    Matrix2x2 dQ;
    Matrix2x2 dK;
    Matrix2x2 dV;
};

AttentionGradients scaledDotProductAttentionBackward(
    const Matrix2x2& Q,
    const Matrix2x2& K,
    const Matrix2x2& V,
    const AttentionResult& fwd,
    const Matrix2x2& dOutput,  // Gradient of loss w.r.t. attention output
    double scale
) {
    AttentionGradients grads;
    
    /**
     * STEP 1: Gradient through output = weights × V
     * 
     * Using matrix calculus:
     * - dL/dWeights = dOutput × V^T
     * - dL/dV = Weights^T × dOutput
     * 
     * WHY these formulas?
     * For C = A × B:
     * - dL/dA = dL/dC × B^T  (gradient w.r.t. left matrix)
     * - dL/dB = A^T × dL/dC  (gradient w.r.t. right matrix)
     * This comes from the chain rule on matrix multiplication.
     */
    Matrix2x2 Vt = V.transpose();
    Matrix2x2 dWeights = dOutput.matmul(Vt);
    
    Matrix2x2 weightsT = fwd.weights.transpose();
    grads.dV = weightsT.matmul(dOutput);
    
    /**
     * STEP 2: Gradient through softmax
     * 
     * dL/dScores = softmax_backward(weights, dWeights)
     * 
     * WHY special handling? Softmax couples all inputs through the sum.
     */
    Matrix2x2 dScores = softmaxBackward(fwd.weights, dWeights);
    
    /**
     * STEP 3: Gradient through scaling
     * 
     * dL/dRawScores = dL/dScores × scale
     * 
     * WHY simple multiplication? 
     * Scaling is element-wise linear: d(ax)/dx = a
     */
    Matrix2x2 dRawScores = dScores.scale(scale);
    
    /**
     * STEP 4: Gradient through Q × K^T
     * 
     * - dL/dQ = dRawScores × K  (not K^T!)
     * - dL/dK = dRawScores^T × Q  then transpose, or equivalently: Q^T × dRawScores^T
     * 
     * WHY? For C = A × B^T:
     * - dL/dA = dL/dC × B
     * - dL/dB = dL/dC^T × A
     */
    grads.dQ = dRawScores.matmul(K);
    
    Matrix2x2 dRawScoresT = dRawScores.transpose();
    grads.dK = dRawScoresT.matmul(Q);
    
    return grads;
}

/**
 * ============================================================================
 * SIMPLE NEURAL NETWORK LAYER WITH ATTENTION
 * ============================================================================
 * 
 * A minimal transformer-like layer:
 * 1. Project input X to Q, K, V using weight matrices
 * 2. Apply attention
 * 3. Compute loss against target
 * 4. Backpropagate to update weights
 * 
 * WHY separate projections?
 * - Allows Q, K, V to live in different spaces
 * - More expressive: model can learn what to attend to (Q, K)
 *   separately from what information to extract (V)
 */
class SimpleAttentionLayer {
public:
    // Weight matrices for projecting to Q, K, V
    Matrix2x2 Wq, Wk, Wv;
    
    // Learning rate
    double lr;
    
    // Dimension for scaling (d_k = 2 for 2x2)
    double scale;
    
    SimpleAttentionLayer(double learning_rate = 0.1) : lr(learning_rate) {
        // Initialize with small values
        // WHY small? Large initial weights cause unstable gradients
        // WHY not zero? Symmetry breaking - neurons must be different
        Wq = Matrix2x2(0.1, 0.2, 0.3, 0.1);
        Wk = Matrix2x2(0.2, 0.1, 0.1, 0.3);
        Wv = Matrix2x2(0.3, 0.1, 0.2, 0.2);
        
        // Scale factor: 1/√d_k where d_k = 2
        scale = 1.0 / std::sqrt(2.0);
    }
    
    /**
     * Forward pass: compute attention output from input
     */
    struct ForwardResult {
        Matrix2x2 Q, K, V;
        AttentionResult attn;
    };
    
    ForwardResult forward(const Matrix2x2& X) {
        ForwardResult result;
        
        // Project to Q, K, V spaces
        // WHY linear projections?
        // - Simple but powerful: any linear transformation
        // - Learnable: weights adapt during training
        // - Efficient: matrix multiply is fast on hardware
        result.Q = X.matmul(Wq);
        result.K = X.matmul(Wk);
        result.V = X.matmul(Wv);
        
        // Apply attention mechanism
        result.attn = scaledDotProductAttention(result.Q, result.K, result.V, scale);
        
        return result;
    }
    
    /**
     * Compute Mean Squared Error loss
     * 
     * WHY MSE?
     * - Simple, differentiable
     * - Penalizes large errors quadratically
     * - Gradient is simply 2(prediction - target)
     * 
     * Loss = (1/n) × sum((pred - target)²)
     */
    double computeLoss(const Matrix2x2& prediction, const Matrix2x2& target) {
        Matrix2x2 diff = prediction.subtract(target);
        return diff.frobeniusNormSquared() / 4.0;  // /4 for mean over 2x2
    }
    
    /**
     * Backward pass: compute gradients w.r.t. weights
     * 
     * Chain rule breakdown:
     * Loss ← Output ← Attention ← (Q,K,V) ← Weights
     */
    void backward(
        const Matrix2x2& X,
        const ForwardResult& fwd,
        const Matrix2x2& target
    ) {
        // Zero all gradients first
        Wq.zeroGrad();
        Wk.zeroGrad();
        Wv.zeroGrad();
        
        /**
         * STEP 1: Gradient of MSE loss w.r.t. output
         * 
         * L = (1/4) × sum((out - target)²)
         * dL/dOut = (2/4) × (out - target) = 0.5 × (out - target)
         * 
         * WHY 0.5? Chain rule through the square and mean.
         */
        Matrix2x2 dOutput = fwd.attn.output.subtract(target).scale(0.5);
        
        /**
         * STEP 2: Backprop through attention
         * 
         * Get dL/dQ, dL/dK, dL/dV
         */
        AttentionGradients attnGrads = scaledDotProductAttentionBackward(
            fwd.Q, fwd.K, fwd.V, fwd.attn, dOutput, scale
        );
        
        /**
         * STEP 3: Backprop through projections to get weight gradients
         * 
         * Q = X × Wq
         * dL/dWq = X^T × dL/dQ
         * 
         * WHY X^T? For Y = X × W:
         * dL/dW = X^T × dL/dY
         * This aligns dimensions and correctly propagates gradients.
         */
        Matrix2x2 Xt = X.transpose();
        
        // Accumulate gradients
        Matrix2x2 dWq = Xt.matmul(attnGrads.dQ);
        Matrix2x2 dWk = Xt.matmul(attnGrads.dK);
        Matrix2x2 dWv = Xt.matmul(attnGrads.dV);
        
        // Store gradients
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                Wq.grad[i][j] = dWq.data[i][j];
                Wk.grad[i][j] = dWk.data[i][j];
                Wv.grad[i][j] = dWv.data[i][j];
            }
        }
    }
    
    /**
     * Update weights using gradient descent
     * 
     * W_new = W_old - lr × gradient
     * 
     * WHY subtract? We want to minimize loss.
     * Gradient points toward steepest increase.
     * Subtracting moves toward steepest decrease.
     */
    void updateWeights() {
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                Wq.data[i][j] -= lr * Wq.grad[i][j];
                Wk.data[i][j] -= lr * Wk.grad[i][j];
                Wv.data[i][j] -= lr * Wv.grad[i][j];
            }
        }
    }
};

/**
 * ============================================================================
 * MAIN: Demonstration with Hand-Calculable Example
 * ============================================================================
 */
int main() {
    std::cout << "================================================================\n";
    std::cout << "TRANSFORMER ATTENTION: MATRIX CORE OF GENERATIVE AI\n";
    std::cout << "================================================================\n\n";
    
    std::cout << "This example uses 2x2 matrices so you can verify every\n";
    std::cout << "calculation by hand on paper.\n\n";
    
    // =========================================================================
    // PART 1: FORWARD PASS (INFERENCE) - DETAILED STEP BY STEP
    // =========================================================================
    
    std::cout << "================================================================\n";
    std::cout << "PART 1: FORWARD PASS (INFERENCE)\n";
    std::cout << "================================================================\n\n";
    
    // Input: represents 2 tokens, each with 2-dimensional embedding
    // Think of row 0 as "hello" and row 1 as "world"
    Matrix2x2 X(1.0, 0.0,    // Token 1 embedding
                0.0, 1.0);   // Token 2 embedding
    
    std::cout << "INPUT: Two tokens with 2D embeddings\n";
    std::cout << "======================================\n";
    X.print("X (input)");
    std::cout << "\nWHY this input? Identity matrix = orthogonal tokens.\n";
    std::cout << "This makes hand calculation easier.\n\n";
    
    // Simple weight matrices for Q, K, V projections
    Matrix2x2 Wq(0.5, 0.0,
                 0.0, 0.5);
    Matrix2x2 Wk(1.0, 0.0,
                 0.0, 1.0);
    Matrix2x2 Wv(0.5, 0.5,
                 0.5, 0.5);
    
    std::cout << "PROJECTION WEIGHTS\n";
    std::cout << "==================\n";
    Wq.print("Wq (query weights)");
    Wk.print("Wk (key weights)");
    Wv.print("Wv (value weights)");
    std::cout << "\n";
    
    // Step 1: Project to Q, K, V
    std::cout << "STEP 1: Project Input to Q, K, V\n";
    std::cout << "================================\n\n";
    
    Matrix2x2 Q = X.matmul(Wq);
    Matrix2x2 K = X.matmul(Wk);
    Matrix2x2 V = X.matmul(Wv);
    
    std::cout << "Q = X × Wq:\n";
    std::cout << "[1 0]   [0.5 0  ]   [0.5 0  ]\n";
    std::cout << "[0 1] × [0   0.5] = [0   0.5]\n\n";
    Q.print("Q (queries)");
    
    std::cout << "\nWHY project to Q? The query represents 'what information\n";
    std::cout << "is this token looking for?' Learned projection lets model\n";
    std::cout << "decide what aspects of embedding are relevant for querying.\n\n";
    
    K.print("K (keys)");
    std::cout << "\nWHY project to K? Keys represent 'what information does\n";
    std::cout << "this token advertise?' Matching queries and keys determines\n";
    std::cout << "attention patterns.\n\n";
    
    V.print("V (values)");
    std::cout << "\nWHY project to V? Values contain 'the actual information\n";
    std::cout << "to retrieve.' Once attention weights are computed, values\n";
    std::cout << "are weighted and summed.\n\n";
    
    // Step 2: Compute attention scores
    std::cout << "STEP 2: Compute Attention Scores\n";
    std::cout << "================================\n\n";
    
    Matrix2x2 Kt = K.transpose();
    Kt.print("K^T (transposed)");
    
    Matrix2x2 scores = Q.matmul(Kt);
    std::cout << "\nQ × K^T (raw scores):\n";
    std::cout << "[0.5 0  ]   [1 0]   [0.5 0  ]\n";
    std::cout << "[0   0.5] × [0 1] = [0   0.5]\n\n";
    scores.print("Raw attention scores");
    
    std::cout << "\nWHY Q × K^T? This computes dot product between each query\n";
    std::cout << "and each key. Score[i][j] = how much should token i attend\n";
    std::cout << "to token j. Higher score = more similar query-key pair.\n\n";
    
    // Step 3: Scale
    std::cout << "STEP 3: Scale Scores\n";
    std::cout << "====================\n\n";
    
    double dk = 2.0;  // dimension of keys
    double scale_factor = 1.0 / std::sqrt(dk);
    std::cout << "Scale factor = 1/√d_k = 1/√2 ≈ " << scale_factor << "\n\n";
    
    Matrix2x2 scaledScores = scores.scale(scale_factor);
    scaledScores.print("Scaled scores");
    
    std::cout << "\nWHY scale? Without scaling, dot products grow with dimension.\n";
    std::cout << "Large values push softmax to extremes (near 0 or 1), causing\n";
    std::cout << "vanishing gradients. Scaling keeps variance stable.\n\n";
    
    // Step 4: Softmax
    std::cout << "STEP 4: Apply Softmax\n";
    std::cout << "=====================\n\n";
    
    Matrix2x2 weights = softmaxRows(scaledScores);
    
    std::cout << "For row 0: scores = [0.3536, 0]\n";
    std::cout << "  exp(0.3536) = " << std::exp(0.3536) << "\n";
    std::cout << "  exp(0) = 1\n";
    std::cout << "  sum = " << std::exp(0.3536) + 1.0 << "\n";
    std::cout << "  softmax = [" << std::exp(0.3536)/(std::exp(0.3536)+1.0) 
              << ", " << 1.0/(std::exp(0.3536)+1.0) << "]\n\n";
    
    weights.print("Attention weights (after softmax)");
    
    std::cout << "\nWHY softmax? Converts arbitrary scores to probability\n";
    std::cout << "distribution (positive, sums to 1). Each query now has\n";
    std::cout << "a probability distribution over keys. Note row sums = 1.\n\n";
    
    // Step 5: Weighted sum of values
    std::cout << "STEP 5: Compute Weighted Sum of Values\n";
    std::cout << "======================================\n\n";
    
    Matrix2x2 output = weights.matmul(V);
    
    std::cout << "Output = Attention_weights × V\n\n";
    output.print("Output");
    
    std::cout << "\nWHY this final step? Each output row is a weighted average\n";
    std::cout << "of value rows, where weights are attention probabilities.\n";
    std::cout << "Token 1 output combines information from all tokens based on\n";
    std::cout << "how much it 'attends' to each one.\n\n";
    
    // =========================================================================
    // PART 2: TRAINING (BACKPROPAGATION)
    // =========================================================================
    
    std::cout << "================================================================\n";
    std::cout << "PART 2: TRAINING (BACKPROPAGATION)\n";
    std::cout << "================================================================\n\n";
    
    // Create a trainable layer
    SimpleAttentionLayer layer(0.1);  // learning rate = 0.1
    
    // Define target output (what we want the network to produce)
    Matrix2x2 target(0.5, 0.5,
                     0.5, 0.5);
    
    std::cout << "Training to produce target output:\n";
    target.print("Target");
    std::cout << "\n";
    
    std::cout << "Initial weights:\n";
    layer.Wq.print("Wq");
    layer.Wk.print("Wk");
    layer.Wv.print("Wv");
    std::cout << "\n";
    
    // Training loop
    std::cout << "TRAINING LOOP (10 iterations)\n";
    std::cout << "============================\n\n";
    
    for (int epoch = 0; epoch < 10; epoch++) {
        // Forward pass
        auto fwd = layer.forward(X);
        
        // Compute loss
        double loss = layer.computeLoss(fwd.attn.output, target);
        
        // Backward pass
        layer.backward(X, fwd, target);
        
        // Update weights
        layer.updateWeights();
        
        if (epoch % 2 == 0) {
            std::cout << "Epoch " << epoch << ": Loss = " << std::fixed 
                      << std::setprecision(6) << loss << "\n";
        }
    }
    
    std::cout << "\nFinal weights after training:\n";
    layer.Wq.print("Wq");
    layer.Wk.print("Wk");
    layer.Wv.print("Wv");
    
    // Final inference
    auto finalFwd = layer.forward(X);
    double finalLoss = layer.computeLoss(finalFwd.attn.output, target);
    
    std::cout << "\nFinal output:\n";
    finalFwd.attn.output.print("Output");
    std::cout << "\nFinal loss: " << finalLoss << "\n\n";
    
    // =========================================================================
    // PART 3: DETAILED BACKPROPAGATION TRACE
    // =========================================================================
    
    std::cout << "================================================================\n";
    std::cout << "PART 3: DETAILED BACKPROPAGATION TRACE\n";
    std::cout << "================================================================\n\n";
    
    std::cout << "Let's trace one backward pass step by step:\n\n";
    
    // Use values that produce non-zero gradients for demonstration
    Matrix2x2 X2(1.0, 0.5, 0.5, 1.0);  // Non-identity input
    SimpleAttentionLayer layer2(0.1);
    layer2.Wq = Matrix2x2(0.3, 0.1, 0.2, 0.4);  // Asymmetric weights
    layer2.Wk = Matrix2x2(0.2, 0.3, 0.1, 0.2);
    layer2.Wv = Matrix2x2(0.4, 0.2, 0.3, 0.1);
    
    auto fwd2 = layer2.forward(X2);
    
    std::cout << "Forward pass results:\n";
    fwd2.Q.print("Q");
    fwd2.K.print("K");
    fwd2.V.print("V");
    fwd2.attn.scores.print("Scores (scaled)");
    fwd2.attn.weights.print("Attention weights");
    fwd2.attn.output.print("Output");
    std::cout << "\n";
    
    // Compute gradients with a different target to get non-zero gradients
    Matrix2x2 target2(0.8, 0.3, 0.4, 0.7);  // Target that differs from output
    double loss2 = layer2.computeLoss(fwd2.attn.output, target2);
    std::cout << "Loss = " << loss2 << "\n\n";
    
    std::cout << "BACKPROPAGATION STEPS:\n";
    std::cout << "=====================\n\n";
    
    std::cout << "Step 1: dL/dOutput = 0.5 × (Output - Target)\n";
    Matrix2x2 dOut = fwd2.attn.output.subtract(target2).scale(0.5);
    dOut.print("dL/dOutput");
    std::cout << "\nWHY? MSE gradient is 2×(pred-target), and /4 for mean gives 0.5×.\n\n";
    
    std::cout << "Step 2: dL/dWeights and dL/dV from output = weights × V\n";
    Matrix2x2 Vt2 = fwd2.V.transpose();
    Matrix2x2 dWeights = dOut.matmul(Vt2);
    Matrix2x2 weightsT2 = fwd2.attn.weights.transpose();
    Matrix2x2 dV = weightsT2.matmul(dOut);
    dWeights.print("dL/dWeights");
    dV.print("dL/dV");
    std::cout << "\nWHY? Matrix calc: dL/dA = dL/dC × B^T for C = A × B\n\n";
    
    std::cout << "Step 3: dL/dScores from softmax backward\n";
    Matrix2x2 dScores = softmaxBackward(fwd2.attn.weights, dWeights);
    dScores.print("dL/dScores");
    std::cout << "\nWHY? Softmax has complex Jacobian: dL/dx = s × (dL/ds - dot(dL/ds, s))\n\n";
    
    std::cout << "Step 4: dL/dQ and dL/dK from Q × K^T\n";
    Matrix2x2 dQ = dScores.scale(layer2.scale).matmul(fwd2.K);
    Matrix2x2 dK = dScores.scale(layer2.scale).transpose().matmul(fwd2.Q);
    dQ.print("dL/dQ");
    dK.print("dL/dK");
    std::cout << "\nWHY? For C = A × B^T: dL/dA = dL/dC × B, dL/dB = dL/dC^T × A\n\n";
    
    std::cout << "Step 5: dL/dWq, dL/dWk, dL/dWv from projections\n";
    Matrix2x2 Xt2 = X2.transpose();
    Matrix2x2 dWq = Xt2.matmul(dQ);
    Matrix2x2 dWk = Xt2.matmul(dK);
    Matrix2x2 dWv = Xt2.matmul(dV);
    dWq.print("dL/dWq");
    dWk.print("dL/dWk");
    dWv.print("dL/dWv");
    std::cout << "\nWHY? For Y = X × W: dL/dW = X^T × dL/dY\n\n";
    
    // =========================================================================
    // SUMMARY
    // =========================================================================
    
    std::cout << "================================================================\n";
    std::cout << "SUMMARY: THE MATRIX CORE OF GENERATIVE AI\n";
    std::cout << "================================================================\n\n";
    
    std::cout << "1. MATRIX MULTIPLY is the core operation:\n";
    std::cout << "   - Projects inputs to Q, K, V (learning what to attend to)\n";
    std::cout << "   - Computes Q×K^T (similarity/attention scores)\n";
    std::cout << "   - Computes weights×V (aggregate information)\n\n";
    
    std::cout << "2. SOFTMAX converts scores to probabilities:\n";
    std::cout << "   - Makes attention weights interpretable\n";
    std::cout << "   - Ensures weighted sum is a proper average\n\n";
    
    std::cout << "3. SCALING by 1/√d_k prevents gradient issues:\n";
    std::cout << "   - Keeps variance stable across different dimensions\n";
    std::cout << "   - Prevents softmax saturation\n\n";
    
    std::cout << "4. BACKPROPAGATION uses chain rule:\n";
    std::cout << "   - Gradients flow backward through operations\n";
    std::cout << "   - Matrix calculus rules for each step\n";
    std::cout << "   - Weights updated to minimize loss\n\n";
    
    std::cout << "5. WHY THIS WORKS for language:\n";
    std::cout << "   - Attention allows each token to gather relevant context\n";
    std::cout << "   - Learned projections adapt to task\n";
    std::cout << "   - Parallel computation (no sequential bottleneck)\n\n";
    
    std::cout << "================================================================\n";
    std::cout << "All values above are hand-calculable with 2x2 matrices!\n";
    std::cout << "================================================================\n";
    
    return 0;
}

/**
 * ============================================================================
 * COMPILATION AND RUNNING
 * ============================================================================
 * 
 * To compile (from repository root):
 *   g++ -std=c++11 -O2 transformer_2x2.cpp -o transformer_2x2
 * 
 * To run:
 *   ./transformer_2x2
 * 
 * ============================================================================
 * FURTHER READING
 * ============================================================================
 * 
 * 1. "Attention Is All You Need" (Vaswani et al., 2017)
 *    - Original transformer paper
 * 
 * 2. Multi-head attention: Run multiple attention heads in parallel,
 *    each with different Q, K, V projections, then concatenate outputs.
 * 
 * 3. In real transformers:
 *    - Larger matrices (d_model = 768 for GPT-2 small)
 *    - Layer normalization for training stability
 *    - Feedforward networks after attention
 *    - Positional encodings (attention is permutation invariant)
 *    - Causal masking for autoregressive generation
 * 
 * ============================================================================
 */
