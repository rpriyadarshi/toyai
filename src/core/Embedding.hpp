#ifndef EMBEDDING_HPP
#define EMBEDDING_HPP

#include "Matrix.hpp"
#include <vector>

/**
 * ============================================================================
 * Embedding Layer: Token to Vector Conversion
 * ============================================================================
 * 
 * Converts discrete token indices to continuous vector representations.
 * 
 * For our 2x2 case:
 * - Vocabulary size: 4 (tokens A, B, C, D)
 * - Embedding dimension: 2
 * - Each token maps to a 2D vector
 */
class Embedding {
public:
    // Embedding matrix: vocab_size × embed_dim
    // For 4 tokens × 2 dimensions
    double embeddings[4][2];
    
    // Whether embeddings are learnable
    bool learnable;
    
    // Gradient storage (if learnable)
    double grad[4][2];
    
    /**
     * Constructor with fixed embeddings
     */
    Embedding();
    
    /**
     * Constructor with custom embeddings
     */
    Embedding(double emb[4][2]);
    
    /**
     * Forward pass: token index → embedding vector
     * Returns a 2D vector (as Matrix for consistency)
     */
    Matrix forward(int token_idx) const;
    
    /**
     * Forward pass for sequence: [token1, token2] → [vec1, vec2]
     * Returns 2x2 matrix where each row is an embedding
     */
    Matrix forwardSequence(const std::vector<int>& tokens) const;
    
    /**
     * Get embedding for token (returns as array for easy access)
     */
    void getEmbedding(int token_idx, double out[2]) const;
    
    /**
     * Set embedding for token
     */
    void setEmbedding(int token_idx, double emb[2]);
    
    /**
     * Zero gradients
     */
    void zeroGrad();
    
    /**
     * Print embeddings
     */
    void print() const;
    
    /**
     * Default embeddings for A, B, C, D
     */
    static void getDefaultEmbeddings(double emb[4][2]);
};

#endif // EMBEDDING_HPP

