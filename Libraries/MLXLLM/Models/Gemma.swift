// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Tokenizers

// Port of https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/gemma.py

private class Attention: Module {
    let args: GemmaConfiguration
    let nHeads: Int      // Number of attention heads (parallel attention computations)
    let nKVHeads: Int    // Number of key-value heads (can be fewer than query heads for efficiency)
    let headDim: Int     // Dimension of each attention head
    let scale: Float     // Scaling factor for attention scores (1/sqrt(head_dim))

    // Linear projections to create queries, keys, values, and output
    @ModuleInfo(key: "q_proj") var wq: Linear  // Projects input to queries
    @ModuleInfo(key: "k_proj") var wk: Linear  // Projects input to keys  
    @ModuleInfo(key: "v_proj") var wv: Linear  // Projects input to values
    @ModuleInfo(key: "o_proj") var wo: Linear  // Projects attention output back to hidden size

    let rope: RoPE  // Rotary Position Encoding - adds position information to queries/keys

    public init(_ args: GemmaConfiguration) {
        self.args = args

        let dim = args.hiddenSize  // Main hidden dimension of the model
        self.nHeads = args.attentionHeads      // e.g., 32 heads
        self.nKVHeads = args.kvHeads           // e.g., 8 heads (grouped query attention)
        self.headDim = args.headDimensions     // e.g., 128 dimensions per head
        
        // Scale factor to prevent attention scores from getting too large
        // Larger head dimensions would produce larger dot products, so we scale down
        self.scale = pow(Float(headDim), -0.5)

        // Create linear layers for Q, K, V projections
        // Note: bias=false is common in modern transformers
        self._wq.wrappedValue = Linear(dim, nHeads * headDim, bias: false)
        self._wk.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        self._wv.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        self._wo.wrappedValue = Linear(nHeads * headDim, dim, bias: false)

        // RoPE adds positional information by rotating query/key vectors
        self.rope = RoPE(
            dimensions: headDim, 
            traditional: args.ropeTraditional, 
            base: args.ropeTheta
        )
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        // x shape: [batch_size, sequence_length, hidden_size]
        // 
        // Input tensor breakdown:
        // - B (Batch size): Number of sequences being processed in parallel (e.g., 4)
        // - L (Sequence length): Number of tokens per sequence (e.g., 100) 
        // - hidden_size: Features per token - the model's internal representation size (e.g., 3072)
        //
        // Example: [4, 100, 3072] = 4 sequences × 100 tokens × 3072 features per token
        let (B, L) = (x.dim(0), x.dim(1))  // Extract batch size and sequence length

        // Project input to queries, keys, and values
        // Each projection transforms [B, L, hidden_size] → [B, L, n_heads * head_dim]
        var queries = wq(x)  // "What am I looking for?"
        var keys = wk(x)     // "What information do I contain?"
        var values = wv(x)   // "What information should I pass forward?"

        // Reshape for multi-head attention: split the head dimension and move heads to dim 1
        // [B, L, n_heads * head_dim] → [B, n_heads, L, head_dim]
        // 
        // Why this reshape? Multi-head attention runs multiple "attention heads" in parallel.
        // Each head looks for different types of relationships (syntax, semantics, etc.)
        // 
        // Before: [B, L, n_heads * head_dim] - all head info mixed together
        // After:  [B, n_heads, L, head_dim] - each head gets its own slice
        // 
        // Example: [4, 100, 3072] → [4, 24, 100, 128] 
        // (24 heads × 128 dims = 3072 total, but now each head is separate)
        // This allows us to compute all heads in parallel efficiently
        queries = queries.reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

        // Apply rotary position encoding to add position information
        if let cache {
            // During generation, we're adding one token at a time
            // offset tells us where we are in the sequence
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
        } else {
            // During training/first pass, we process the full sequence
            queries = rope(queries)
            keys = rope(keys)
        }

        // Core attention computation: 
        // 1. Compute attention scores: queries @ keys.T
        // 2. Scale by 1/sqrt(head_dim) 
        // 3. Apply mask (prevents looking at future tokens)
        // 4. Softmax to get attention weights
        // 5. Weighted sum of values
        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,      // For efficient generation (stores past keys/values)
            scale: scale,      // Prevents attention scores from being too large
            mask: mask         // Causal mask: can't attend to future positions
        )
        .transposed(0, 2, 1, 3)  // Move heads back: [B, n_heads, L, head_dim] → [B, L, n_heads, head_dim]
        .reshaped(B, L, -1)      // Concatenate heads: [B, L, n_heads, head_dim] → [B, L, n_heads * head_dim]

        // Final projection back to hidden dimension
        return wo(output)
    }
}

private class MLP: Module, UnaryLayer {
    // Multi-Layer Perceptron: the "thinking" component of each transformer layer
    @ModuleInfo(key: "gate_proj") var gate: Linear  // Gating mechanism - decides what info to keep
    @ModuleInfo(key: "down_proj") var down: Linear  // Projects back down to hidden size
    @ModuleInfo(key: "up_proj") var up: Linear      // Projects up to larger intermediate size

    public init(dimensions: Int, hiddenDimensions: Int) {
        // dimensions = hidden_size (e.g., 3072)
        // hiddenDimensions = intermediate_size (e.g., 8192) - larger for more "thinking space"
        self._gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        self._up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // SwiGLU activation: combines gating with GLU (Gated Linear Unit)
        // 1. gate(x): decides what information is important
        // 2. up(x): transforms input to higher dimension for complex reasoning
        // 3. gelu: adds non-linearity (smooth version of ReLU)
        // 4. element-wise multiply: gate controls what passes through
        // 5. down: project back to original dimension
        return down(gelu(gate(x)) * up(x))
    }
}

private class TransformerBlock: Module {
    // One layer of the transformer: the fundamental building block
    @ModuleInfo(key: "self_attn") var attention: Attention  // Self-attention mechanism
    let mlp: MLP  // Feed-forward network for reasoning

    // Layer normalization before each sub-layer (Pre-LN architecture)
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Gemma.RMSNorm

    public init(_ args: GemmaConfiguration) {
        self._attention.wrappedValue = Attention(args)
        self.mlp = MLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        
        // Same normalization parameters for both norm layers
        self._inputLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        // Standard transformer layer computation with residual connections:
        
        // 1. Self-attention sub-layer with residual connection
        var r = attention(inputLayerNorm(x), mask: mask, cache: cache)  // Normalize then attend
        let h = x + r  // Residual connection: add input to attention output
        
        // 2. Feed-forward sub-layer with residual connection  
        r = mlp(postAttentionLayerNorm(h))  // Normalize then apply MLP
        return h + r  // Another residual connection
        
        // Residual connections are crucial: they create "highways" for gradients
        // and prevent information loss in deep networks
    }
}

private class GemmaModelInner: Module {
    // The core model without the final language modeling head
    let args: GemmaConfiguration
    let vocabularySize: Int     // Number of tokens in vocabulary (e.g., 256,000)
    let numHiddenLayers: Int    // Number of transformer layers (e.g., 28)

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding  // Converts token IDs to vectors
    fileprivate let layers: [TransformerBlock]  // Stack of transformer layers
    fileprivate let norm: Gemma.RMSNorm         // Final normalization layer

    public init(_ args: GemmaConfiguration) {
        precondition(args.vocabularySize > 0)

        self.args = args
        self.vocabularySize = args.vocabularySize
        self.numHiddenLayers = args.hiddenLayers

        // Embedding layer: maps token IDs (integers) to dense vectors
        // Each of vocabularySize tokens gets a learnable vector of hiddenSize dimensions
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        // Create a stack of identical transformer layers
        // Each layer will learn different patterns and abstractions
        self.layers = (0 ..< args.hiddenLayers)
            .map { _ in
                TransformerBlock(args)
            }
        
        // Final normalization before output
        self.norm = Gemma.RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        // inputs: [batch_size, sequence_length] - token IDs
        // 
        // Input shape: [B, L] where:
        // - B (Batch size): How many text sequences we're processing simultaneously
        // - L (Sequence length): How many tokens (words/subwords) in each sequence
        // 
        // Each element is an integer token ID (e.g., token 5234 might represent "hello")
        // Example: [4, 100] means 4 sentences, each with 100 tokens
        
        // 1. Convert token IDs to embeddings: [B, L] → [B, L, hidden_size]
        // This transforms sparse integer IDs into dense vector representations
        // Each token ID gets mapped to a learned vector of hidden_size dimensions
        var h = embedTokens(inputs)
        
        // 2. Scale embeddings by sqrt(hidden_size) - common in transformer architectures
        // This helps with training stability when hidden_size is large
        h = h * pow(Float(args.hiddenSize), 0.5)

        // 3. Create attention mask to prevent attending to future tokens (causal masking)
        let mask = createAttentionMask(h: h, cache: cache)

        // 4. Pass through all transformer layers sequentially
        // Each layer refines the representation, building up understanding
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        // 5. Final normalization - ensures stable outputs
        return norm(h)
    }
}

public class GemmaModel: Module, LLMModel, KVCacheDimensionProvider {
    // The complete model including the language modeling head
    public let vocabularySize: Int
    public let kvHeads: [Int]  // Number of KV heads for each layer (for cache allocation)

    let modelType: String
    private let model: GemmaModelInner  // The core transformer

    public init(_ args: GemmaConfiguration) {
        self.modelType = args.modelType
        self.vocabularySize = args.vocabularySize
        // All layers have the same number of KV heads in Gemma
        self.kvHeads = Array(repeating: args.kvHeads, count: args.hiddenLayers)
        self.model = GemmaModelInner(args)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        // 1. Get hidden states from the transformer: [B, L, hidden_size]
        // 
        // Tensor shape breakdown:
        // - B (Batch size): How many sequences we're processing at once (e.g., 4 sentences)
        // - L (Sequence length): How many tokens in each sequence (e.g., 100 words per sentence)  
        // - hidden_size: The model's internal representation size (e.g., 3072 features per token)
        //
        // Example: [4, 100, 3072] means we're processing 4 sentences of 100 tokens each,
        // where each token is represented as a 3072-dimensional vector containing the
        // model's learned understanding of that token in context
        let out = model(inputs, cache: cache)
        
        // 2. Language modeling head: convert hidden states to vocabulary logits
        // We reuse the embedding matrix as the output projection (weight tying)
        // This reduces parameters and often improves performance
        return model.embedTokens.asLinear(out)  // [B, L, hidden_size] → [B, L, vocab_size]
        
        // Final output shape: [B, L, vocab_size]
        // - B: Same batch size as input
        // - L: Same sequence length as input  
        // - vocab_size: Probability distribution over all possible next tokens (e.g., 256,000 tokens)
        //
        // Example: [4, 100, 256000] means for each of the 4 sentences, for each of the 100 positions,
        // we have a 256,000-dimensional vector where each element is the model's confidence score
        // for that specific token being the next word. Higher scores = more likely next token.
        //
        // The output logits represent the model's prediction for the next token
        // Higher logits = higher probability for that token
    }

    public func messageGenerator(tokenizer: any Tokenizer) -> any MessageGenerator {
        // Gemma doesn't use system messages, so we use a no-op generator
        NoSystemMessageGenerator()
    }
}

public struct GemmaConfiguration: Codable, Sendable {
    // Model architecture hyperparameters
    var modelType: String           // e.g., "gemma"
    var hiddenSize: Int            // Main hidden dimension (e.g., 3072)
                                   // ⚠️  KEY DESIGN PRINCIPLE: This is the size of ALL hidden representations:
                                   //    • Token embeddings: [vocab_size] → [hidden_size]
                                   //    • Attention outputs: [B, L, hidden_size] 
                                   //    • MLP outputs: [B, L, hidden_size]
                                   //    • Layer outputs: [B, L, hidden_size]
                                   //    This uniform sizing enables residual connections (x + f(x))
                                   //    and seamless layer stacking throughout the transformer
    var hiddenLayers: Int          // Number of transformer layers (e.g., 28)
                                   // Each layer refines the representation, building understanding progressively:
                                   // • Early layers: basic patterns, syntax, word relationships
                                   // • Middle layers: semantics, context, longer dependencies  
                                   // • Late layers: complex reasoning, task-specific knowledge
                                   // More layers = more sophisticated understanding, but also more computation
                                   // Typical range: 6-12 for smaller models, 24-96+ for large models
    
    var intermediateSize: Int      // MLP intermediate size (e.g., 8192) - usually 4x hidden_size
                                   // The MLP expands hidden_size → intermediateSize → hidden_size
                                   // Why expand? Creates a "thinking space" for complex computations:
                                   // • Input: [B, L, 3072] → Gate/Up projections: [B, L, 8192] 
                                   // • More neurons = more computational capacity for reasoning
                                   // • 4x rule: empirically found to work well (balance of power vs efficiency)
                                   // • After gated activation, Down projection: [B, L, 8192] → [B, L, 3072]
    
    var attentionHeads: Int        // Number of attention heads (e.g., 24)
                                   // Multi-head attention runs multiple "attention functions" in parallel
                                   // Why multiple heads? Each head can focus on different relationships:
                                   // • Head 1: syntactic dependencies ("the cat" → noun-determiner)
                                   // • Head 2: semantic similarity ("king" ↔ "queen") 
                                   // • Head 3: long-range context (pronoun → antecedent across sentences)
                                   // • Head 4: positional patterns (sequence ordering)
                                   // Total attention dimension = attentionHeads × headDimensions = hiddenSize
    
    var headDimensions: Int        // Dimension per attention head (e.g., 128)
                                   // How we split the attention computation: hiddenSize = heads × headDim
                                   // Example: 3072 = 24 heads × 128 dimensions per head
                                   // Each head gets its own slice of the representation to work with
                                   // Smaller headDim = more heads = more diverse attention patterns
                                   // Larger headDim = fewer heads = more complex patterns per head
                                   // Common values: 64, 80, 96, 128 (powers of 2 for efficient computation)
    
    var rmsNormEps: Float         // Epsilon for numerical stability in normalization (e.g., 1e-5)
                                   // Prevents division by zero in RMSNorm: x / sqrt(mean(x²) + eps)
                                   // Without eps: if all values in x are exactly 0, we'd divide by 0 → NaN
                                   // Too large eps: normalization becomes less effective
                                   // Too small eps: risk of numerical instability in float16/float32
                                   // 1e-5 is empirically good balance for most cases
    
    var vocabularySize: Int        // Size of token vocabulary (e.g., 256,000)
                                   // How many unique "tokens" (words/subwords) the model knows
                                   // Text preprocessing: "Hello world!" → [23421, 1085, 77] (token IDs)
                                   // Larger vocab = more nuanced language understanding, but:
                                   // • More parameters in embedding layer (vocab_size × hidden_size)
                                   // • More computation in final language modeling head
                                   // • Diminishing returns beyond ~100K for most languages
                                   // Gemma uses SentencePiece tokenization with ~256K tokens
    
    var kvHeads: Int              // Number of key-value heads (often < attention heads for efficiency)
                                   // Modern optimization: "Grouped Query Attention" (GQA)
                                   // Instead of: 24 query heads + 24 key heads + 24 value heads
                                   // We use: 24 query heads + 8 key heads + 8 value heads (example)
                                   // Each K/V head is shared across multiple Q heads (24÷8=3 queries per K/V)
                                   // Benefits: ~3x less memory for KV cache, faster inference
                                   // Trade-off: slight quality reduction vs major efficiency gain
                                   // When kvHeads = attentionHeads: standard multi-head attention
    
    // RoPE (Rotary Position Encoding) parameters
    private let _ropeTheta: Float?      // Base frequency for position encoding
    public var ropeTheta: Float { _ropeTheta ?? 10_000 }
    private let _ropeTraditional: Bool? // Whether to use traditional RoPE implementation
    public var ropeTraditional: Bool { _ropeTraditional ?? false }

    // JSON field mappings from HuggingFace config files
    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case headDimensions = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case _ropeTheta = "rope_theta"
        case _ropeTraditional = "rope_traditional"
    }
}

// MARK: - LoRA

extension GemmaModel: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        // For LoRA fine-tuning, we typically adapt the query and value projections
        // This allows efficient fine-tuning by only updating a small number of parameters
        model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }
}
