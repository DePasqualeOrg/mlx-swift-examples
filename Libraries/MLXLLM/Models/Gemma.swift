// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Tokenizers

// Port of https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/gemma.py

// Specialized norm for Gemma
private class RMSNorm: Module, UnaryLayer {
    let weight: MLXArray  // Learnable scaling parameters for each feature dimension
    let eps: Float       // Small constant to prevent division by zero

    public init(dimensions: Int, eps: Float = 1e-5) {
        // Initialize weights as ones - these will be learned during training
        self.weight = MLXArray.ones([dimensions])
        self.eps = eps
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // RMSNorm: normalizes based on root mean square, simpler than LayerNorm
        // Formula: x / sqrt(mean(x²) + eps) * (1 + weight)
        // This keeps activations in a stable range as they flow through layers
        return MLXFast.rmsNorm(x, weight: 1.0 + self.weight, eps: self.eps)
    }
}

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
        let (B, L) = (x.dim(0), x.dim(1))  // B=batch size, L=sequence length

        // Project input to queries, keys, and values
        // Each projection transforms [B, L, hidden_size] → [B, L, n_heads * head_dim]
        var queries = wq(x)  // "What am I looking for?"
        var keys = wk(x)     // "What information do I contain?"
        var values = wv(x)   // "What information should I pass forward?"

        // Reshape for multi-head attention: split the head dimension and move heads to dim 1
        // [B, L, n_heads * head_dim] → [B, n_heads, L, head_dim]
        // This allows us to compute all heads in parallel
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
        
        // 1. Convert token IDs to embeddings: [B, L] → [B, L, hidden_size]
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
        let out = model(inputs, cache: cache)
        
        // 2. Language modeling head: convert hidden states to vocabulary logits
        // We reuse the embedding matrix as the output projection (weight tying)
        // This reduces parameters and often improves performance
        return model.embedTokens.asLinear(out)  // [B, L, hidden_size] → [B, L, vocab_size]
        
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
    var hiddenLayers: Int          // Number of transformer layers (e.g., 28) 
    var intermediateSize: Int      // MLP intermediate size (e.g., 8192) - usually 4x hidden_size
    var attentionHeads: Int        // Number of attention heads (e.g., 24)
    var headDimensions: Int        // Dimension per attention head (e.g., 128)
    var rmsNormEps: Float         // Epsilon for numerical stability in normalization
    var vocabularySize: Int        // Size of token vocabulary (e.g., 256,000)
    var kvHeads: Int              // Number of key-value heads (often < attention heads for efficiency)
    
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
