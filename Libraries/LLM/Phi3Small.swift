import Foundation
import MLX
import MLXNN
import MLXFast

// MARK: - Model Arguments for Phi 3 Small
struct ModelArgs {
  let modelType: String
  let hiddenSize: Int
  let vocabSize: Int
  let numHiddenLayers: Int
  let numAttentionHeads: Int
  let numKeyValueHeads: Int?
  let denseAttentionEveryNLayers: Int
  let ffIntermediateSize: Int
  let gegeluLimit: Float
  let layerNormEpsilon: Float
  let mupAttnMultiplier: Float
  let mupUseScaling: Bool
  let mupEmbeddingMultiplier: Float
  let mupWidthMultiplier: Float
  let ropeEmbeddingBase: Float
  let ropePositionScale: Float
  let blocksparseBlockSize: [Int]
  let blocksparseNumLocalBlocks: Int
  let blocksparseVertStride: Int
}

// MARK: - GeGLU Activation Function
func gegelu(_ x: MLXArray, limit: Float) -> MLXArray {
  let aGelu, aLinear = x.split(at: [x.shape[1] / 2], axis: -1)
  let outGelu = aGelu * MLX.tanh(aGelu * 1.702)
  return MLX.clip(outGelu * (aLinear + 1.0), min: -limit, max: limit)
}

// MARK: - Attention Module
class Attention: Module {
  let args: ModelArgs
  let queryKeyValue: Linear
  let dense: Linear
  let scale: Float
  let rope: RoPE

  init(_ args: ModelArgs, layerIndex: Int) {
    self.args = args
    self.queryKeyValue = Linear(args.hiddenSize, (args.numAttentionHeads + 2 * args.numKeyValueHeads!) * (args.hiddenSize / args.numAttentionHeads))
    self.dense = Linear(args.hiddenSize, args.hiddenSize)
    self.scale = args.mupUseScaling ? (Float(args.hiddenSize) / args.mupAttnMultiplier).squareRoot() : 1.0
    self.rope = RoPE(dimensions: args.hiddenSize / args.numAttentionHeads, base: args.ropeEmbeddingBase, scale: args.ropePositionScale)
  }

  func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, cache: MLXArray? = nil) -> MLXArray {
    let qkv = queryKeyValue(x)
    let (queries, keys, values) = qkv.split(at: [args.numAttentionHeads, 2 * args.numAttentionHeads], axis: -1)

    if cache != nil {
      queries = rope(queries, offset: cache!.shape[2])
      keys = rope(keys, offset: cache!.shape[2])
      keys = concatenated([cache!, keys], axis: 2)
      values = concatenated([cache!, values], axis: 2)
    } else {
      queries = rope(queries)
      keys = rope(keys)
    }

    let output = MLXFast.scaledDotProductAttention(queries: queries, keys: keys, values: values, scale: scale, mask: mask)
    output = output.transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], -1)
    return dense(output)
  }
}

// MARK: - MLP Module
class MLP: Module {
  let upProj: Linear
  let downProj: Linear
  let gegeluLimit: Float

  init(dimensions: Int, hiddenDimensions: Int, limit: Float) {
    self.upProj = Linear(dimensions, 2 * hiddenDimensions)
    self.downProj = Linear(hiddenDimensions, dimensions)
    self.gegeluLimit = limit
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    let up = upProj(x)
    return downProj(gegelu(up, limit: gegeluLimit))
  }
}

// MARK: - Transformer Block
class TransformerBlock: Module {
  let attention: Attention
  let mlp: MLP
  let inputLayerNorm: LayerNorm
  let postAttentionLayerNorm: LayerNorm

  init(_ args: ModelArgs, layerIndex: Int) {
    self.attention = Attention(args, layerIndex: layerIndex)
    self.mlp = MLP(dimensions: args.hiddenSize, hiddenDimensions: args.ffIntermediateSize, limit: args.gegeluLimit)
    self.inputLayerNorm = LayerNorm(dimensions: args.hiddenSize, eps: args.layerNormEpsilon)
    self.postAttentionLayerNorm = LayerNorm(dimensions: args.hiddenSize, eps: args.layerNormEpsilon)
  }

  func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, cache: MLXArray? = nil) -> MLXArray {
    var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
    let h = x + r
    r = mlp(postAttentionLayerNorm(h))
    return h + r
  }
}

// MARK: - Phi 3 Small Model
class Phi3SmallModel: Module {
  let args: ModelArgs
  let embedTokens: Embedding
  let layers: [TransformerBlock]
  let finalLayerNorm: LayerNorm

  init(_ args: ModelArgs) {
    self.args = args
    self.embedTokens = Embedding(embeddingCount: args.vocabSize, dimensions: args.hiddenSize)
    self.layers = (0..<args.numHiddenLayers).map { TransformerBlock(args, layerIndex: $0) }
    self.finalLayerNorm = LayerNorm(dimensions: args.hiddenSize, eps: args.layerNormEpsilon)
  }

  func callAsFunction(_ inputs: MLXArray, cache: [MLXArray]? = nil) -> MLXArray {
    var h = embedTokens(inputs) * args.mupEmbeddingMultiplier
    let mask: MLXArray? = nil  // Add mask if required

    var newCache = [MLXArray]()
    for (i, layer) in layers.enumerated() {
      let cacheUpdate = cache?[i]
      h = layer(h, mask: mask, cache: cacheUpdate)
      newCache.append(cacheUpdate)
    }
    return finalLayerNorm(h)
  }
}
