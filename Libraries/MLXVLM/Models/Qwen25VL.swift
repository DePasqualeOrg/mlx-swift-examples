// Port of https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/qwen2_5_vl

import CoreImage
import Foundation
import Hub
import MLX
import MLXLMCommon
import MLXNN
import Tokenizers

// MARK: - Language

private enum Language {

    /// Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors
    static private func applyMultimodalRotaryPositionEmbedding(
        q: MLXArray, k: MLXArray, cos: MLXArray, sin: MLXArray,
        positionIds: MLXArray, mropeSection: [Int]
    ) -> (MLXArray, MLXArray) {
        var cos = cos[positionIds]
        var sin = sin[positionIds]

        cos =
            concatenated(
                // [m[i % 3] for i, m in enumerate(mx.split(cos, mrope_section, axis=-1))]
                split(cos, indices: mropeSection, axis: -1).enumerated().map { i, m in m[i % 3] },
                axis: -1
            )[0..., .newAxis, 0..., 0...]

        sin =
            concatenated(
                split(sin, indices: mropeSection, axis: -1).enumerated().map { i, m in m[i % 3] },
                axis: -1
            )[0..., .newAxis, 0..., 0...]

        // Apply rotary embedding
        let qEmbed = (q * cos) + (QwenVL.rotateHalf(q) * sin)
        let kEmbed = (k * cos) + (QwenVL.rotateHalf(k) * sin)
        return (qEmbed, kEmbed)
    }

    fileprivate class Attention: Module {

        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float
        let mropeSection: [Int]

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "o_proj") var wo: Linear

        @ModuleInfo(key: "rotary_emb") var rotaryEmbedding: QwenVL.Qwen2RotaryEmbedding

        public init(_ args: Qwen25VLConfiguration.TextConfiguration) {
            let dim = args.hiddenSize
            self.heads = args.attentionHeads
            self.kvHeads = args.kvHeads
            self.headDim = dim / heads
            self.scale = pow(Float(headDim), -0.5)

            self._wq.wrappedValue = Linear(dim, heads * headDim, bias: true)
            self._wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
            self._wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
            self._wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

            if let v = args.ropeScaling?["mrope_section"], let array = v.asInts() {
                // Python: mrope_section = np.cumsum(mrope_section * 2)[:-1].tolist()
                let doubled = array.map { $0 * 2 }
                var cumsum = [Int]()
                var running = 0
                for val in doubled {
                    running += val
                    cumsum.append(running)
                }
                // Remove the last element [:-1]
                self.mropeSection = Array(cumsum.dropLast())
            } else {
                fatalError("rope_scaling['mrope_section'] must be an array of integers")
            }

            self._rotaryEmbedding.wrappedValue = QwenVL.Qwen2RotaryEmbedding(
                dimensions: headDim,
                maxPositionEmbeddings: args.maxPositionEmbeddings,
                base: args.ropeTheta)
        }

        public func callAsFunction(
            _ x: MLXArray, mask: MLXArray? = nil, cache: KVCache? = nil,
            positionIds: MLXArray? = nil
        ) -> MLXArray {
            let (B, L) = (x.dim(0), x.dim(1))

            var queries = wq(x)
            var keys = wk(x)
            var values = wv(x)

            // Prepare the queries, keys and values for the attention computation
            queries = queries.reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
            keys = keys.reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
            values = values.reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)

            var kvSeqLen = keys.dim(-2)
            var computedPositionIds: MLXArray

            if let positionIds {
                kvSeqLen += (cache?.offset ?? 0) + 1
                computedPositionIds = positionIds
            } else {
                let offset = cache?.offset ?? 0
                kvSeqLen += offset + 1
                computedPositionIds = MLXArray(offset ..< (offset + L))
                computedPositionIds = expandedDimensions(computedPositionIds, axis: 0)
                computedPositionIds = tiled(computedPositionIds, repetitions: [3, 1, 1])
            }

            let (cos, sin) = rotaryEmbedding(values, seqLen: kvSeqLen)

            // Ensure mask has correct dtype and shape
            var processedMask = mask?[0..., 0 ..< keys.dim(-2)]
            if let mask = processedMask, mask.dtype != queries.dtype {
                processedMask = mask.asType(queries.dtype)
            }

            // Apply multimodal rotary position embedding
            (queries, keys) = Language.applyMultimodalRotaryPositionEmbedding(
                q: queries, k: keys, cos: cos, sin: sin,
                positionIds: computedPositionIds, mropeSection: mropeSection
            )

            if let cache = cache {
                (keys, values) = cache.update(keys: keys, values: values)
            }

            let output = MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: scale, mask: processedMask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)

            return wo(output)
        }
    }

    fileprivate class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "up_proj") var up: Linear
        @ModuleInfo(key: "down_proj") var down: Linear

        public init(dimensions: Int, hiddenDimensions: Int) {
            self._gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            self._up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            self._down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(silu(gate(x)) * up(x))
        }
    }

    fileprivate class Qwen25VLDecoderLayer: Module {

        @ModuleInfo(key: "self_attn") var attention: Attention
        let mlp: MLP

        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        public init(_ args: Qwen25VLConfiguration.TextConfiguration) {
            self._attention.wrappedValue = Attention(args)
            self.mlp = MLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
            self._inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
            self._postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
        }

        public func callAsFunction(
            _ x: MLXArray, mask: MLXArray? = nil, cache: KVCache? = nil,
            positionIds: MLXArray? = nil
        ) -> MLXArray {
            var r = attention(inputLayerNorm(x), mask: mask, cache: cache, positionIds: positionIds)
            let h = x + r
            r = mlp(postAttentionLayerNorm(h))
            let out = h + r
            return out
        }
    }

    fileprivate class Qwen25Model: Module {

        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

        fileprivate let layers: [Qwen25VLDecoderLayer]
        fileprivate let norm: RMSNorm

        public init(_ args: Qwen25VLConfiguration.TextConfiguration) {
            precondition(args.vocabularySize > 0)

            self._embedTokens.wrappedValue = Embedding(
                embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

            self.layers = (0 ..< args.hiddenLayers)
                .map { _ in
                    Qwen25VLDecoderLayer(args)
                }
            self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        }

        public func callAsFunction(
            _ inputs: MLXArray?,
            cache: [KVCache]? = nil,
            inputEmbedding: MLXArray? = nil,
            positionIds: MLXArray? = nil
        ) -> MLXArray {
            var h: MLXArray
            if let inputEmbedding = inputEmbedding {
                h = inputEmbedding
            } else if let inputs = inputs {
                h = embedTokens(inputs)
            } else {
                fatalError("one of inputs or inputEmbedding must be non-nil")
            }

            let mask: MLXArray? = createAttentionMask(h: h, cache: cache)

            for (i, layer) in layers.enumerated() {
                h = layer(h, mask: mask, cache: cache?[i], positionIds: positionIds)
            }

            return norm(h)
        }
    }

    fileprivate class LanguageModel: Module, KVCacheDimensionProvider {
        @ModuleInfo var model: Qwen25Model
        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        var kvHeads: [Int]
        private var ropeDeltas: MLXArray?

        // Store the needed config values
        private let spatialMergeSize: Int
        private let imageTokenId: Int
        private let videoTokenId: Int
        private let visionStartTokenId: Int

        public init(_ args: Qwen25VLConfiguration.TextConfiguration, config: Qwen25VLConfiguration)
        {
            self.model = Qwen25Model(args)

            if !args.tieWordEmbeddings {
                _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
            }

            self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }

            // Store the config values we need
            self.spatialMergeSize = config.visionConfiguration.spatialMergeSize
            self.imageTokenId = config.baseConfiguration.imageTokenId
            self.videoTokenId = config.baseConfiguration.videoTokenId
            self.visionStartTokenId = config.baseConfiguration.visionStartTokenId
        }

        private func getRopeIndex(
            inputIds: MLXArray,
            imageGridThw: [THW]?,
            videoGridThw: [THW]?,
            attentionMask: MLXArray? = nil
        ) -> (MLXArray, MLXArray) {
            let batchSize = inputIds.dim(0)
            let seqLength = inputIds.dim(1)

            // Default position IDs
            var positionIds = MLXArray(0 ..< seqLength)
            positionIds = broadcast(positionIds[.newAxis, 0...], to: [batchSize, seqLength])

            var mropePositionDeltas = [Int]()

            let hasImages = imageGridThw?.isEmpty == false
            let hasVideos = videoGridThw?.isEmpty == false

            if hasImages || hasVideos {
                // Create 3D position IDs for multimodal content
                positionIds = ones([3, batchSize, seqLength]).asType(inputIds.dtype)

                var imageIndex = 0
                var videoIndex = 0

                for batchIdx in 0 ..< batchSize {
                    let inputIdsForBatch = inputIds[batchIdx]

                    // Count image and video tokens
                    let imageCount = sum(inputIdsForBatch .== imageTokenId).item(Int.self)
                    let videoCount = sum(inputIdsForBatch .== videoTokenId).item(Int.self)

                    var llmPosIdsList = [MLXArray]()
                    var start = 0
                    var remainingImages = imageCount
                    var remainingVideos = videoCount

                    // Process each image/video token
                    for _ in 0 ..< (imageCount + videoCount) {
                        // Find next image or video token position
                        let inputTokens = inputIdsForBatch.asArray(Int.self)

                        var edImage = inputTokens.count + 1
                        var edVideo = inputTokens.count + 1

                        if remainingImages > 0 {
                            if let relativePos = inputTokens[start...].firstIndex(of: imageTokenId)
                            {
                                edImage = relativePos  // firstIndex already returns absolute index
                            }
                        }
                        if remainingVideos > 0 {
                            if let relativePos = inputTokens[start...].firstIndex(of: videoTokenId)
                            {
                                edVideo = relativePos  // firstIndex already returns absolute index
                            }
                        }

                        let (t, h, w): (Int, Int, Int)
                        let ed: Int

                        if edImage < edVideo {
                            // Use image frame - make sure we have images and valid index
                            guard let imageGridThw = imageGridThw, imageIndex < imageGridThw.count
                            else {
                                break
                            }
                            let frame = imageGridThw[imageIndex]
                            (t, h, w) = frame.values
                            imageIndex += 1
                            remainingImages -= 1
                            ed = edImage
                        } else {
                            // Use video frame - make sure we have videos and valid index
                            guard let videoGridThw = videoGridThw, videoIndex < videoGridThw.count
                            else {
                                break
                            }
                            let frame = videoGridThw[videoIndex]
                            (t, h, w) = frame.values
                            videoIndex += 1
                            remainingVideos -= 1
                            ed = edVideo
                        }

                        let llmGridT = t
                        let llmGridH = h / spatialMergeSize
                        let llmGridW = w / spatialMergeSize

                        // Text tokens before this token
                        let textLen = ed - start
                        if textLen > 0 {
                            let stIdx =
                                llmPosIdsList.isEmpty
                                ? 0 : (llmPosIdsList.last!.max() + 1).item(Int.self)
                            let textIndices = broadcast(
                                (MLXArray(0 ..< textLen) + stIdx)[.newAxis, 0...],
                                to: [3, textLen]
                            )
                            llmPosIdsList.append(textIndices)
                        }

                        // Vision tokens
                        let stIdx =
                            llmPosIdsList.isEmpty
                            ? 0 : (llmPosIdsList.last!.max() + 1).item(Int.self)

                        let tIndex = broadcast(
                            MLXArray(0 ..< llmGridT)[0..., .newAxis],
                            to: [llmGridT, llmGridH * llmGridW]
                        ).flattened()

                        let hIndex = broadcast(
                            MLXArray(0 ..< llmGridH)[.newAxis, 0..., .newAxis],
                            to: [llmGridT, llmGridH, llmGridW]
                        ).flattened()

                        let wIndex = broadcast(
                            MLXArray(0 ..< llmGridW)[.newAxis, .newAxis, 0...],
                            to: [llmGridT, llmGridH, llmGridW]
                        ).flattened()

                        let visionIndices =
                            stacked([tIndex, hIndex, wIndex], axis: 0) + MLXArray(textLen + stIdx)
                        llmPosIdsList.append(visionIndices)

                        start = ed + llmGridT * llmGridH * llmGridW
                    }

                    // Remaining text tokens
                    if start < seqLength {
                        let remainingLen = seqLength - start
                        let stIdx =
                            llmPosIdsList.isEmpty
                            ? 0 : (llmPosIdsList.last!.max() + 1).item(Int.self)
                        let remainingIndices = broadcast(
                            (MLXArray(0 ..< remainingLen) + stIdx)[.newAxis, 0...],
                            to: [3, remainingLen]
                        )
                        llmPosIdsList.append(remainingIndices)
                    }

                    // Concatenate all position IDs for this batch
                    if !llmPosIdsList.isEmpty {
                        let llmPositions = concatenated(llmPosIdsList, axis: 1)

                        // Validate shape match before assignment
                        if llmPositions.dim(1) != seqLength {
                            print(
                                "Warning: Position ID length mismatch. Expected: \(seqLength), got: \(llmPositions.dim(1))"
                            )
                            // Pad or truncate to match expected length
                            if llmPositions.dim(1) < seqLength {
                                let padWidth = seqLength - llmPositions.dim(1)
                                let lastPos = llmPositions[0..., -1].max() + 1
                                let padding = broadcast(
                                    (MLXArray(0 ..< padWidth) + lastPos)[.newAxis, 0...],
                                    to: [3, padWidth]
                                )
                                let paddedPositions = concatenated(
                                    [llmPositions, padding], axis: 1)
                                positionIds[0..., batchIdx, 0...] = paddedPositions
                            } else {
                                // Truncate if too long
                                positionIds[0..., batchIdx, 0...] =
                                    llmPositions[0..., 0 ..< seqLength]
                            }
                        } else {
                            positionIds[0..., batchIdx, 0...] = llmPositions
                        }

                        mropePositionDeltas.append(
                            (llmPositions[0..., 0 ..< min(llmPositions.dim(1), seqLength)].max() + 1
                                - seqLength).item(Int.self))
                    } else {
                        mropePositionDeltas.append(0)
                    }
                }
            } else {
                // Handle text-only case or case with attention mask
                if let attentionMask = attentionMask {
                    // Python: position_ids = mx.cumsum(attention_mask.astype(mx.int64), axis=-1) - 1
                    positionIds = cumsum(attentionMask.asType(.int64), axis: -1) - 1
                    positionIds = `where`(
                        attentionMask .== 0,
                        ones(like: positionIds),
                        positionIds
                    )
                    positionIds = expandedDimensions(positionIds[0], axis: 0)
                    positionIds = tiled(positionIds, repetitions: [3, 1, 1])

                    let maxPositionIds = positionIds.max(axis: 0, keepDims: false).max(
                        axis: -1, keepDims: true)
                    mropePositionDeltas = [
                        (maxPositionIds + 1 - attentionMask.dim(-1)).item(Int.self)
                    ]
                } else {
                    // Simple case - just sequential position IDs
                    positionIds = MLXArray(0 ..< seqLength)[.newAxis, 0...]
                    positionIds = broadcast(positionIds, to: [batchSize, seqLength])
                    positionIds = broadcast(
                        positionIds[.newAxis, 0..., 0...],
                        to: [3, batchSize, seqLength]
                    )
                    mropePositionDeltas = Array(repeating: 0, count: batchSize)
                }
            }

            let deltaArray = MLXArray(mropePositionDeltas.isEmpty ? [0] : mropePositionDeltas)
            return (positionIds, deltaArray)
        }

        public func callAsFunction(
            _ inputs: MLXArray,  // Make this required, not optional
            cache: [KVCache]? = nil,
            inputEmbedding: MLXArray? = nil,
            positionIds: MLXArray? = nil,
            pixelValues: MLXArray? = nil,
            imageGridThw: [THW]? = nil,
            videoGridThw: [THW]? = nil
        ) -> LMOutput {
            var computedPositionIds = positionIds

            // Reset rope deltas when processing new image/video
            if pixelValues != nil {
                ropeDeltas = nil
            }

            // Calculate position IDs if not provided and mask conditions are met
            if computedPositionIds == nil {
                // Calculate RoPE index once per generation in the pre-fill stage only
                let shouldCalculateRope =
                    ((cache != nil && cache?[0].offset == 0) || ropeDeltas == nil || cache == nil)

                if shouldCalculateRope {
                    let (newPositionIds, newRopeDeltas) = getRopeIndex(
                        inputIds: inputs,
                        imageGridThw: imageGridThw,
                        videoGridThw: videoGridThw
                    )
                    computedPositionIds = newPositionIds
                    ropeDeltas = newRopeDeltas
                } else {
                    // Use pre-calculated rope deltas for generation continuation
                    let batchSize = inputs.dim(0)
                    let seqLength = inputs.dim(1)

                    var delta = ropeDeltas ?? MLXArray([0])
                    if let cache = cache {
                        let cacheOffset = MLXArray([cache.last?.offset ?? 0])
                        delta = delta + cacheOffset
                    }

                    // Create position IDs for this sequence
                    var newPositionIds = broadcast(
                        MLXArray(0 ..< seqLength)[.newAxis, 0...],
                        to: [batchSize, seqLength]
                    )

                    // Add delta to each batch - ensure shapes are compatible
                    if batchSize > 0 && delta.size > 0 {
                        // Ensure delta has the right shape [batchSize]
                        let deltaRepeated: MLXArray
                        if delta.size == 1 {
                            deltaRepeated = broadcast(delta, to: [batchSize])
                        } else if delta.dim(0) != batchSize {
                            deltaRepeated = broadcast(delta[0], to: [batchSize])
                        } else {
                            deltaRepeated = delta
                        }

                        newPositionIds = newPositionIds + deltaRepeated[0..., .newAxis]
                    }

                    // Broadcast to 3D for multimodal RoPE
                    computedPositionIds = broadcast(
                        newPositionIds[.newAxis, 0..., 0...],
                        to: [3, batchSize, seqLength]
                    )
                }
            }

            // Get the embeddings
            var h: MLXArray
            if let inputEmbedding = inputEmbedding {
                h = inputEmbedding
            } else {
                h = model.embedTokens(inputs)
            }

            // Create attention mask
            let mask: MLXArray? = createAttentionMask(h: h, cache: cache)

            // Process through transformer layers
            for (i, layer) in model.layers.enumerated() {
                h = layer(h, mask: mask, cache: cache?[i], positionIds: computedPositionIds)
            }

            // Apply final layer norm
            h = model.norm(h)

            // Apply output projection
            if let lmHead = lmHead {
                h = lmHead(h)
            } else {
                // Use tied embeddings
                h = model.embedTokens.asLinear(h)
            }

            return LMOutput(logits: h)
        }
    }
}

// MARK: - Vision

private enum Vision {

    static fileprivate func applyMultimodalRotaryPositionEmbedding(
        _ tensor: MLXArray, freqs: MLXArray
    ) -> MLXArray {
        var cos = cos(freqs)
        var sin = sin(freqs)

        cos = expandedDimensions(cos, axis: 1)
        cos = tiled(cos, repetitions: [1, 1, 2])
        cos = expandedDimensions(cos, axis: 0)

        sin = expandedDimensions(sin, axis: 1)
        sin = tiled(sin, repetitions: [1, 1, 2])
        sin = expandedDimensions(sin, axis: 0)

        let output = (tensor * cos) + (QwenVL.rotateHalf(tensor) * sin)
        return output.asType(tensor.dtype)
    }

    fileprivate class PatchMerger: Module, UnaryLayer {
        let hiddenSize: Int
        @ModuleInfo(key: "ln_q") var layerNormQ: RMSNorm
        @ModuleInfo var mlp: (Linear, GELU, Linear)

        init(dimensions: Int, contextDimensions: Int, spatialMergeSize: Int) {
            self.hiddenSize = contextDimensions * (spatialMergeSize * spatialMergeSize)
            self._layerNormQ.wrappedValue = RMSNorm(dimensions: contextDimensions, eps: 1e-6)
            self.mlp = (
                Linear(hiddenSize, hiddenSize),
                GELU(),
                Linear(hiddenSize, dimensions)
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var x = layerNormQ(x).reshaped(-1, hiddenSize)
            x = mlp.0(x)
            x = mlp.1(x)
            x = mlp.2(x)
            return x
        }
    }

    fileprivate class Attention: Module {

        let numHeads: Int
        let scale: Float

        @ModuleInfo(key: "qkv") var qkv: Linear
        @ModuleInfo(key: "proj") var proj: Linear

        public init(dims: Int, numHeads: Int) {
            self.numHeads = numHeads
            let headDim = dims / numHeads
            self.scale = pow(Float(headDim), -0.5)

            self._qkv.wrappedValue = Linear(dims, 3 * dims, bias: true)
            self._proj.wrappedValue = Linear(dims, dims)
        }

        public func callAsFunction(
            _ x: MLXArray, attentionMask: MLXArray, rotaryPositionEmbedding: MLXArray
        ) -> MLXArray {
            let sequenceLength = x.dim(0)

            let qkv = qkv(x)
            let s = split(qkv, parts: 3, axis: -1)
            var (q, k, v) = (s[0], s[1], s[2])

            q = q.reshaped(sequenceLength, numHeads, -1)
            k = k.reshaped(sequenceLength, numHeads, -1)
            v = v.reshaped(sequenceLength, numHeads, -1)

            q = applyMultimodalRotaryPositionEmbedding(q, freqs: rotaryPositionEmbedding)
            k = applyMultimodalRotaryPositionEmbedding(k, freqs: rotaryPositionEmbedding)

            q = q.reshaped(1, sequenceLength, numHeads, -1).transposed(0, 2, 1, 3)
            k = k.reshaped(1, sequenceLength, numHeads, -1).transposed(0, 2, 1, 3)
            v = v.reshaped(1, sequenceLength, numHeads, -1).transposed(0, 2, 1, 3)

            let output = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: scale, mask: attentionMask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(sequenceLength, -1)

            return proj(output)
        }
    }

    fileprivate class MLP: Module, UnaryLayer {

        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "up_proj") var up: Linear
        @ModuleInfo(key: "down_proj") var down: Linear

        public init(dimensions: Int, hiddenDimensions: Int) {
            self._gate.wrappedValue = Linear(dimensions, hiddenDimensions)
            self._up.wrappedValue = Linear(dimensions, hiddenDimensions)
            self._down.wrappedValue = Linear(hiddenDimensions, dimensions)
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(silu(gate(x)) * up(x))
        }
    }

    fileprivate class Qwen25VLVisionBlock: Module {

        @ModuleInfo var norm1: RMSNorm
        @ModuleInfo var norm2: RMSNorm
        @ModuleInfo(key: "attn") var attention: Attention
        @ModuleInfo var mlp: MLP

        public init(_ config: Qwen25VLConfiguration.VisionConfiguration) {
            self.norm1 = RMSNorm(dimensions: config.hiddenSize, eps: 1e-6)
            self.norm2 = RMSNorm(dimensions: config.hiddenSize, eps: 1e-6)

            self._attention.wrappedValue = Attention(
                dims: config.hiddenSize, numHeads: config.numHeads)

            self.mlp = MLP(
                dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
        }

        func callAsFunction(
            _ hiddenStates: MLXArray, attentionMask: MLXArray, rotaryPositionEmbedding: MLXArray
        ) -> MLXArray {
            var hiddenStates =
                hiddenStates
                + attention(
                    norm1(hiddenStates),
                    attentionMask: attentionMask,
                    rotaryPositionEmbedding: rotaryPositionEmbedding
                )
            hiddenStates = hiddenStates + mlp(norm2(hiddenStates))
            return hiddenStates
        }
    }

    fileprivate class VisionModel: Module {

        @ModuleInfo(key: "patch_embed") var patchEmbed: QwenVL.PatchEmbed
        @ModuleInfo(key: "rotary_pos_emb") var rotaryPositionEmbedding: QwenVL.VisionRotaryEmbedding
        @ModuleInfo(key: "blocks") var blocks: [Qwen25VLVisionBlock]
        @ModuleInfo(key: "merger") var patchMerger: PatchMerger

        let spatialMergeSize: Int
        let windowSize: Int
        let patchSize: Int
        let spatialMergeUnit: Int
        let fullattBlockIndexes: [Int]

        public init(_ config: Qwen25VLConfiguration.VisionConfiguration) {
            self.spatialMergeSize = config.spatialMergeSize
            self.windowSize = config.windowSize
            self.patchSize = config.patchSize
            self.spatialMergeUnit = config.spatialMergeSize * config.spatialMergeSize
            self.fullattBlockIndexes = config.fullattBlockIndexes

            self._patchEmbed.wrappedValue = QwenVL.PatchEmbed(
                patchSize: config.patchSize,
                temporalPatchSize: config.temporalPatchSize,
                inChannels: config.inChannels,
                hiddenSize: config.hiddenSize)

            let headDimensions = config.hiddenSize / config.numHeads
            self._rotaryPositionEmbedding.wrappedValue = QwenVL.VisionRotaryEmbedding(
                dimensions: headDimensions / 2, theta: 10_000)

            self._blocks.wrappedValue = (0 ..< config.depth).map { _ in
                Qwen25VLVisionBlock(config)
            }
            self._patchMerger.wrappedValue = PatchMerger(
                dimensions: config.outHiddenSize, contextDimensions: config.hiddenSize,
                spatialMergeSize: config.spatialMergeSize)
        }

        func rotaryPositionEmbedding(_ frames: [THW]) -> MLXArray {
            var positionIds = [MLXArray]()

            for row in frames {
                let (t, h, w) = row.values

                var hposIds = expandedDimensions(MLXArray(0 ..< h), axis: 1)
                hposIds = repeated(hposIds, count: w, axis: 1)
                hposIds =
                    hposIds
                    .reshaped(
                        h / spatialMergeSize,
                        spatialMergeSize,
                        w / spatialMergeSize,
                        spatialMergeSize
                    )
                    .transposed(0, 2, 1, 3)
                    .flattened()

                var wposIds = expandedDimensions(MLXArray(0 ..< w), axis: 0)
                wposIds = repeated(wposIds, count: h, axis: 0)
                wposIds =
                    wposIds
                    .reshaped(
                        h / spatialMergeSize,
                        spatialMergeSize,
                        w / spatialMergeSize,
                        spatialMergeSize
                    )
                    .transposed(0, 2, 1, 3)
                    .flattened()

                let stackedPosIds = stacked([hposIds, wposIds], axis: -1)
                positionIds.append(tiled(stackedPosIds, repetitions: [t, 1]))
            }

            let indices = concatenated(positionIds, axis: 0)
            let maxFrameSize = frames.lazy.map { max($0.h, $0.w) }.max() ?? 0
            let rotaryPositionEmbedFull = rotaryPositionEmbedding(sequenceLength: maxFrameSize)[
                indices]

            return rotaryPositionEmbedFull.reshaped(indices.dim(0), -1)
        }

        func getWindowIndex(_ frames: [THW]) -> (MLXArray, MLXArray) {
            var windowIndex = [MLXArray]()
            var cuWindowSeqlens = [0]
            var windowIndexId = 0
            let vitMergerWindowSize = windowSize / spatialMergeSize / patchSize

            for frame in frames {
                let (gridT, gridH, gridW) = frame.values
                let llmGridH = gridH / spatialMergeSize
                let llmGridW = gridW / spatialMergeSize

                let index = MLXArray(0 ..< (gridT * llmGridH * llmGridW)).reshaped(
                    gridT, llmGridH, llmGridW)

                let padH = vitMergerWindowSize - llmGridH % vitMergerWindowSize
                let padW = vitMergerWindowSize - llmGridW % vitMergerWindowSize
                let numWindowsH = (llmGridH + padH) / vitMergerWindowSize
                let numWindowsW = (llmGridW + padW) / vitMergerWindowSize

                // Pad the index
                let indexPadded = padded(
                    index,
                    widths: [[0, 0], [0, padH], [0, padW]],
                    mode: .constant,
                    value: MLXArray(-100)
                )

                // Reshape and transpose
                let indexReshaped = indexPadded.reshaped(
                    gridT,
                    numWindowsH,
                    vitMergerWindowSize,
                    numWindowsW,
                    vitMergerWindowSize
                )

                let indexTransposed = indexReshaped.transposed(0, 1, 3, 2, 4).reshaped(
                    gridT,
                    numWindowsH * numWindowsW,
                    vitMergerWindowSize,
                    vitMergerWindowSize
                )

                // Calculate sequence lengths
                let seqlens = sum(indexTransposed .!= -100, axes: [2, 3]).reshaped(-1)

                // Get valid indices
                let indexFlattened = indexTransposed.flattened()
                let validIndices = indexFlattened.asArray(Int.self).enumerated()
                    .filter { $0.element != -100 }
                    .map { $0.offset }

                let validValues = indexFlattened[MLXArray(validIndices)]

                // Add to window index
                windowIndex.append(validValues + windowIndexId)

                // Update cumulative sequence lengths
                var cuSeqlens = [Int]()
                for frame in frames {
                    let seqLen = frame.h * frame.w
                    for _ in 0 ..< frame.t {
                        cuSeqlens.append(seqLen)
                    }
                }

                windowIndexId += gridT * llmGridH * llmGridW
            }

            // Concatenate all window indices
            let combinedWindowIndex = concatenated(windowIndex, axis: 0)
            let cuWindowSeqlensArray = MLXArray(cuWindowSeqlens)

            // Get unique values in cuWindowSeqlens
            var seen = Set<Int>()
            var uniqueIndices = [Int]()

            for (i, value) in cuWindowSeqlens.enumerated() {
                if !seen.contains(value) {
                    seen.insert(value)
                    uniqueIndices.append(i)
                }
            }

            let uniqueCuWindowSeqlens = cuWindowSeqlensArray[MLXArray(uniqueIndices)]

            return (combinedWindowIndex, uniqueCuWindowSeqlens)
        }

        func attentionMask(sequenceLength: Int, cuSeqlens: MLXArray, dtype: DType) -> MLXArray {
            // Create attention mask filled with minimum values for the specific dtype
            let minValue: Float
            switch dtype {
            case .float16:
                minValue = -65504.0  // Minimum value for float16
            case .float32:
                minValue = -Float.greatestFiniteMagnitude
            case .bfloat16:
                minValue = -3.38953e38  // Approximate minimum for bfloat16
            default:
                minValue = -Float.greatestFiniteMagnitude
            }

            var attentionMask = full(
                [1, sequenceLength, sequenceLength], values: MLXArray(minValue)
            ).asType(dtype)

            // Update mask for each sequence (set to 0 for valid attention)
            let cuSeqlens = cuSeqlens.asArray(Int.self)
            for i in 1 ..< cuSeqlens.count {
                let start = cuSeqlens[i - 1]
                let end = cuSeqlens[i]
                attentionMask[0..., start ..< end, start ..< end] = MLXArray(0.0).asType(dtype)
            }

            return attentionMask
        }

        public func callAsFunction(_ hiddenStates: MLXArray, frames: [THW]) -> MLXArray {
            var hiddenStates = patchEmbed(hiddenStates)
            let rotaryPosEmb = rotaryPositionEmbedding(frames)

            // Get window indices and sequence lengths
            let (windowIndex, cuWindowSeqlens) = getWindowIndex(frames)

            var cuSeqlens = [Int]()
            for frame in frames {
                let seqLen = frame.h * frame.w
                for _ in 0 ..< frame.t {
                    cuSeqlens.append(seqLen)
                }
            }

            // Convert to cumulative sum and pad with 0 at beginning
            var cumulativeSeqlens = [0]
            var runningSum = 0
            for seqLen in cuSeqlens {
                runningSum += seqLen
                cumulativeSeqlens.append(runningSum)
            }
            let cuSeqlensArray = MLXArray(cumulativeSeqlens)

            let seqLen = hiddenStates.dim(0)
            let fullAttentionMask = attentionMask(
                sequenceLength: seqLen, cuSeqlens: cuSeqlensArray, dtype: hiddenStates.dtype)
            let windowAttentionMask = attentionMask(
                sequenceLength: seqLen, cuSeqlens: cuWindowSeqlens, dtype: hiddenStates.dtype)

            // Reshape and reindex hidden states
            hiddenStates = hiddenStates.reshaped(seqLen / spatialMergeUnit, spatialMergeUnit, -1)
            hiddenStates = hiddenStates[windowIndex, 0..., 0...]
            hiddenStates = hiddenStates.reshaped(seqLen, -1)

            // Reshape and reindex rotary position embeddings
            var rotaryPosEmbReshaped = rotaryPosEmb.reshaped(
                seqLen / spatialMergeUnit, spatialMergeUnit, -1)
            rotaryPosEmbReshaped = rotaryPosEmbReshaped[windowIndex, 0..., 0...]
            rotaryPosEmbReshaped = rotaryPosEmbReshaped.reshaped(seqLen, -1)

            // Process through blocks
            for (i, block) in blocks.enumerated() {
                let attentionMask =
                    fullattBlockIndexes.contains(i) ? fullAttentionMask : windowAttentionMask

                hiddenStates = block(
                    hiddenStates,
                    attentionMask: attentionMask,
                    rotaryPositionEmbedding: rotaryPosEmbReshaped
                )
            }

            // Apply patch merger
            hiddenStates = patchMerger(hiddenStates)

            // Reorder back to original sequence
            let reverseIndices = argSort(windowIndex, axis: 0)
            hiddenStates = hiddenStates[reverseIndices, 0...]

            return hiddenStates
        }

        private func isMLXWeight(_ array: MLXArray) -> Bool {
            if array.ndim != 4, array.ndim != 5 {
                return false
            }

            if array.dim(-1) == 3 {
                return true
            }

            let (outChannels, kH, kW) = (array.dim(1), array.dim(2), array.dim(3))
            return outChannels >= kH && outChannels >= kW && kH == kW
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            var sanitizedWeights = [String: MLXArray]()

            for (k, v) in weights {
                if k.contains("position_id") {
                    // Remove unused position_ids
                    continue
                } else if k.contains("patch_embed.proj.weight") {
                    // PyTorch conv2d weight tensors have shape:
                    //   [B, out_channels, in_channels, kH, KW]
                    // MLX conv2d expects the weight be of shape:
                    //   [B, out_channels, kH, KW, in_channels]
                    if isMLXWeight(v) {
                        sanitizedWeights[k] = v
                    } else {
                        sanitizedWeights[k] = v.transposed(0, 2, 3, 4, 1)
                    }
                } else {
                    sanitizedWeights[k] = v
                }
            }

            return sanitizedWeights
        }
    }
}

// MARK: - Processor

/// Qwen2.5VL VLM `UserInputProcessor`.
///
/// This is meant to be used with ``Qwen25VL`` and is typically created by ``VLMModelFactory``.
public class Qwen25VLProcessor: UserInputProcessor {
    private let config: Qwen25VLProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: Qwen25VLProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    func preprocess(image: CIImage, resizedSize: CGSize) -> CIImage {
        image
            .toSRGB()
            .resampled(to: resizedSize, method: .bicubic)
            .normalized(mean: config.imageMeanTuple, std: config.imageStdTuple)
    }

    public func preprocess(images: [CIImage], processing: UserInput.Processing?) throws -> (
        MLXArray, THW
    ) {
        // First apply the user requested resizing, etc. if any
        let images = images.map { MediaProcessing.apply($0, processing: processing) }

        // image_processing_qwen2_vl._preprocess
        let size = images[0].extent.size
        let (resizedHeight, resizedWidth) = try QwenVL.targetSize(
            height: Int(size.height), width: Int(size.width),
            factor: config.patchSize * config.mergeSize,
            minPixels: config.size.minPixels, maxPixels: config.size.maxPixels)
        let resizedSize = CGSize(width: resizedWidth, height: resizedHeight)

        // Process images
        let processedImages =
            try images
            .map {
                MediaProcessing.inSRGBToneCurveSpace($0)
            }
            .map {
                return try MediaProcessing.resampleBicubic($0, to: resizedSize)
            }
            .map {
                MediaProcessing.normalize(
                    $0, mean: config.imageMeanTuple, std: config.imageStdTuple)
            }
            .map {
                MediaProcessing.asMLXArray($0)
            }

        return try QwenVL.patchify(
            images: processedImages, mergeSize: config.mergeSize, patchSize: config.patchSize,
            temporalPatchSize: config.temporalPatchSize)
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let messages = Qwen2VLMessageGenerator().generate(from: input)

        var promptTokens = try tokenizer.applyChatTemplate(messages: messages)

        // Text-only input
        if input.images.isEmpty, input.videos.isEmpty {
            return LMInput(tokens: MLXArray(promptTokens))
        }

        // Process images if any
        var processedImage: LMInput.ProcessedImage?
        if !input.images.isEmpty {
            let imagePixelsAndFrames = try input.images.map {
                try preprocess(images: [$0.asCIImage()], processing: input.processing)
            }
            let imagePixelsConcatenated = concatenated(imagePixelsAndFrames.map { $0.0 })
            processedImage = LMInput.ProcessedImage(
                pixels: imagePixelsConcatenated, frames: imagePixelsAndFrames.map { $0.1 })

            if let imageFrames = processedImage?.frames {
                promptTokens = try QwenVL.replacePaddingTokens(
                    in: promptTokens, frames: imageFrames, paddingToken: "<|image_pad|>",
                    mergeSize: config.mergeSize, tokenizer: tokenizer)
            }
        }

        // Process videos if any
        var processedVideo: LMInput.ProcessedVideo?
        if !input.videos.isEmpty {
            var videosAsImageSequences = [[MLXArray]]()
            var resizedSize: CGSize = .zero
            for video in input.videos {
                let imageSequence = try await MediaProcessing.asProcessedSequence(
                    video.asAVAsset(), samplesPerSecond: 2
                ) { frame in
                    // first apply the user requested resizing, etc. if any
                    let resizedImage = MediaProcessing.apply(
                        frame.frame, processing: input.processing)
                    if resizedSize == .zero {
                        let size = resizedImage.extent.size
                        let (resizedHeight, resizedWidth) = try QwenVL.targetSize(
                            height: Int(size.height), width: Int(size.width),
                            factor: config.patchSize * config.mergeSize,
                            minPixels: config.minPixels, maxPixels: config.maxPixels)
                        resizedSize = CGSize(width: resizedWidth, height: resizedHeight)
                    }
                    let processedImage = preprocess(image: resizedImage, resizedSize: resizedSize)
                    return VideoFrame(frame: processedImage, timeStamp: frame.timeStamp)
                }
                videosAsImageSequences.append(imageSequence.frames)
            }
            let videoPixelsAndFrames = try videosAsImageSequences.map {
                try QwenVL.patchify(
                    images: $0, mergeSize: config.mergeSize, patchSize: config.patchSize,
                    temporalPatchSize: config.temporalPatchSize)
            }
            let videoPixelsConcatenated = concatenated(videoPixelsAndFrames.map { $0.0 })
            processedVideo = LMInput.ProcessedVideo(
                pixels: videoPixelsConcatenated, frames: videoPixelsAndFrames.map { $0.1 })
            if let videoFrames = processedVideo?.frames {
                promptTokens = try QwenVL.replacePaddingTokens(
                    in: promptTokens, frames: videoFrames, paddingToken: "<|video_pad|>",
                    mergeSize: config.mergeSize, tokenizer: tokenizer)
            }
        }

        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)
        return LMInput(
            text: .init(tokens: promptArray, mask: mask),
            image: processedImage,
            video: processedVideo)
    }
}

// MARK: - Model

/// Qwen2.5VL VLM
///
/// This is typically created by ``VLMModelFactory``.
public class Qwen25VL: Module, VLMModel, KVCacheDimensionProvider {

    @ModuleInfo(key: "vision_tower") private var visionModel: Vision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: Language.LanguageModel

    public let config: Qwen25VLConfiguration

    public var vocabularySize: Int { config.baseConfiguration.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    public func loraLinearLayers() -> MLXLMCommon.LoRALinearLayers {
        languageModel.model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }

    public init(_ config: Qwen25VLConfiguration) {
        self.config = config
        self._visionModel.wrappedValue = Vision.VisionModel(config.visionConfiguration)
        self._languageModel.wrappedValue = Language.LanguageModel(
            config.textConfiguration, config: config)  // Pass full config
    }

    private func inputEmbeddings(inputIds: MLXArray, pixelValues: MLXArray?, frames: [THW]?)
        -> MLXArray
    {
        guard let pixelValues, let frames else {
            return languageModel.model.embedTokens(inputIds[.newAxis, .ellipsis])
        }

        // Get the input embeddings from the language model
        let inputEmbeds = languageModel.model.embedTokens(inputIds)

        // Get the ouptut hidden states from the vision model
        var hiddenStates = self.visionModel(pixelValues, frames: frames)

        if hiddenStates.ndim == 2 {
            hiddenStates = hiddenStates[.newAxis, 0..., 0...]
        }

        // Insert special image tokens in the input_ids
        return QwenVL.mergeInputIdsWithImageFeatures(
            inputIds: inputIds, inputEmbeds: inputEmbeds, imageFeatures: hiddenStates,
            imageTokenId: config.baseConfiguration.imageTokenId,
            videoTokenId: config.baseConfiguration.videoTokenId)
    }

    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        let dtype = visionModel.patchEmbed.proj.weight.dtype

        // Process both images and videos together
        var allPixels: MLXArray?
        var allFrames: [THW] = []

        if let imagePixels = input.image?.pixels, let imageFrames = input.image?.frames {
            allPixels = imagePixels.asType(dtype)
            allFrames.append(contentsOf: imageFrames)
        }

        if let videoPixels = input.video?.pixels, let videoFrames = input.video?.frames {
            if allPixels == nil {
                allPixels = videoPixels.asType(dtype)
            } else {
                allPixels = concatenated([allPixels!, videoPixels.asType(dtype)])
            }
            allFrames.append(contentsOf: videoFrames)
        }

        let inputEmbeddings = self.inputEmbeddings(
            inputIds: input.text.tokens, pixelValues: allPixels,
            frames: allFrames.isEmpty ? nil : allFrames)

        // Pass BOTH the original input IDs AND the input embeddings
        // The input IDs are needed for position calculation even when using embeddings
        let result = languageModel(
            input.text.tokens.squeezed(axis: 0),  // Remove batch dimension and pass original IDs
            cache: cache,
            inputEmbedding: inputEmbeddings,
            positionIds: nil,
            pixelValues: allPixels,
            imageGridThw: input.image?.frames,
            videoGridThw: input.video?.frames
        )

        return .logits(result)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache).logits
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        visionModel.sanitize(
            weights:
                Dictionary(
                    uniqueKeysWithValues: weights.map { key, value in
                        var key = key
                        if !key.contains("vision_tower") {
                            key = key.replacingOccurrences(of: "visual", with: "vision_tower")
                        }
                        if !key.contains("language_model") {
                            key = key.replacingOccurrences(
                                of: "model", with: "language_model.model")
                            key = key.replacingOccurrences(
                                of: "lm_head", with: "language_model.lm_head")
                        }

                        return (key, value)
                    })
        )
    }
}

// MARK: - Configuration

/// Configuration for ``Qwen25VL``
public struct Qwen25VLConfiguration: Codable, Sendable {

    public struct TextConfiguration: Codable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let intermediateSize: Int
        public let attentionHeads: Int
        private let _rmsNormEps: Float?
        public var rmsNormEps: Float { _rmsNormEps ?? 1e-6 }
        public let vocabularySize: Int
        public let kvHeads: Int
        private let _maxPositionEmbeddings: Int?
        public var maxPositionEmbeddings: Int { _maxPositionEmbeddings ?? 128000 }
        private let _ropeTheta: Float?
        public var ropeTheta: Float { _ropeTheta ?? 1_000_000 }
        private let _ropeTraditional: Bool?
        public var ropeTraditional: Bool { _ropeTraditional ?? false }
        public let ropeScaling: [String: StringOrNumber]?
        private let _tieWordEmbeddings: Bool?
        public var tieWordEmbeddings: Bool { _tieWordEmbeddings ?? true }
        private let _slidingWindow: Int?
        public var slidingWindow: Int { _slidingWindow ?? 32768 }
        private let _useSlidingWindow: Bool?
        public var useSlidingWindow: Bool { _useSlidingWindow ?? false }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case attentionHeads = "num_attention_heads"
            case _rmsNormEps = "rms_norm_eps"
            case vocabularySize = "vocab_size"
            case kvHeads = "num_key_value_heads"
            case _maxPositionEmbeddings = "max_position_embeddings"
            case _ropeTheta = "rope_theta"
            case _ropeTraditional = "rope_traditional"
            case ropeScaling = "rope_scaling"
            case _tieWordEmbeddings = "tie_word_embeddings"
            case _slidingWindow = "sliding_window"
            case _useSlidingWindow = "use_sliding_window"
        }
    }

    public struct VisionConfiguration: Codable, Sendable {
        public let depth: Int
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let outHiddenSize: Int
        public let numHeads: Int
        public let patchSize: Int
        private let _inChannels: Int?
        public var inChannels: Int { _inChannels ?? 3 }
        private let _layerNormEps: Float?
        public var layerNormEps: Float { _layerNormEps ?? 1e-6 }
        public let spatialPatchSize: Int
        public let spatialMergeSize: Int
        public let temporalPatchSize: Int
        public let windowSize: Int
        public let fullattBlockIndexes: [Int]
        private let _tokensPerSecond: Int?
        public var tokensPerSecond: Int { _tokensPerSecond ?? 2 }
        private let _mlpRatio: Float?
        public var mlpRatio: Float { _mlpRatio ?? 4.0 }
        private let _imageSize: Int?
        public var imageSize: Int { _imageSize ?? 384 }
        private let _vocabularySize: Int?
        public var vocabularySize: Int { _vocabularySize ?? 32000 }

        enum CodingKeys: String, CodingKey {
            case depth
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case outHiddenSize = "out_hidden_size"
            case numHeads = "num_heads"
            case patchSize = "patch_size"
            case _inChannels = "in_channels"
            case _layerNormEps = "layer_norm_eps"
            case spatialPatchSize = "spatial_patch_size"
            case spatialMergeSize = "spatial_merge_size"
            case temporalPatchSize = "temporal_patch_size"
            case windowSize = "window_size"
            case fullattBlockIndexes = "fullatt_block_indexes"
            case _tokensPerSecond = "tokens_per_second"
            case _mlpRatio = "mlp_ratio"
            case _imageSize = "image_size"
            case _vocabularySize = "vocab_size"
        }
    }

    public struct BaseConfiguration: Codable, Sendable {
        public let modelType: String
        public let vocabularySize: Int
        public let imageTokenId: Int
        public let videoTokenId: Int
        public let visionStartTokenId: Int
        public let visionEndTokenId: Int
        public let visionTokenId: Int
        public let hiddenSize: Int
        public let numAttentionHeads: Int
        public let numHiddenLayers: Int
        public let intermediateSize: Int
        public let numKeyValueHeads: Int
        public let slidingWindow: Int
        public let useSlidingWindow: Bool
        public let maxWindowLayers: Int

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case vocabularySize = "vocab_size"
            case imageTokenId = "image_token_id"
            case videoTokenId = "video_token_id"
            case visionStartTokenId = "vision_start_token_id"
            case visionEndTokenId = "vision_end_token_id"
            case visionTokenId = "vision_token_id"
            case hiddenSize = "hidden_size"
            case numAttentionHeads = "num_attention_heads"
            case numHiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case numKeyValueHeads = "num_key_value_heads"
            case slidingWindow = "sliding_window"
            case useSlidingWindow = "use_sliding_window"
            case maxWindowLayers = "max_window_layers"
        }
    }

    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let baseConfiguration: BaseConfiguration

    enum CodingKeys: String, CodingKey {
        case visionConfiguration = "vision_config"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        // this is a sub-dictionary
        self.visionConfiguration = try container.decode(
            VisionConfiguration.self, forKey: .visionConfiguration)

        // these are overlaid in the top level
        self.textConfiguration = try TextConfiguration(from: decoder)
        self.baseConfiguration = try BaseConfiguration(from: decoder)
    }
}

/// Configuration for ``Qwen25VLProcessor``
public struct Qwen25VLProcessorConfiguration: Codable, Sendable {
    public struct Size: Codable, Sendable {
        public let maxPixels: Int
        public let minPixels: Int

        enum CodingKeys: String, CodingKey {
            case maxPixels = "max_pixels"
            case minPixels = "min_pixels"
        }
    }

    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let minPixels: Int
    public let maxPixels: Int
    public let mergeSize: Int
    public let patchSize: Int
    public let temporalPatchSize: Int
    public let imageProcessorType: String

    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }
    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }

    public var size: Size {
        Size(maxPixels: maxPixels, minPixels: minPixels)
    }

    enum CodingKeys: String, CodingKey {
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case minPixels = "min_pixels"
        case maxPixels = "max_pixels"
        case mergeSize = "merge_size"
        case patchSize = "patch_size"
        case temporalPatchSize = "temporal_patch_size"
        case imageProcessorType = "image_processor_type"
    }
}
