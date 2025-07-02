//
//  Gemma.swift
//  mlx-swift-examples
//
//  Created by Anthony DePasquale on 17.03.2025.
//

import Foundation
import MLX
import MLXFast
import MLXNN

public enum Gemma {
    /// Specialized RMSNorm (Root Mean Square Layer Normalization) for Gemma models
    /// 
    /// RMSNorm is a simplified version of LayerNorm that:
    /// - Only normalizes by the root mean square (no centering around mean)
    /// - Uses fewer parameters and computations than standard LayerNorm
    /// - Provides similar normalization benefits with better efficiency
    /// 
    /// Formula: RMSNorm(x) = x / sqrt(mean(x²) + eps) * (1 + weight)
    /// where weight is a learnable scaling parameter per feature dimension
    public class RMSNorm: Module, UnaryLayer {
        
        // MARK: - Properties
        
        /// Learnable scaling parameters - one per feature dimension
        /// - Initialized to ones, but gets updated during training via backpropagation
        /// - Even though declared as `let`, the underlying MLXArray data can be modified by MLX
        /// - Each dimension can learn its own optimal scaling factor
        /// - Formula uses (1.0 + weight): with weight starting as ones, initial scaling is 2.0
        /// - This appears to be intentional for Gemma models, though unusual for RMSNorm
        let weight: MLXArray
        
        /// Small constant added to denominator for numerical stability
        /// - Prevents division by zero when all elements in a feature are near zero
        /// - Default 1e-5 is a good balance between stability and precision
        /// - Same epsilon concept used in other normalization techniques
        let eps: Float

        // MARK: - Initialization
        
        /// Creates a new RMSNorm layer
        /// - Parameter dimensions: The number of features to normalize (typically hidden_size)
        /// - Parameter eps: Small constant for numerical stability during normalization
        public init(dimensions: Int, eps: Float = 1e-5) {
            // Initialize weight parameters to ones for each feature dimension
            // Shape: [dimensions] - one scaling factor per feature
            // Combined with (1.0 + weight) formula, this gives initial scaling of 2.0
            // This is the standard pattern for Gemma models in MLX
            self.weight = MLXArray.ones([dimensions])
            
            // Store epsilon for use in normalization computation
            self.eps = eps
            
            // Call parent Module initializer to register this as a trainable module
            // This ensures MLX framework discovers and manages the weight parameters
            super.init()
        }

        // MARK: - Forward Pass
        
        /// Applies RMS normalization to the input
        /// - Parameter x: Input tensor of shape [..., dimensions] where last dimension matches init dimensions
        /// - Returns: Normalized tensor of same shape as input
        /// 
        /// The normalization process:
        /// 1. Compute RMS: sqrt(mean(x²) + eps) for numerical stability
        /// 2. Normalize: x / RMS to center the variance 
        /// 3. Scale: apply learnable weights (1.0 + weight) for adaptive scaling
        /// 4. Result: Each feature dimension normalized with learned scaling (initially 2.0)
        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            // Use MLXFast.rmsNorm for optimized computation
            // This efficiently computes: x / sqrt(mean(x²) + eps) * scale_weights
            // 
            // Key points:
            // - weight: learnable parameters for adaptive scaling per dimension (initialized to 1.0)
            // - (1.0 + weight): creates scaling factors (initially 2.0, but learned during training)
            // - eps: prevents numerical instability from division by tiny values
            // - MLXFast: uses optimized kernels for better performance than manual computation
            return MLXFast.rmsNorm(x, weight: 1.0 + self.weight, eps: self.eps)
        }
    }

    /// Clips residual connections to prevent overflow in float16 operations
    /// 
    /// This is crucial for mixed-precision training where float16 has limited range.
    /// IEEE 754 half-precision can only represent values up to ±65504.
    /// Without clipping, residual connections can cause numerical overflow.
    static public func clipResidual(_ x: MLXArray, _ y: MLXArray) -> MLXArray {
        // For float32 and higher precision, no clipping needed
        if x.dtype != .float16 {
            return x + y
        }
        
        // IEEE 754 half-precision maximum finite value
        let bound: Float = 65504.0  // Float16 maximum finite value
        
        // Convert to float32 for safe arithmetic
        let xFloat32 = x.asType(.float32)
        let yFloat32 = y.asType(.float32)
        let result = xFloat32 + yFloat32
        
        // Clip to valid float16 range and convert back
        // This prevents overflow while preserving gradient flow
        return clip(result, min: MLXArray(-bound), max: MLXArray(bound)).asType(.float16)
    }
}
