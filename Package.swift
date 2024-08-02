// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "mlx-libraries",
    platforms: [.macOS(.v14), .iOS(.v16)],
    products: [
        .library(
            name: "LLM",
            targets: ["MLXLLM"]),
    ],
    dependencies: [
      .package(url: "https://github.com/ml-explore/mlx-swift", .branch("main")),
        .package(url: "https://github.com/huggingface/swift-transformers", .branch("main")),
    ],
    targets: [
        .target(
            name: "MLXLLM",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "Libraries/LLM",
            exclude: [
                "README.md",
                "LLM.h",
            ]
        ),
    ]
)
