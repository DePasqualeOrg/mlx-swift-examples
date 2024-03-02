#  LLMEval

An example that:

- downloads a huggingface model (phi-2) and tokenizer
- evaluates a prompt
- displays the output as it generates text

> Note: this _must_ be built Release, otherwise you will encounter
stack overflows.

You will need to set the Team on the LLMEval target in order to build and
run on iOS.

Some notes about the setup:

- this downloads models from hugging face so LLMEval -> Signing & Capabilities has the "Outgoing Connections (Client)" set in the App Sandbox
- LLM models are large so this uses the Increased Memory Limit entitlement on iOS to allow ... increased memory limits for devices that have more memory
- The Phi2 4 bit model is small enough to run on some iPhone models
    - this can be changed by editing `let modelConfiguration = ModelConfiguration.phi4bit`

### Troubleshooting

If the program crashes with a very deep stack trace you may need to build
in Release configuration.  This seems to depend on the size of the model.

There are a couple options:

- build Release
- force the model evaluation to run on the main thread, e.g. using @MainActor
- build `Cmlx` with optimizations by modifying `mlx/Package.swift` and adding `.unsafeOptions(["-O3"]),` around line 87

Building in Release / optimizations will remove a lot of tail calls in the C++ 
layer.  These lead to the stack overflows.

See discussion here: https://github.com/ml-explore/mlx-swift-examples/issues/3