"""
Debug script to trace CPU/GPU consistency issues in convolutional model.
"""

import torch
from testing_utils import generate_uniform_input

from difflut.models import ModelConfig, SimpleConvolutional


def main():
    print("=" * 60)
    print("Debugging CPU/GPU Consistency")
    print("=" * 60)

    # Create config
    config = ModelConfig(
        model_type="convolutional",
        layer_type="random",
        node_type="probabilistic",
        encoder_config={"name": "thermometer", "num_bits": 3},
        node_input_dim=6,
        layer_widths=[4],
        num_classes=10,
        dataset="test",
        input_size=None,
        runtime={
            "conv_kernel_size": 3,
            "conv_stride": 1,
            "conv_padding": 1,
            "input_channels": 1,
            "input_height": 8,
            "input_width": 8,
            "conv_layer_widths": [4],
        },
    )

    # Create models with SAME seed
    print("\n1. Creating models...")
    torch.manual_seed(42)
    model_cpu = SimpleConvolutional(config)
    print(f"   CPU model created, random state: {torch.initial_seed()}")

    torch.manual_seed(42)
    model_gpu = SimpleConvolutional(config).cuda()
    print(f"   GPU model created, random state: {torch.initial_seed()}")

    # Fit encoders with SAME data
    print("\n2. Fitting encoders...")
    data_cpu = generate_uniform_input((4, 1, 8, 8), seed=42)
    data_gpu = data_cpu.cuda()

    print(
        f"   Data CPU: min={data_cpu.min():.4f}, max={data_cpu.max():.4f}, mean={data_cpu.mean():.4f}"
    )
    print(
        f"   Data GPU: min={data_gpu.min():.4f}, max={data_gpu.max():.4f}, mean={data_gpu.mean():.4f}"
    )
    print(f"   Data diff: {(data_cpu - data_gpu.cpu()).abs().max().item():.2e}")

    model_cpu.fit_encoder(data_cpu)
    model_gpu.fit_encoder(data_gpu)

    # Compare encoder parameters
    print("\n3. Checking encoder parameters...")
    encoder_cpu_params = list(model_cpu.encoder.parameters())
    encoder_gpu_params = list(model_gpu.encoder.parameters())
    if len(encoder_cpu_params) > 0:
        print(f"   Encoder has {len(encoder_cpu_params)} parameters")
        for i, (p_cpu, p_gpu) in enumerate(zip(encoder_cpu_params, encoder_gpu_params)):
            diff = (p_cpu - p_gpu.cpu()).abs().max().item()
            print(f"   Param {i}: max_diff = {diff}")
    else:
        print("   Encoder has no trainable parameters (thermometer is non-parametric)")

    # Compare layer weights
    print("\n4. Checking convolutional layer weights...")
    for layer_idx, (layer_cpu, layer_gpu) in enumerate(
        zip(model_cpu.conv_layers, model_gpu.conv_layers)
    ):
        print(f"\n   Layer {layer_idx}:")
        print(f"   CPU trees: {len(layer_cpu.trees)}, GPU trees: {len(layer_gpu.trees)}")

        for tree_idx in range(min(2, len(layer_cpu.trees))):  # Check first 2 trees
            tree_cpu = layer_cpu.trees[tree_idx]
            tree_gpu = layer_gpu.trees[tree_idx]

            print(f"   Tree {tree_idx}: {len(tree_cpu)} sublayers")
            for sublayer_idx, (sublayer_cpu, sublayer_gpu) in enumerate(zip(tree_cpu, tree_gpu)):
                # Check weights
                params_cpu = list(sublayer_cpu.parameters())
                params_gpu = list(sublayer_gpu.parameters())

                if len(params_cpu) > 0:
                    max_diff = max(
                        (p_cpu - p_gpu.cpu()).abs().max().item()
                        for p_cpu, p_gpu in zip(params_cpu, params_gpu)
                    )
                    print(
                        f"     Sublayer {sublayer_idx}: {len(params_cpu)} params, max_diff = {max_diff:.2e}"
                    )
                else:
                    print(f"     Sublayer {sublayer_idx}: No parameters (random layer)")

    # Test forward pass step by step
    print("\n5. Testing forward pass step by step...")
    model_cpu.eval()
    model_gpu.eval()

    input_cpu = generate_uniform_input((2, 1, 8, 8), seed=123)
    input_gpu = input_cpu.cuda()

    print(f"   Input shape: {input_cpu.shape}")
    print(f"   Input max_diff: {(input_cpu - input_gpu.cpu()).abs().max().item():.2e}")

    with torch.no_grad():
        # Step through encoding
        print("\n   a) Encoding...")
        encoded_cpu = model_cpu.encoder(input_cpu)
        encoded_gpu = model_gpu.encoder(input_gpu)
        print(f"      Encoded shape CPU: {encoded_cpu.shape}")
        print(f"      Encoded shape GPU: {encoded_gpu.shape}")
        print(f"      Encoded max_diff: {(encoded_cpu - encoded_gpu.cpu()).abs().max().item():.2e}")

        # Step through convolutional layers
        print("\n   b) Convolutional layers...")
        x_cpu = encoded_cpu
        x_gpu = encoded_gpu

        for layer_idx, (layer_cpu, layer_gpu) in enumerate(
            zip(model_cpu.conv_layers, model_gpu.conv_layers)
        ):
            x_cpu = layer_cpu(x_cpu)
            x_gpu = layer_gpu(x_gpu)
            diff = (x_cpu - x_gpu.cpu()).abs().max().item()
            print(f"      Layer {layer_idx} output: shape={x_cpu.shape}, max_diff={diff:.6f}")

            # Check if this layer introduces divergence
            if diff > 0.01:
                print(f"      ⚠️  DIVERGENCE DETECTED IN LAYER {layer_idx}!")
                # First 2 channels
                print(f"         CPU output:\n{x_cpu[:, :2]}")
                print(f"         GPU output:\n{x_gpu.cpu()[:, :2]}")

        # Full forward pass
        print("\n   c) Full forward pass...")
        output_cpu = model_cpu(input_cpu)
        output_gpu = model_gpu(input_gpu).cpu()

    print(f"\n6. Final output comparison:")
    print(f"   CPU output shape: {output_cpu.shape}")
    print(f"   GPU output shape: {output_gpu.shape}")
    print(f"   Max absolute difference: {(output_cpu - output_gpu).abs().max().item():.6f}")
    print(f"   Mean absolute difference: {(output_cpu - output_gpu).abs().mean().item():.6f}")

    # Check specific values
    print(f"\n7. Detailed comparison:")
    diff = output_cpu - output_gpu
    large_diffs = []
    for i in range(output_cpu.shape[0]):
        for j in range(output_cpu.shape[1]):
            if abs(diff[i, j]) > 0.01:
                large_diffs.append(
                    (i, j, output_cpu[i, j].item(), output_gpu[i, j].item(), diff[i, j].item())
                )

    if large_diffs:
        print(f"   Found {len(large_diffs)} outputs with diff > 0.01:")
        for i, j, cpu_val, gpu_val, d in large_diffs[:10]:  # Show first 10
            print(f"   [{i},{j}]: CPU={cpu_val:.6f}, GPU={gpu_val:.6f}, diff={d:.6f}")
    else:
        print("   All outputs agree within 0.01!")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
