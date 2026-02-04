# Copyright (c) 2026 Dmitry Feklin
# Apache License 2.0

import torch
import os
from collections import defaultdict
from diffusers import StableDiffusionPipeline
from GABE import GABE

def group_conv_weights_with_modules(model: torch.nn.Module):
    """
    Groups Conv2d weights and modules by weight shape.
    """
    groups = defaultdict(lambda: {'modules': []})
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            # Use the original shape of the scale as a key
            key = tuple(module.weight.shape)
            groups[key]['modules'].append(module)
    return {k: v for k, v in groups.items() if len(v['modules']) > 1}

def break_denoising_process_test():
    """
    A test that systematically varies GABE components in UNet to study their impact on noise reduction.
    """
    # --- 1. SETUP ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    base_dir = "unet_break_denoising"
    os.makedirs(base_dir, exist_ok=True)

    prompt = (
        "a cinematic ultra-detailed fantasy castle on a cliff at sunset, "
        "dramatic lighting, 8k, masterpiece"
    )

    # --- 2. LOADING THE MODEL ---
    print("Loading Stable Diffusion v1.5...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)

    # TARGET OF CHANGES - UNet
    target_model = pipe.unet
    target_model.eval()

    # --- 3. PREPARATION FOR MODIFICATION ---
    print(f"Group layers into {type(target_model).__name__}...")
    groups = group_conv_weights_with_modules(target_model)
    print(f"Found {len(groups)} Conv2d groups to modify.")
    
    compressor = GABE()

    # Create a backup of the original scales
    backup = {}
    for group in groups.values():
        for mod in group["modules"]:
            backup[id(mod)] = mod.weight.detach().clone()

    scales = [0.0, 0.15, 0.35, 0.6, 1.0, 1.5]
    modes = ["w_bar", "basis", "coeffs", "affine"]

    BASE_SEED = 1234 # Fixed seed for reproducibility

    # --- 4. MAIN CYCLE OF THE EXPERIMENT ---
    for mode in modes:
        print(f"\n=== UNET DISTORTION MODE: {mode.upper()} ===")
        mode_dir = os.path.join(base_dir, mode)
        os.makedirs(mode_dir, exist_ok=True)

        for scale in scales:

            # Fully restore the weights to their original state before each iteration
            for group in groups.values():
                for mod in group["modules"]:
                    mod.weight.data.copy_(backup[id(mod)])

            if scale > 0.0:
                total_norm_change = 0.0

                # Apply modifications to each group
                for group in groups.values():
                    modules = group["modules"]
                    # Collecting tensors for GABE (it is important to do this after recovery)
                    tensors = [
                        mod.weight.detach().clone().view(mod.weight.shape[0], -1)
                        for mod in modules
                    ]

                    # Compress the current group
                    compressed = compressor.compress(
                        tensors,
                        basis_rank=1,
                        w_bar_rank=16
                    )

                    # Modifying GABE components depending on the mode
                    if mode in ["w_bar", "affine"]:
                        w_bar = compressor._decompress_matrix(
                            compressed["w_bar_formulas"],
                            compressed["w_bar_residuals"]
                        )
                        # Add noise to the remainder w_bar
                        noise = torch.randn_like(w_bar) * scale * 0.1 * w_bar.std()
                        compressed["w_bar_residuals"] += noise.to(compressed["w_bar_residuals"].dtype)

                    if mode in ["basis", "affine"] and compressed["basis_residuals"].numel() > 0:
                        B = compressor._decompress_matrix(
                            compressed["basis_formulas"],
                            compressed["basis_residuals"]
                        )
                        # Adding noise to the remainder of the basis
                        noise = torch.randn_like(B) * scale * 0.1 * B.std()
                        compressed["basis_residuals"] += noise.to(compressed["basis_residuals"].dtype)

                    if mode in ["coeffs", "affine"]:
                        c = compressed["coeffs"]
                        # Adding noise to the coefficients
                        noise = torch.randn_like(c) * scale * c.std()
                        compressed["coeffs"] = c + noise

                    # Recovering weights from modified components
                    reconstructed = compressor.decompress(compressed)

                    # Write the modified weights back into the model
                    for mod, rec in zip(modules, reconstructed):
                        new_w = rec.view(mod.weight.shape)
                        total_norm_change += torch.norm(
                            new_w - backup[id(mod)]
                        ).item()
                        mod.weight.data.copy_(new_w)

                print(f"  scale={scale:.2f} | Total change in weights ΔW={total_norm_change:.2f}")

            # generator with the same seed for identical initial noise
            generator = torch.Generator(device=device).manual_seed(BASE_SEED)

            with torch.no_grad():
                image = pipe(
                    prompt,
                    generator=generator,
                    num_inference_steps=30,
                    guidance_scale=7.5
                ).images[0]

            out_path = os.path.join(mode_dir, f"scale_{scale:.2f}.png")
            image.save(out_path)
            print(f"  → Saved in {out_path}")

    print("\nThe results are in the 'unet_break_denoising' folder..")


if __name__ == "__main__":
    break_denoising_process_test()
