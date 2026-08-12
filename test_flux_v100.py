from __future__ import annotations

import time

import torch
from diffusers import FluxPipeline


REPO_ID = "black-forest-labs/FLUX.1-schnell"
REVISION = (
    "741f7c3ce8b383c54771c7003378a50191e9efe9"
)


def gib(value: int) -> float:
    return value / (1024 ** 3)


def main() -> None:
    assert torch.cuda.is_available()

    print("GPU:", torch.cuda.get_device_name(0))

    print(
        "Native BF16:",
        torch.cuda.is_bf16_supported(
            including_emulation=False
        ),
    )

    print("Loading FLUX...")

    load_start = time.perf_counter()

    pipe = FluxPipeline.from_pretrained(
        REPO_ID,
        revision=REVISION,
        torch_dtype=torch.float16,
        local_files_only=True,
        use_safetensors=True,
    )

    print(
        "Load time:",
        f"{time.perf_counter() - load_start:.2f}s",
    )

    pipe.enable_sequential_cpu_offload()

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    torch.cuda.reset_peak_memory_stats()

    generator = torch.Generator(
        device="cpu"
    ).manual_seed(42)

    print("Starting generation...")

    start = time.perf_counter()

    image = pipe(
        prompt="a photo of Snoopy in a park",
        width=512,
        height=512,
        num_inference_steps=4,
        guidance_scale=0.0,
        max_sequence_length=256,
        generator=generator,
    ).images[0]

    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    image.save(
        "outputs/v100_flux_smoke.png"
    )

    print()
    print("Generation completed.")
    print("Time:", f"{elapsed:.2f}s")

    print(
        "Peak allocated:",
        f"{gib(torch.cuda.max_memory_allocated()):.2f} GiB",
    )

    print(
        "Peak reserved:",
        f"{gib(torch.cuda.max_memory_reserved()):.2f} GiB",
    )


if __name__ == "__main__":
    main()
