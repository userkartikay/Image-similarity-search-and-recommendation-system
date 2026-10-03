import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import models


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def build_catalog(image_dir: Path, output_dir: Path, batch_size: int, device_name: str) -> None:
    image_dir = image_dir.resolve()
    output_dir = output_dir.resolve()
    image_paths = sorted(
        path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in {image_dir}")

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch installation has no CUDA support.")

    torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))
    torch.set_num_interop_threads(1)
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    transform = weights.transforms()
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier = torch.nn.Identity()
    model = model.to(device).eval()

    embeddings = []
    valid_paths = []
    for start in range(0, len(image_paths), batch_size):
        batch_images = []
        batch_paths = []
        for image_path in image_paths[start:start + batch_size]:
            try:
                with Image.open(image_path) as image:
                    batch_images.append(transform(image.convert("RGB")))
                batch_paths.append(image_path)
            except (OSError, ValueError):
                continue

        if not batch_images:
            continue

        batch_tensor = torch.stack(batch_images).to(device)
        with torch.inference_mode():
            batch_embeddings = model(batch_tensor).flatten(1)
            batch_embeddings = torch.nn.functional.normalize(batch_embeddings, dim=1)
            batch_embeddings = batch_embeddings.cpu().numpy()
        embeddings.append(batch_embeddings)
        valid_paths.extend(batch_paths)
        print(f"Processed {min(start + batch_size, len(image_paths))}/{len(image_paths)} images")

    if not embeddings:
        raise RuntimeError("No readable images were available for embedding.")

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "embeddings.npy", np.concatenate(embeddings).astype(np.float16))
    relative_paths = [path.relative_to(Path.cwd()).as_posix() for path in valid_paths]
    np.save(output_dir / "file_paths.npy", np.array(relative_paths))
    print(f"Saved {len(relative_paths)} aligned image embeddings to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the StyleSearch image catalog.")
    parser.add_argument("--image-dir", type=Path, default=Path("myntradataset/images"))
    parser.add_argument("--output-dir", type=Path, default=Path("catalog"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Embedding device. 'auto' uses CUDA when available.",
    )
    args = parser.parse_args()
    build_catalog(args.image_dir, args.output_dir, args.batch_size, args.device)


if __name__ == "__main__":
    main()
