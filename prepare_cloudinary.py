import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import sleep

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def prepare_thumbnails(source_dir: Path, output_dir: Path, size: int, quality: int) -> int:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    count = 0
    for source_path in sorted(source_dir.rglob("*")):
        if source_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative_path = source_path.relative_to(source_dir).with_suffix(".jpg")
        output_path = output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with Image.open(source_path) as image:
                image = image.convert("RGB")
                image.thumbnail((size, size), Image.Resampling.LANCZOS)
                image.save(output_path, "JPEG", quality=quality, optimize=True)
        except (OSError, ValueError):
            continue
        count += 1
    return count


def upload_one_thumbnail(
    thumbnail_path: Path,
    thumbnail_dir: Path,
    retries: int,
) -> None:
    import cloudinary
    import cloudinary.uploader

    public_id = thumbnail_path.relative_to(thumbnail_dir).with_suffix("").as_posix()
    last_error = None
    for attempt in range(retries + 1):
        try:
            cloudinary.uploader.upload(
                str(thumbnail_path),
                public_id=public_id,
                folder="stylesearch",
                overwrite=True,
                resource_type="image",
            )
            return
        except Exception as error:
            last_error = error
            if attempt < retries:
                sleep(2 ** attempt)
    raise RuntimeError(f"{thumbnail_path.name}: {last_error}") from last_error


def upload_thumbnails(
    thumbnail_dir: Path,
    cloud_name: str,
    api_key: str,
    api_secret: str,
    workers: int,
    retries: int,
) -> int:
    import cloudinary

    cloudinary.config(cloud_name=cloud_name, api_key=api_key, api_secret=api_secret, secure=True)
    thumbnail_paths = sorted(thumbnail_dir.rglob("*.jpg"))
    if not thumbnail_paths:
        raise FileNotFoundError(f"No thumbnails found in {thumbnail_dir}")

    uploaded = 0
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(upload_one_thumbnail, path, thumbnail_dir, retries): path
            for path in thumbnail_paths
        }
        for future in as_completed(futures):
            path = futures[future]
            try:
                future.result()
                uploaded += 1
                if uploaded % 100 == 0 or uploaded == len(thumbnail_paths):
                    print(f"Uploaded {uploaded}/{len(thumbnail_paths)} thumbnails")
            except Exception as error:
                failures.append(str(error))

    if failures:
        preview = "; ".join(failures[:5])
        raise RuntimeError(
            f"{len(failures)} uploads failed. Re-run the command to retry them. "
            f"Examples: {preview}"
        )
    return uploaded


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and optionally upload catalog thumbnails.")
    parser.add_argument("--source-dir", type=Path, default=Path("myntradataset/images"))
    parser.add_argument("--thumbnail-dir", type=Path, default=Path(".cloudinary-thumbnails"))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--quality", type=int, default=82)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument(
        "--upload-existing",
        action="store_true",
        help="Upload existing thumbnails without regenerating them.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    if args.workers < 1 or args.retries < 0:
        raise ValueError("--workers must be positive and --retries cannot be negative")
    if args.upload_existing:
        upload = True
    else:
        count = prepare_thumbnails(args.source_dir, args.thumbnail_dir, args.size, args.quality)
        print(f"Prepared {count} thumbnails in {args.thumbnail_dir}")
        upload = args.upload
    if not upload:
        return
    required = ("CLOUDINARY_CLOUD_NAME", "CLOUDINARY_API_KEY", "CLOUDINARY_API_SECRET")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")
    uploaded = upload_thumbnails(
        args.thumbnail_dir,
        os.environ["CLOUDINARY_CLOUD_NAME"],
        os.environ["CLOUDINARY_API_KEY"],
        os.environ["CLOUDINARY_API_SECRET"],
        args.workers,
        args.retries,
    )
    print(f"Uploaded {uploaded} thumbnails to Cloudinary")


if __name__ == "__main__":
    main()
