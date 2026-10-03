import base64
import io
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT_DIR = Path(__file__).resolve().parent
CATALOG_DIR = ROOT_DIR / "catalog"
EMBEDDINGS_PATH = CATALOG_DIR / "embeddings.npy"
FILE_PATHS_PATH = CATALOG_DIR / "file_paths.npy"
WEB_DIR = ROOT_DIR / "frontend"
IMAGE_CDN_BASE_URL = os.getenv("IMAGE_CDN_BASE_URL", "").rstrip("/")
os.environ.setdefault("TORCH_HOME", str(ROOT_DIR / ".torch-cache"))

app = FastAPI(title="StyleSearch API", version="1.0.0")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@lru_cache(maxsize=1)
def load_catalog():
    if not EMBEDDINGS_PATH.exists() or not FILE_PATHS_PATH.exists():
        raise FileNotFoundError(
            "Catalog files are missing. Add catalog/embeddings.npy and "
            "catalog/file_paths.npy before searching."
        )
    embeddings = np.load(EMBEDDINGS_PATH, mmap_mode="r")
    file_paths = np.load(FILE_PATHS_PATH, allow_pickle=True)
    if embeddings.shape[0] != len(file_paths):
        raise ValueError("The embeddings and file paths contain different numbers of items.")
    return embeddings, file_paths


@lru_cache(maxsize=1)
def feature_extractor():
    import torch
    from torchvision import models

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    model.to("cpu")
    return model, weights.transforms()


def resolve_image_path(raw_path) -> Path:
    normalized = str(raw_path).replace("\\", "/")
    candidate = Path(normalized)
    if not candidate.is_absolute():
        candidate = ROOT_DIR / candidate
    if candidate.exists():
        return candidate.resolve()
    dataset_candidate = ROOT_DIR / "myntradataset" / Path(normalized)
    return dataset_candidate.resolve()


def image_url(index: int, raw_path) -> str:
    if not IMAGE_CDN_BASE_URL:
        return f"/api/images/{index}"
    relative_path = Path(str(raw_path).replace("\\", "/"))
    return f"{IMAGE_CDN_BASE_URL}/{relative_path.stem}.jpg"


def image_as_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=88)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def rank_similarities(query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    query_vector = query_embedding.reshape(-1).astype(np.float32)
    query_vector /= np.linalg.norm(query_vector) + 1e-8
    similarities = np.empty(len(embeddings), dtype=np.float32)
    for start in range(0, len(embeddings), 2048):
        end = min(start + 2048, len(embeddings))
        chunk = np.asarray(embeddings[start:end], dtype=np.float32)
        similarities[start:end] = chunk @ query_vector
    return similarities


def get_image(index: int, upscale: bool = False) -> Image.Image:
    try:
        _, file_paths = load_catalog()
        image_path = resolve_image_path(file_paths[index])
    except (IndexError, ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="Product image is unavailable.")
    try:
        image = Image.open(image_path).convert("RGB")
        return image
    except Exception as error:
        raise HTTPException(status_code=500, detail="Could not process this product image.") from error


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
def health():
    catalog_ready = EMBEDDINGS_PATH.exists() and FILE_PATHS_PATH.exists()
    return {
        "status": "ok",
        "catalog_ready": catalog_ready,
        "upscaler_ready": False,
    }


@app.post("/api/search")
async def search(image: UploadFile = File(...), upscale: bool = Query(False)):
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Upload a JPG, PNG, or WebP image.")
    try:
        contents = await image.read()
        query_image = Image.open(io.BytesIO(contents)).convert("RGB")
        if query_image.width > 1600 or query_image.height > 1600:
            query_image.thumbnail((1600, 1600))
        processed_image = query_image
        embeddings, _ = load_catalog()
        model, transform = feature_extractor()
        import torch

        with torch.inference_mode():
            tensor = transform(processed_image).unsqueeze(0)
            query_embedding = model(tensor).squeeze().reshape(1, -1).numpy()
        similarities = rank_similarities(query_embedding, embeddings)
        indices = np.argsort(similarities)[::-1][:5]
        matches = []
        for index in indices:
            path = resolve_image_path(load_catalog()[1][index])
            matches.append({
                "index": int(index),
                "score": round(float(similarities[index]), 4),
                "available": bool(IMAGE_CDN_BASE_URL) or path.is_file(),
                "image_url": image_url(int(index), load_catalog()[1][index]),
                "enhance_url": f"/api/images/{int(index)}?upscale=true",
            })
        return {"query_preview": image_as_data_url(processed_image), "matches": matches}
    except HTTPException:
        raise
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Search failed: {error}") from error


@app.get("/api/images/{index}")
def product_image(index: int, upscale: bool = Query(False)):
    image = get_image(index, upscale)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=88)
    return Response(
        content=buffer.getvalue(),
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.exception_handler(FileNotFoundError)
async def missing_asset_handler(_, error):
    return JSONResponse(status_code=503, content={"detail": str(error)})
