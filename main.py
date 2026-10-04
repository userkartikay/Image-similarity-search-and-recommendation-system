import base64
import gc
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

ROOT_DIR = Path(__file__).resolve().parent
CATALOG_DIR = ROOT_DIR / "catalog"
EMBEDDINGS_PATH = CATALOG_DIR / "embeddings.npy"
FILE_PATHS_PATH = CATALOG_DIR / "file_paths.npy"
MODEL_PATH = ROOT_DIR / "models" / "mobilenet_v3_small.onnx"
WEB_DIR = ROOT_DIR / "frontend"
IMAGE_CDN_BASE_URL = os.getenv("IMAGE_CDN_BASE_URL", "").rstrip("/")

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_SOURCE_PIXELS = 60_000_000
TOP_K = 5

# Matches torchvision MobileNet_V3_Small_Weights.DEFAULT.transforms():
# resize shorter side to 256 (bilinear), center crop 224, ImageNet normalization.
RESIZE_SIZE = 256
CROP_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

Image.MAX_IMAGE_PIXELS = MAX_SOURCE_PIXELS

state: dict = {}


def load_catalog():
    if not EMBEDDINGS_PATH.exists() or not FILE_PATHS_PATH.exists():
        raise FileNotFoundError(
            "Catalog files are missing. Add catalog/embeddings.npy and "
            "catalog/file_paths.npy before searching."
        )
    # Fully loaded (not mmap) so the first search does not page the file in from disk.
    embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    file_paths = np.load(FILE_PATHS_PATH, allow_pickle=True)
    if embeddings.shape[0] != len(file_paths):
        raise ValueError("The embeddings and file paths contain different numbers of items.")
    return embeddings, file_paths


def load_model():
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.enable_cpu_mem_arena = False
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(MODEL_PATH), options, providers=["CPUExecutionProvider"])


@asynccontextmanager
async def lifespan(_app: FastAPI):
    state["embeddings"], state["file_paths"] = load_catalog()
    state["session"] = load_model()
    # Warm-up run so the first real request does not pay graph initialization cost.
    state["session"].run(None, {"input": np.zeros((1, 3, CROP_SIZE, CROP_SIZE), np.float32)})
    gc.collect()
    yield
    state.clear()


app = FastAPI(title="StyleSearch API", version="2.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


def open_downscaled(data: bytes, target: int = RESIZE_SIZE) -> Image.Image:
    image = Image.open(io.BytesIO(data))
    # For JPEGs, decode directly at a reduced scale (1/2, 1/4, 1/8) instead of full resolution.
    # A 12 MP photo then never materializes as a ~36 MB RGB buffer.
    image.draft("RGB", (target * 2, target * 2))
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image.thumbnail((target * 2, target * 2), Image.Resampling.BILINEAR, reducing_gap=2.0)
    return image


def preprocess(image: Image.Image) -> np.ndarray:
    width, height = image.size
    scale = RESIZE_SIZE / min(width, height)
    resized = image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.BILINEAR,
    )
    left = (resized.width - CROP_SIZE) // 2
    top = (resized.height - CROP_SIZE) // 2
    cropped = resized.crop((left, top, left + CROP_SIZE, top + CROP_SIZE))
    array = np.asarray(cropped, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return ((array - MEAN) / STD)[np.newaxis].astype(np.float32)


def embed(image: Image.Image) -> np.ndarray:
    output = state["session"].run(None, {"input": preprocess(image)})[0].reshape(-1)
    return output / (np.linalg.norm(output) + 1e-8)


def top_matches(query: np.ndarray, k: int = TOP_K):
    similarities = state["embeddings"] @ query
    candidates = np.argpartition(-similarities, k)[:k]
    ordered = candidates[np.argsort(-similarities[candidates])]
    return ordered, similarities


def resolve_image_path(raw_path) -> Path:
    normalized = str(raw_path).replace("\\", "/")
    candidate = Path(normalized)
    if not candidate.is_absolute():
        candidate = ROOT_DIR / candidate
    if candidate.exists():
        return candidate.resolve()
    return (ROOT_DIR / "myntradataset" / Path(normalized)).resolve()


def image_url(index: int, raw_path) -> str:
    if not IMAGE_CDN_BASE_URL:
        return f"/api/images/{index}"
    return f"{IMAGE_CDN_BASE_URL}/{Path(str(raw_path).replace(chr(92), '/')).stem}.jpg"


def image_as_data_url(image: Image.Image) -> str:
    preview = image.copy()
    preview.thumbnail((384, 384))
    buffer = io.BytesIO()
    preview.save(buffer, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def run_search(contents: bytes):
    query_image = open_downscaled(contents)
    del contents
    indices, similarities = top_matches(embed(query_image))
    file_paths = state["file_paths"]
    matches = []
    for index in indices:
        raw_path = file_paths[index]
        matches.append({
            "index": int(index),
            "score": round(float(similarities[index]), 4),
            "available": bool(IMAGE_CDN_BASE_URL) or resolve_image_path(raw_path).is_file(),
            "image_url": image_url(int(index), raw_path),
            "enhance_url": f"/api/images/{int(index)}",
        })
    return {"query_preview": image_as_data_url(query_image), "matches": matches}


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "catalog_ready": "embeddings" in state,
        "model_ready": "session" in state,
        "upscaler_ready": False,
    }


@app.post("/api/search")
async def search(image: UploadFile = File(...), upscale: bool = Query(False)):
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Upload a JPG, PNG, or WebP image.")
    contents = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 15 MB.")
    try:
        return await run_in_threadpool(run_search, contents)
    except Image.DecompressionBombError as error:
        raise HTTPException(status_code=413, detail="Image resolution is too large.") from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Search failed: {error}") from error
    finally:
        gc.collect()


@app.get("/api/images/{index}")
def product_image(index: int, upscale: bool = Query(False)):
    try:
        image_path = resolve_image_path(state["file_paths"][index])
    except (IndexError, KeyError) as error:
        raise HTTPException(status_code=404, detail="Product not found.") from error
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="Product image is unavailable.")
    return FileResponse(image_path, headers={"Cache-Control": "public, max-age=86400"})


@app.exception_handler(FileNotFoundError)
async def missing_asset_handler(_, error):
    return JSONResponse(status_code=503, content={"detail": str(error)})
