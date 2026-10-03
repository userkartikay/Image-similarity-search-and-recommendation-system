# StyleSearch

StyleSearch is a visual product discovery app. It extracts a MobileNetV3-Small embedding from an uploaded image, compares it with the catalog embeddings, and returns the five closest products. Product thumbnails are served from Cloudinary when configured.

## Run locally

Create a virtual environment, install the dependencies, then start the API:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000` in a browser.

The catalog files are stored at `catalog/embeddings.npy` and `catalog/file_paths.npy`. Embeddings are normalized and stored as `float16` to reduce Render memory usage; they are converted in small chunks during search. Product thumbnails are stored in Cloudinary; the original image dataset is needed only locally when rebuilding the catalog.

## Push to GitHub

The repository uses Git LFS for the large catalog and model files. GitHub rejects ordinary files larger than 100 MB, so install Git LFS before the first commit:

```powershell
git init
git lfs install
git add .gitattributes
git add .
git commit -m "Initial StyleSearch application"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

The catalog arrays use Git LFS. Keep the source image dataset and enhancement models outside the deployment repository.

## Rebuild the catalog (optional)

The current catalog paths match the source dataset order and can be used as-is. Rebuild the paired files only if you replace the dataset or regenerate embeddings with a different model:

```powershell
python rebuild_catalog.py --batch-size 32 --device auto
```

This uses the same MobileNetV3-Small weights and preprocessing as the API and writes normalized, aligned files to `catalog/`. With a CUDA-enabled PyTorch installation, `--device auto` uses the GPU; otherwise it falls back to CPU. The catalog must be rebuilt after changing the encoder.

For the local RTX 2050 environment, install the CUDA build of PyTorch before rebuilding:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe rebuild_catalog.py --batch-size 64 --device cuda
```

The Render deployment intentionally keeps the normal CPU-compatible requirements; Render's free instances do not provide a GPU.

### Optional Cloudinary thumbnails

For a smaller Render deployment, prepare 512px thumbnails locally and upload them to Cloudinary:

```powershell
pip install -r requirements-storage.txt
python prepare_cloudinary.py
$env:CLOUDINARY_CLOUD_NAME = "your-cloud-name"
$env:CLOUDINARY_API_KEY = "your-api-key"
$env:CLOUDINARY_API_SECRET = "your-api-secret"
python prepare_cloudinary.py --upload
```

For an interrupted upload, reuse existing thumbnails with parallel workers:

```powershell
python prepare_cloudinary.py --thumbnail-dir .cloudinary-thumbnails --upload-existing --workers 8
```

Set `IMAGE_CDN_BASE_URL` on Render to the Cloudinary delivery prefix, for example:
`https://res.cloudinary.com/your-cloud-name/image/upload/f_auto,q_auto,w_512/stylesearch`.
The API then serves thumbnails directly from Cloudinary while enhancement continues through the local API endpoint. Keep the original dataset locally for catalog regeneration; do not commit `.cloudinary-thumbnails/`.

## Deploy on Render

This repository includes `render.yaml`. Create a new Blueprint from the repository, or configure a Python web service with:

- Build command: `pip install --upgrade pip && pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Health check path: `/health`

Render instances are CPU-based by default. The Render build command downloads the MobileNetV3-Small weights into `.torch-cache`, so the first search does not wait for a model download. ESRGAN enhancement is significantly slower than normal search. For a public deployment, use a persistent asset store or a build step to provide the catalog arrays and ESRGAN checkpoint.

Image enhancement is intentionally not included in the deployment. This keeps the Render service small and avoids the ESRGAN model and dependencies.

## API surface

- `GET /health` reports whether the catalog and ESRGAN assets are present.
- `POST /api/search` accepts an `image` multipart upload and returns ranked matches.
- `GET /api/images/{index}` serves a catalog image when local source images are available.
