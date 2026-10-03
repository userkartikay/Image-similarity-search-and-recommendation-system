# StyleSearch

StyleSearch is a visual product discovery app. It extracts a ResNet50 embedding from an uploaded image, compares it with the catalog embeddings, and returns the five closest products. Real-ESRGAN enhancement is available for the query and individual results.

## Run locally

Create a virtual environment, install the dependencies, then start the API:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000` in a browser.

The catalog files must be available at `Real-ESRGAN/embeddings.npy` and `Real-ESRGAN/file_paths.npy`. The optional ESRGAN checkpoint belongs at `Real-ESRGAN/weights/RealESRGAN_x4plus.pth`. Large binary assets should be stored with Git LFS or downloaded from object storage during the Render build; they should not be committed as ordinary Git files.

## Push to GitHub

The repository uses Git LFS for the large catalog and model files. GitHub rejects ordinary files larger than 100 MB, so install Git LFS before the first commit:

```powershell
git init
git lfs install
git add .gitattributes
git add .
git add -f Real-ESRGAN/weights/RealESRGAN_x4plus.pth
git commit -m "Initial StyleSearch application"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

The `-f` flag is required because the vendored Real-ESRGAN project ignores its `weights` directory. GitHub LFS storage and bandwidth quotas depend on your account; if the LFS quota is too small, keep the source repository on GitHub and download these three assets from object storage during the Render build instead.

## Rebuild the catalog (optional)

The current catalog paths match the source dataset order and can be used as-is. Rebuild the paired files only if you replace the dataset or regenerate embeddings with a different model:

```powershell
python rebuild_catalog.py --batch-size 32 --device auto
```

This uses the same ResNet18 weights and preprocessing as the API and writes normalized, aligned files to `Real-ESRGAN/`. With a CUDA-enabled PyTorch installation, `--device auto` uses the GPU; otherwise it falls back to CPU. The catalog must be rebuilt after changing the encoder.

For the local RTX 2050 environment, install the CUDA build of PyTorch before rebuilding:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe rebuild_catalog.py --batch-size 64 --device cuda
```

The Render deployment intentionally keeps the normal CPU-compatible requirements; Render's free instances do not provide a GPU.

## Deploy on Render

This repository includes `render.yaml`. Create a new Blueprint from the repository, or configure a Python web service with:

- Build command: `pip install --upgrade pip && pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Health check path: `/health`

Render instances are CPU-based by default. The Render build command downloads the ResNet18 weights into `.torch-cache`, so the first search does not wait for a model download. ESRGAN enhancement is significantly slower than normal search. For a public deployment, use a persistent asset store or a build step to provide the catalog arrays and ESRGAN checkpoint.

For Render's free instance, keep `ENABLE_UPSCALING=false` as configured in `render.yaml`. Search uses a memory-mapped catalog and does not load the ESRGAN model. Set `ENABLE_UPSCALING=true` only on a machine with more memory and CPU capacity.

To enable enhancement locally, install the optional dependencies with `pip install -r requirements-enhancement.txt` and set `ENABLE_UPSCALING=true` before starting the server.

## API surface

- `GET /health` reports whether the catalog and ESRGAN assets are present.
- `POST /api/search?upscale=false` accepts an `image` multipart upload and returns ranked matches.
- `GET /api/images/{index}?upscale=false` serves a catalog image.
