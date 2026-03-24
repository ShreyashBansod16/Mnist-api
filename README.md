# Production-Ready TensorFlow Digit Recognition

This project trains a TensorFlow CNN model using your local MNIST IDX files and serves predictions through a FastAPI backend with a browser canvas UI.

## What You Get

- MNIST IDX loader for your existing dataset files.
- Production-ready CNN training pipeline with:
  - checkpointing,
  - early stopping,
  - LR scheduling,
  - SavedModel and TFLite export.
- Inference API (`/predict`) using FastAPI.
- Canvas-based front-end for drawing digits on a laptop and predicting instantly.

## 1) Install Dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2) Train Model

Run from this workspace root:

```powershell
python train.py --data-root . --output-dir model_artifacts --epochs 18 --batch-size 128
```

Expected outputs in `model_artifacts/`:

- `best_model.keras`
- `digit_cnn.keras`
- `saved_model/`
- `digit_cnn.tflite`
- `metadata.json`

## 3) Start API + Canvas App

```powershell
uvicorn serve:app --host 0.0.0.0 --port 8000
```

Open in browser:

- http://localhost:8000

## API Endpoints

- `GET /health` -> service status
- `POST /predict` -> upload an image file (`multipart/form-data`, field name: `file`)

Example curl:

```bash
curl -X POST "http://localhost:8000/predict" -F "file=@digit.png"
```

## Production Notes

- Use `digit_cnn.keras` or `saved_model/` for server inference.
- Use `digit_cnn.tflite` for lightweight edge/mobile deployment.
- Behind reverse proxy (Nginx/Apache), run with multiple workers:

```powershell
uvicorn serve:app --host 0.0.0.0 --port 8000 --workers 2
```

- Keep input preprocessing unchanged between training and inference for stable accuracy.
