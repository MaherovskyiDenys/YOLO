# ML Project Structure (PyTorch + API + Docker)

This repository follows a clean and scalable structure for training, evaluating, and deploying machine learning models.

---

## 📁 Root Level

```
my_project/
│
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── main.py
```

### Files

* **README.md** – Project documentation.
* **requirements.txt** – Python dependencies.
* **Dockerfile** – Defines container image for API/service.
* **docker-compose.yml** – Simplifies running the API container.
* **main.py** – Main training entry point.

---

## ⚙️ configs/

```
configs/
└── config.json
```

* Stores hyperparameters and paths.
* Keeps code clean (no hardcoded values).

---

## 📊 data/

```
data/
├── raw/
├── processed/
└── splits/
```

* **raw/** – Original downloaded dataset (unchanged).
* **processed/** – Converted dataset (e.g., VOC XML → YOLO txt).
* **splits/** – Train/val/test file lists.

---

## 🧠 src/ (Core ML Code)

```
src/
├── dataset/
├── models/
├── training/
├── inference/
└── utils/
```

### dataset/

* Dataset class
* Transforms
* Dataloader logic

### models/

* Model architectures
* Custom layers / blocks

### training/

* Training loop
* Validation loop
* Loss functions
* Optimizer & scheduler setup

### inference/

* Standalone prediction logic
* Model export (ONNX / TorchScript)

### utils/

* Metrics
* Checkpoint saving/loading
* Visualization
* Seed setup

---

## 🚀 api/ (Model Serving)

```
api/
├── app.py
├── routes/
├── services/
├── schemas/
└── utils/
```

* **app.py** – FastAPI app entry point.
* **routes/** – API endpoints (e.g., `/predict`).
* **services/** – Loads model and runs inference.
* **schemas/** – Request/response models.
* **utils/** – API-specific preprocessing helpers.

---

## 📦 outputs/

```
outputs/
├── checkpoints/
├── logs/
└── runs/
```

* **checkpoints/** – Saved model weights.
* **logs/** – Training logs.
* **runs/** – TensorBoard or experiment tracking files.

---

# 🔁 Typical Workflow

1. Download dataset → `data/raw/`
2. Convert labels → `data/processed/`
3. Train model → `main.py`
4. Save best model → `outputs/checkpoints/`
5. Run API → Docker + FastAPI
6. Serve predictions via `/predict`
