# ML Model Container - Setup & Usage

## Quick Start

### 1. Build the Docker Image
The Docker image bundles your YOLO model (checkpoint_best_total.pth) with an inference API.

```bash
docker build -t ml-model-inference:latest .
```

The build may take 10-15 minutes on first run due to PyTorch installation (532MB).

### 2. Run with Docker Compose

```bash
docker compose up --pull always
```

This starts the API on `http://localhost:5000`

### 3. Test the API

#### Health Check
```bash
curl http://localhost:5000/health
```

#### Single Image Inference
```bash
curl -X POST -F "image=@path/to/image.jpg" http://localhost:5000/predict
```

#### Batch Inference
```bash
curl -X POST \
  -F "images=@image1.jpg" \
  -F "images=@image2.jpg" \
  -F "images=@image3.jpg" \
  http://localhost:5000/batch
```

#### Model Info
```bash
curl http://localhost:5000/info
```

---

## API Endpoints

### `GET /health`
Health check endpoint. Returns model status and configuration.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "/app/models/ankesh_sir model/checkpoint_best_total.pth"
}
```

### `POST /predict`
Run inference on a single image.

**Request:**
- File upload: `image` (multipart/form-data)

**Response:**
```json
{
  "success": true,
  "filename": "image.jpg",
  "num_detections": 5,
  "detections": [
    {
      "class_id": 0,
      "class_name": "Artwork",
      "confidence": 0.95,
      "bbox_normalized": {
        "x_center": 0.5, 
        "y_center": 0.4,
        "width": 0.3,
        "height": 0.4
      },
      "bbox_pixel": {
        "x1": 100,
        "y1": 80,
        "x2": 200,
        "y2": 160
      }
    }
  ],
  "config": {...}
}
```

### `POST /batch`
Run inference on multiple images.

**Request:**
- File upload: `images` (multiple files, multipart/form-data)

**Response:**
```json
{
  "success": true,
  "total_images": 3,
  "results": [
    {
      "filename": "image1.jpg",
      "num_detections": 5,
      "detections": [...]
    },
    ...
  ]
}
```

### `GET /info`
Get model and configuration information.

**Response:**
```json
{
  "model_path": "/app/models/ankesh_sir model/checkpoint_best_total.pth",
  "class_names": {
    "0": "Artwork",
    "1": "Not_Artwork",
    "2": "Front_Panel",
    ...
  },
  "config": {
    "conf_threshold": 0.25,
    "iou_threshold": 0.7,
    "img_size": 512
  }
}
```

---

## Configuration

Edit `docker-compose.yml` to customize:

```yaml
environment:
  - MODEL_PATH=/app/models/ankesh_sir model/checkpoint_best_total.pth
  - CONF_THRESHOLD=0.25        # Detection confidence threshold
  - IOU_THRESHOLD=0.7          # NMS IOU threshold
  - IMG_SIZE=512               # Input image size
```

---

## Directory Structure

```
task3/
├── Dockerfile              # Container image definition
├── docker-compose.yml      # Multi-container orchestration
├── app.py                  # Flask API server
├── run_inference.py        # Legacy inference script
├── requirements.txt        # Python dependencies
├── trained_models/
│   └── ankesh_sir model/
│       └── checkpoint_best_total.pth  # Your trained model
├── data/
│   ├── input/              # Place input images here
│   └── output/             # Output annotations saved here
└── .dockerignore           # Excludes unnecessary files from image
```

---

## Running Different Scenarios

### Local Development (no Docker)
```bash
pip install -r requirements.txt
python app.py
```

### Production Deployment
For production, consider:
- Pushing image to Docker Hub or registry: `docker tag ml-model-inference:latest myusername/ml-inference:v1.0 && docker push myusername/ml-inference:v1.0`
- Using GPU: Add `runtime: nvidia` to docker-compose.yml
- Horizontal scaling: Use Docker Swarm or Kubernetes
- Reverse proxy: Add Nginx/HAProxy for load balancing

### Running with GPU
Edit `docker-compose.yml`:
```yaml
ml-inference:
  runtime: nvidia
  environment:
    - CUDA_VISIBLE_DEVICES=0
```

---

## Troubleshooting

### Build Takes Too Long
PyTorch download can be slow. Consider pre-building and caching:
```bash
docker build -t ml-model-inference:latest . --progress=plain
```

### Port Already in Use
Change in `docker-compose.yml`:
```yaml
ports:
  - "8080:5000"  # Access API on http://localhost:8080
```

### Model Not Found
Ensure `trained_models/ankesh_sir model/checkpoint_best_total.pth` exists locally.

### Out of Memory
Increase Docker memory limit or reduce `IMG_SIZE` in docker-compose.yml.

---

## Files Included

- **Dockerfile**: Multi-stage or simplified single-stage Python image with PyTorch, ultralytics, and Flask
- **docker-compose.yml**: Orchestrates the container with proper volume mounts and configuration
- **app.py**: Flask API with `/predict`, `/batch`, `/health`, and `/info` endpoints
- **.dockerignore**: Excludes unnecessary files to optimize image size
- **requirements.txt**: Updated with `flask` and `requests` for the API
