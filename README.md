# Face Recognition System using InsightFace

A lightweight real-time face recognition system built with **Python, OpenCV, and InsightFace**. The system detects faces, generates face embeddings using ArcFace, compares them with stored reference faces, and displays the recognition result with a bounding box and similarity score.

It supports both:

* 💻 Laptop/Webcam
* 📷 Intel RealSense D435i

The application is optimized for CPU-based execution and includes configurable detection, processing, frame-rate, and matching parameters.

---

## Features

* Real-time face detection and recognition
* ArcFace-based face embeddings
* Cosine similarity matching
* Reference face image support
* `Match` / `Unknown` classification
* Similarity score displayed on the bounding box
* Temporal matching history to reduce unstable predictions
* Configurable recognition threshold
* Configurable detection resolution
* Configurable camera resolution and FPS
* Frame skipping for better CPU performance
* Processing-scale optimization
* Laptop webcam support
* Intel RealSense D435i support
* Optional RealSense depth visualization
* FPS monitoring for display and recognition

---

## Technologies Used

* **Python**
* **OpenCV**
* **NumPy**
* **InsightFace**
* **ArcFace**
* **ONNX Runtime / CPU Execution Provider**
* **Intel RealSense SDK** *(only required when using the RealSense camera)*

---

## Project Structure

```text
Face-Recognition/
│
├── face.py
├── reference_faces/
│   ├── person1.jpg
│   ├── person2.jpg
│   └── ...
│
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

### 2. Install dependencies

```bash
pip install opencv-python numpy insightface onnxruntime
```

For Intel RealSense support:

```bash
pip install pyrealsense2
```

> RealSense functionality requires an Intel RealSense camera and the appropriate RealSense software/SDK setup.

---

## Reference Faces

Create a folder named:

```text
reference_faces
```

Place the reference images inside this folder.

Supported formats:

```text
.jpg
.jpeg
.png
.bmp
.webp
```

Example:

```text
reference_faces/
├── vansh.jpg
├── person2.jpg
└── person3.jpg
```

The program generates an embedding for each reference image and uses these embeddings for comparison with faces detected by the camera.

---

## Configuration

The main configuration parameters are located near the beginning of `face.py`.

### Camera Resolution

```python
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_FPS = 30
```

These values specify the requested camera resolution and frame rate.

For example:

```python
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
```

The actual resolution depends on what the camera and driver support.

---

### Face Detection Resolution

```python
DETECTION_SIZE = (320, 320)
```

This controls the resolution used internally by the InsightFace detector.

A larger value can improve detection of smaller faces but requires more processing.

Example:

```python
DETECTION_SIZE = (640, 640)
```

For CPU systems, `320x320` is generally preferable when performance is important.

---

### Processing Scale

```python
PROCESSING_SCALE = 0.75
```

This resizes the camera frame before face detection and recognition.

Examples:

```python
PROCESSING_SCALE = 1.0
```

Processes the original frame.

```python
PROCESSING_SCALE = 0.5
```

Processes a smaller frame and can improve performance.

---

### Processing Frequency

```python
PROCESS_EVERY_N_FRAMES = 8
```

Face recognition is performed once every N frames.

For example:

```text
1 → process every frame
4 → process every 4th frame
8 → process every 8th frame
```

Increasing this value can improve overall display FPS but makes recognition updates less frequent.

---

### Recognition Threshold

```python
MATCH_THRESHOLD = 0.4
```

This controls how similar a detected face must be to a reference embedding to be classified as a match.

Increasing the threshold makes matching stricter.

For example:

```python
MATCH_THRESHOLD = 0.5
```

may reduce false matches but can also cause genuine faces to be classified as `Unknown`.

---

### Temporal Matching

The system also maintains a short history of matching results:

```python
MATCH_HISTORY_LEN = 1
MATCH_VOTE_THRESHOLD = 1
```

This mechanism can be configured to require multiple positive detections before confirming a match.

For example:

```python
MATCH_HISTORY_LEN = 5
MATCH_VOTE_THRESHOLD = 3
```

would require at least 3 positive results within the last 5 processed detections.

---

## Running the Program

### Laptop Webcam

```bash
python face.py --source laptop
```

To use a specific camera index:

```bash
python face.py --source laptop --camera-index 0
```

---

### Intel RealSense D435i

```bash
python face.py --source realsense
```

The program uses the RealSense color stream for face recognition.

Depth visualization can optionally be enabled by changing:

```python
SHOW_DEPTH = True
```

---

## Command-Line Arguments

| Argument             | Description                            | Default           |
| -------------------- | -------------------------------------- | ----------------- |
| `--source`           | Camera source: `laptop` or `realsense` | `laptop`          |
| `--camera-index`     | OpenCV camera index                    | `0`               |
| `--reference-folder` | Folder containing reference images     | `reference_faces` |

Examples:

```bash
python face.py --source laptop
```

```bash
python face.py --source laptop --camera-index 1
```

```bash
python face.py --source realsense
```

```bash
python face.py --source laptop --reference-folder reference_faces
```

---

## How It Works

The recognition pipeline works approximately as follows:

```text
Camera
   │
   ▼
Capture Frame
   │
   ▼
Resize / Processing Scale
   │
   ▼
InsightFace Face Detection
   │
   ▼
Face Embedding (ArcFace)
   │
   ▼
Cosine Similarity
   │
   ▼
Compare with Reference Embeddings
   │
   ├── Similarity ≥ Threshold
   │          │
   │          ▼
   │       Match
   │
   └── Similarity < Threshold
              │
              ▼
           Unknown
```

---

## Performance Optimization

The system contains several parameters designed to reduce CPU usage:

### 1. Detection Size

```python
DETECTION_SIZE = (320, 320)
```

Smaller detector input generally requires less computation.

### 2. Processing Scale

```python
PROCESSING_SCALE = 0.75
```

The frame is resized before recognition.

### 3. Frame Skipping

```python
PROCESS_EVERY_N_FRAMES = 8
```

Recognition does not need to run on every camera frame.

### 4. CPU Execution

The InsightFace analyzer is configured to use:

```python
providers=["CPUExecutionProvider"]
```

The project therefore does not require a CUDA GPU for inference.

---

## FPS Monitoring

The application displays:

```text
Display FPS
Recognition FPS
```

This helps evaluate the performance of the camera and recognition pipeline.

---

## Controls

Press:

```text
Q
```

to exit the application.

---

## Important Notes

* The camera must support the requested resolution for that resolution to actually be used.
* Increasing `FRAME_WIDTH` and `FRAME_HEIGHT` increases camera/display resolution but can increase processing requirements.
* Increasing `DETECTION_SIZE` can improve detection of smaller faces but increases CPU usage.
* Lowering `PROCESSING_SCALE` can significantly improve performance.
* Increasing `PROCESS_EVERY_N_FRAMES` reduces recognition frequency.
* The recognition threshold should be tuned according to the environment and reference images.
* Reference images should contain clear and recognizable faces.
* Multiple faces in a reference image are handled by selecting the largest detected face.

---

## Future Improvements

Possible improvements include:

* Multi-person identity database
* Better face tracking
* Improved temporal voting
* Automatic camera resolution detection
* GPU acceleration
* Face enrollment interface
* Database-based identity management
* RealSense depth-based distance estimation
* Anti-spoofing / liveness detection
* Web-based monitoring dashboard

---

## License

This project is intended for educational, research, and development purposes.

Add an appropriate open-source license if you plan to distribute the project publicly.
