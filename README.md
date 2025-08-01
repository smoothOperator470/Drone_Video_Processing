# Custom Object Detection for Inclined-Angle Drone Footage using YOLOv8 🚀

A custom YOLOv8 model trained specifically for object detection on drone video captured from low, inclined angles. This project addresses the unique challenge of detecting objects from perspectives that standard pre-trained models aren't optimized for.

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project successfully develops a custom object detection solution for drone footage captured from challenging inclined angles. Using YOLOv8n (nano), we created a lightweight model capable of accurately identifying vehicles, people, buildings, trees, and bushes from unique aerial perspectives.

**Key Achievement:** 48% mean Average Precision (mAP50) on validation set with a lightweight, deployable model.

## 🚨 Problem Statement

Standard pre-trained object detection models are typically trained on:
- Ground-level imagery
- High-altitude top-down views (remote sensing)

Our challenge was detecting objects in drone footage from **intermediate, inclined angles** - a perspective that existing models handle poorly. This required creating a custom dataset and training a specialized model.

### Target Classes
- 🚗 **Vehicle**: Cars, trucks, and other vehicles
- 👥 **Person**: People in various poses and positions  
- 🏢 **Building**: Structures and buildings
- 🌳 **Tree**: Trees and large vegetation
- 🌿 **Bush**: Bushes and smaller vegetation

## ✨ Features

- **Custom YOLOv8n Model**: Optimized for inclined-angle drone footage
- **Real-time Detection**: Lightweight model suitable for on-device deployment
- **Comprehensive Logging**: Frame-by-frame detection logs with timestamps
- **Data Augmentation Pipeline**: Robust dataset enhancement techniques
- **Video Processing**: Complete workflow from training to inference

## 📊 Dataset

### Dataset Creation Process

#### Version 1 (Initial Dataset)
- 250+ manually annotated images extracted from drone video frames
- Manual annotation using Roboflow workspace
- Semi-automated labeling using trained model as "Roboflow bot"
- Maintained class balance across all target categories

#### Version 2 (Augmented Dataset)
- Enhanced dataset with aggressive data augmentation
- Applied transformations: tilt, rotation, translation, noise, blur
- Additional manual annotations for improved robustness
- Final training dataset used for production model

### Dataset Structure
```
/dataset
├── images/
│   ├── train/          # Training images
│   └── val/            # Validation images
├── labels/
│   ├── train/          # Training labels (YOLO format)
│   └── val/            # Validation labels (YOLO format)
└── data.yaml           # Dataset configuration
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- OpenCV
- Ultralytics YOLOv8

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/drone-object-detection.git
cd drone-object-detection
```

2. **Install dependencies:**
```bash
pip install ultralytics opencv-python
```

3. **Prepare your dataset:**
   - Organize images and labels according to the dataset structure above
   - Update `data.yaml` with your dataset paths and class names

## 🚀 Usage

### Training the Model

Run the training script to train your custom YOLOv8n model:

```bash
python train.py
```

**Training Configuration:**
- **Model**: YOLOv8n (nano) - optimized for speed and efficiency
- **Epochs**: 100
- **Image Size**: 640x640
- **Batch Size**: 16
- **Workers**: 4

### Running Detection

Perform object detection on video files:

```bash
python detect.py
```

**Output:**
- `output_video.mp4`: Annotated video with bounding boxes
- `detections_log.txt`: Detailed frame-by-frame detection log
- Real-time console output showing detections

### Complete Code Implementation

#### `train.py` - Model Training Script
```python
# train.py
# This script trains a custom YOLOv8 model for object detection.

from ultralytics import YOLO

# --- 1. Load a Pre-trained Model ---
# Load the YOLOv8 Nano ('n') model. 'yolov8n.pt' contains weights pre-trained on the COCO dataset.
# This is the starting point for our custom training (transfer learning).
# The nano model is the smallest and fastest, ideal for devices with less compute power.
model = YOLO('yolov8n.pt')

# --- 2. Train the Model on a Custom Dataset ---
# The train() method starts the training process.
results = model.train(
   data='data.yaml',        # Path to the dataset configuration file. This file tells YOLO where to find the
                           # training/validation images and what the class names are.
   epochs=100,              # The total number of times the model will cycle through the entire training dataset.
   imgsz=640,               # The input image size. All images will be resized to 640x640 pixels before
                           # being fed into the network.
   batch=16,                # The number of images to process in a single batch. A larger batch size can speed up
                           # training but requires more memory (VRAM).
   name='yolov8_custom',    # The name of the experiment. Results (weights, logs) will be saved in a
                           # directory named 'runs/detect/yolov8_custom'.
   workers=4                # The number of worker threads for loading data. Speeds up data preprocessing.
)

print("✅ Training complete.")
```

#### `detect.py` - Detection/Inference Script
```python
# detect.py
# This script uses the trained custom YOLOv8 model to perform object detection on a video file.
# It saves the output as a new video with bounding boxes and creates a text log of all detections.

import cv2
from ultralytics import YOLO

# --- 1. Initialization ---

# Load the custom-trained YOLOv8 model.
# 'best.pt' is the model file with the best validation performance from our training session.
model = YOLO("runs/detect/yolov8_custom/weights/best.pt")

# Define the input video path and open it using OpenCV.
video_path = "input.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties (FPS, width, height) to create the output video correctly.
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_number = 0

# Set up the VideoWriter object to save the output video.
# 'mp4v' is the codec for the .mp4 format.
out = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

# Open a log file in write mode to store detection details.
log_file = open("detections_log.txt", "w")

print("🎬 Starting detection on video...")

# --- 2. Main Processing Loop ---
while True:
    # Read one frame from the video. 'ret' is True if a frame was read successfully.
    ret, frame = cap.read()
    if not ret:
        break  # Exit the loop if the video has ended.

    # --- Perform Inference ---
    # Pass the frame to the model for detection. `verbose=False` keeps the console clean.
    # `[0]` selects the detection results for the first (and only) image.
    results = model(frame, verbose=False)[0]
    annotated_frame = frame.copy() # Create a copy to draw on, leaving the original intact.

    # --- Process and Log Detections ---
    detections = []
    # Iterate through each detected bounding box in the results.
    for box in results.boxes:
        cls_id = int(box.cls)            # Get the class ID (e.g., 0 for 'person').
        conf = float(box.conf)           # Get the confidence score of the detection.
        label = model.names[cls_id]      # Get the class name from the model's metadata.
        x1, y1, x2, y2 = map(int, box.xyxy[0]) # Get the bounding box coordinates.

        # Draw the bounding box on the frame.
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Draw the label and confidence score above the box.
        cv2.putText(annotated_frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add the detection info to our list for logging.
        detections.append(f"{label} ({conf:.2f})")

    # --- Write to Log File ---
    timestamp = frame_number / fps # Calculate the timestamp in seconds.
    if detections:
        line = f"Frame {frame_number}, Time {timestamp:.2f}s: " + ", ".join(detections)
    else:
        line = f"Frame {frame_number}, Time {timestamp:.2f}s: No objects detected"
    
    log_file.write(line + "\n")
    print(line) # Also print the log line to the console for real-time feedback.

    # --- Display and Save Frame ---
    # Display the frame with detections in a window.
    cv2.imshow("YOLOv8 Detection", annotated_frame)
    # Save the annotated frame to the output video file.
    out.write(annotated_frame)
    
    frame_number += 1
    
    # Exit the loop if the 'q' key is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("✅ Detection and logging complete.")
cap.release()
out.release()
log_file.close()
cv2.destroyAllWindows()
```

#### `data.yaml` - Dataset Configuration
```yaml
# Dataset configuration file for YOLOv8 training
# This file defines the dataset structure and class names

# Dataset paths
train: dataset/images/train  # Path to training images
val: dataset/images/val      # Path to validation images

# Number of classes
nc: 5

# Class names (must match the order used during annotation)
names:
  0: person     # People in various poses and positions
  1: vehicle    # Cars, trucks, and other vehicles  
  2: building   # Structures and buildings
  3: tree       # Trees and large vegetation
  4: bush       # Bushes and smaller vegetation
```

## 📈 Model Performance

### Validation Results
- **mAP50**: 48% (mean Average Precision at IoU threshold 0.5)
- **Model Size**: YOLOv8n (lightweight, suitable for edge deployment)
- **Inference Speed**: Real-time capable on standard hardware

### Performance Analysis
- **Strong Points**: Effective detection of vehicles and people
- **Areas for Improvement**: Confusion between visually similar classes (tree vs bush)
- **Recommendation**: Larger models (YOLOv8m/YOLOv8l) for improved accuracy

## 📁 Project Structure

```
drone-object-detection/
├── train.py                 # Model training script
├── detect.py                # Detection/inference script
├── data.yaml                # Dataset configuration
├── dataset/                 # Training dataset
│   ├── images/
│   └── labels/
├── runs/detect/yolov8_custom/    # Training outputs
│   └── weights/
│       └── best.pt          # Best model weights
├── input.mp4                # Input video file
├── output_video.mp4         # Annotated output video
├── detections_log.txt       # Detection log file
└── README.md               # This file
```

## 🎯 Results

### Key Achievements
✅ Successfully trained custom model for challenging inclined-angle drone footage  
✅ Achieved 48% mAP50 on validation set  
✅ Created comprehensive dataset with effective augmentation pipeline  
✅ Developed complete end-to-end detection workflow  
✅ Lightweight model suitable for drone deployment  

### Sample Detections
The trained model successfully detects:
- Vehicles from aerial perspectives
- People in various outdoor settings
- Buildings and structures
- Vegetation (trees and bushes)

### Confusion Matrix Insights
- Model performs well on distinct object classes
- Some confusion between similar vegetation types (tree/bush)
- Clear direction for future dataset improvements

## 🔮 Future Work

### Immediate Improvements
- **Larger Models**: Train YOLOv8m or YOLOv8l for higher accuracy
- **Dataset Expansion**: Add more diverse examples, especially for confused classes
- **Hardware Optimization**: Deploy on high-performance GPUs

### Long-term Goals
- **Production Deployment**: Real-time inference on drone hardware
- **Multi-angle Training**: Expand to various drone angles and altitudes
- **Advanced Augmentation**: Implement more sophisticated data enhancement
- **Class Refinement**: Better distinguish between similar object types

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ultralytics** for the YOLOv8 framework
- **Roboflow** for annotation tools and dataset management
- **OpenCV** for video processing capabilities

## 📞 Contact

For questions, suggestions, or collaboration opportunities:
- **Project Maintainer**: [Your Name]
- **Email**: [your.email@example.com]
- **GitHub**: [@yourusername](https://github.com/yourusername)

---

**Note**: This project demonstrates a complete workflow for custom object detection on specialized drone footage. The methodology and results provide a solid foundation for developing production-ready aerial detection systems.