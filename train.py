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

print("Training complete.")
