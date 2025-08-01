Custom Object Detection for Inclined-Angle Drone Footage using YOLOv8 🚀

![alt text](https://img.shields.io/badge/Project-Custom%20Object%20Detection-blue)

![alt text](https://img.shields.io/badge/Python-3.9+-blue.svg)

![alt text](https://img.shields.io/badge/Framework-YOLOv8-red)

![alt text](https://img.shields.io/badge/License-MIT-green)

This project delivers a complete workflow for training a custom YOLOv8 model to perform object detection on video captured from a low, inclined-angle drone perspective. By creating a bespoke dataset and leveraging aggressive data augmentation, we successfully developed a lightweight model capable of accurately identifying key objects from this unique and challenging viewpoint.
📝 Table of Contents

    Problem Statement

    Our Solution

    Methodology

        Model Selection

        Dataset Creation & Augmentation

        Data Structure

    Getting Started

        Prerequisites

        Training the Model

        Running Detection

    Results & Performance

    Conclusion & Future Work

1. Problem Statement

The primary goal was to perform object detection on video frames to identify key classes: man, tree, bush, building, and vehicle. The core challenge stemmed from the video footage, which was recorded by a drone at a low, inclined angle. Standard object detection models are typically pre-trained on ground-level or high-altitude, top-down imagery and thus perform poorly on this intermediate, angled perspective. This gap necessitated the creation and training of a custom model tailored specifically to our data.
2. Our Solution

To address this challenge, we developed a custom object detection model using YOLOv8. A significant portion of the project involved building a bespoke dataset from video frames. Through an iterative process of manual annotation, semi-automated labeling with Roboflow, and extensive data augmentation, we trained a lightweight YOLOv8n model capable of accurately identifying objects from this unique viewpoint.
3. Methodology
3.1. Model Selection

Due to hardware constraints and the potential for real-time, on-device deployment, the YOLOv8n (nano) model was the ideal choice. It offers an excellent trade-off between inference speed and accuracy, making it perfectly suited for applications on resource-constrained devices like drones.
3.2. Dataset Creation & Augmentation

The unavailability of a suitable public dataset was a major hurdle. Our dataset creation process was iterative and methodical:

    Initial Dataset (V1):

        Extracted frames from the source videos.

        Manually annotated approximately 250 images in Roboflow.
