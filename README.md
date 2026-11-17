# LeafScope

LeafScope is an AI-powered plant leaf analysis project for detecting crop diseases and estimating disease severity from leaf images. It combines deep learning with computer-vision techniques to provide an interpretable diagnosis through a local web interface.

## Overview

The project:

- Classifies plant diseases across the 38 crop and disease categories in the PlantVillage dataset.
- Compares ResNet-50 and MobileNetV2 transfer-learning models.
- Uses Grad-CAM visualizations to show which parts of a leaf influenced each prediction.
- Estimates lesion severity using OpenCV-based leaf and tissue segmentation.
- Evaluates accuracy, F1 scores, confusion matrices, model size, latency, and robustness to lighting changes.
- Provides an interactive browser interface for uploading leaf images and reviewing predictions, attention maps, and severity results.

## Technology

LeafScope is built with:

- Python
- PyTorch and torchvision
- OpenCV
- NumPy
- Matplotlib
- HTML, CSS, and JavaScript
- Python's built-in `http.server`

## Project Structure

| Component | Purpose |
|---|---|
| `models.py` | ResNet-50 and MobileNetV2 model definitions |
| `train.py` | Transfer-learning training pipeline |
| `evaluate.py` | Model comparison and evaluation |
| `gradcam.py` | Prediction visualizations using Grad-CAM |
| `severity.py` | OpenCV-based lesion and severity estimation |
| `prepare_data.py` | Dataset preparation and train/validation/test splits |
| `inference.py` | Single-image prediction |
| `app/` | Local web application and user interface |

## Dataset

LeafScope uses the public [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset), an image collection of healthy and diseased plant leaves. Because the dataset was captured mostly in controlled conditions, results may not directly represent performance on field photographs.

## Scope and Limitations

The project is intended as an educational and experimental computer-vision system. Disease predictions are limited to the classes represented in PlantVillage, while severity is an image-based estimate rather than a medically or agriculturally validated measurement.
