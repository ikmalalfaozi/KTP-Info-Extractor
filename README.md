# KTP Information Extractor

This repository contains a complete pipeline for extracting information from KTP (Kartu Tanda Penduduk) images using advanced deep learning techniques. The pipeline includes segmentation, corner detection, image warping, orientation correction, and information extraction using a fine-tuned Donut model.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Model Information](#model-information)
4. [Installation](#installation)
5. [Usage](#usage)
   - [Step-by-Step Example](#step-by-step-example)
   - [Pipeline Execution](#pipeline-execution)
7. [Acknowledgments](#acknowledgments)

---

## Introduction

This repository contains a modular pipeline for processing KTP images and extracting relevant information using trained models. The pipeline is designed for flexibility and robustness, handling cases where multiple KTPs are present in a single image.

---

## Features

- **KTP Detection**: YOLO-based segmentation model to detect KTP regions in an image.
- **Corner Detection**: Corner detection using the `docaligner` library and contour-based methods.
- **Image Warping**: Using detected corners, the KTP is warped to a normalized perspective. 
- **Orientation Correction**: YOLO-based KTP orientation classifier.
- **Information Extraction**: Fine-tuned Donut model for extracting structured information.

---

## Model Information
### KTP Segmentation Model
- **Architecture**: YOLOv11-based segmentation
- **Model Performance**:  
    YOLO segmentation models trained on the [KTP Image Segmentation (v1)](https://www.kaggle.com/datasets/ikmalalfaozi/ktp-image-segmentation-v1) dataset.

   | Model       | Box mAP@50<br/>(train/valid/test) | Box mAP@50-95<br/>(train/valid/test) | Mask mAP@50<br/>(train/valid/test) | Mask mAP@50-95<br/>(train/valid/test) |
   |-------------|-----------------------------------|--------------------------------------|------------------------------------|---------------------------------------|
   | YOLO11n-seg | 99.5 / 99.4 / 99.5                | 99.2 / 99.4 / 99.1                   | 99.5 / 99.4 / 99.5                 | 99.3 / 99.3 / 99.3                    |
   | YOLO11s-seg | 99.5 / 99.4 / 99.5                | 99.4 / 99.4 / 99.3                   | 99.5 / 99.4 / 99.5                 | 99.4 / 99.4 / 99.2                    |
   | YOLO11m-seg | 99.5 / 99.5 / 99.5                | 99.5 / 99.5 / 99.4                   | 99.5 / 99.5 / 99.5                 | 99.4 / 99.4 / 99.4                    |

### Corner Detection Model
- **Architecture**: 
  - Corner detection is handled primarily by the `docaligner` library. 
  - If `docaligner` fails to detect the corners, a contour-based method is used as a fallback.
- **Accuracy**: 
  - Evaluation metrics for `docaligner` on the SmartDoc 2015 dataset can be seen on [this website](https://docsaid.org/en/docs/docaligner/benchmark).

### Orientation Classification Model
- **Architecture**: YOLOv11-based classifier
- **Accuracy**: 1 / 1 / 99.78 (train/valid/test)  
    YOLO classification model trained on the [KTP Orientation Classification](https://www.kaggle.com/datasets/ikmalalfaozi/ktp-orientation-classification) dataset.

### Information Extraction Model
- **Architecture**: Donut (fine-tuned)
- **n-TED Accuracy**: 98.69 / 96.37 (train/test)  
    Fine-tuned Donut model trained on the [ktp-donut-v1](https://huggingface.co/datasets/ikmalalfaozi/ktp-donut-v1) dataset. n-TED Accuracy measures the normalized edit distance between the predicted and ground truth JSON structures.

---


## Installation

1. Install Python
2. Clone the repository:
    ```bash
    git clone https://github.com/ikmalalfaozi/KTP-Info-Extractor.git
    cd KTP-Info-Extractor
    ```
3. Install all required dependencies with:
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

### Step-by-Step Example

#### 1. **KTP Detection**

Detect KTP regions in an image using the YOLO-based segmentation model:

```python
from src import KTPDetector

ktp_detector = KTPDetector("models/YOLO11n-seg-ktp.pt")
ktp_regions = ktp_detector.detect_and_segment("sample_image.jpg", conf=0.9)
print(ktp_regions) # [{bbox_1, mask_1}, {bbox_2, mask_2}, ...]
```

#### 2. **Corner Detection**

Detect corners within the KTP region using the `docaligner` library with a fallback to contour-based methods:

```python
from src import CornerDetector
from docaligner import ModelType
from capybara import Backend

corner_detector = CornerDetector(
    model_type=ModelType.heatmap,
    model_cfg="fastvit_t8",
    backend=Backend.cpu
)
corners = corner_detector.detect_corners(roi, mask_roi)

if not corners:
    corners = corner_detector.detect_corners_using_contour(roi)

print(corners) # [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
```

#### 3. Image Warping

Warp KTP image to a normalized perspective.

```python
from src import ImageWarper

warped_image = ImageWarper.warp_image(roi, corners, output_size=(856, 540))
```

#### 4. **Orientation Correction**

Correct the orientation of the warped KTP image:

```python
from src import ImageOrientation

ktp_orientation = ImageOrientation("models/YOLO11n-ktp-oc.pt")
predicted_orientation = ktp_orientation.detect_orientation(warped_image)
corrected_image = ktp_orientation.correct_orientation(warped_image, predicted_orientation)
```

#### 4. **Information Extraction**

Extract structured information using the Donut model:

```python
from src.extract_info import KTPInfoExtractor

extractor = KTPInfoExtractor(model_name="ikmalalfaozi/donut-base-finetuned-ktp-v1")
result = extractor.predict(corrected_image)
print(result)
```

### Pipeline Execution
- **Run the entire pipeline for image alignment**:

    ```bash
    python image_alignment.py
    ```
  
    Provide or change the necessary paths for input images folder, models, and output folders in the script.

- **Run the entire pipeline for extracting information from KTP image**:

    ```bash
    python src/run_pipeline.py
    ```

    Provide or change the necessary paths for input images folder, models, and output folders in the script.

- **Run the application demo**:

  ```bash
  streamlit run app.py
  ```

---


## Acknowledgments

This project leverages the following open-source tools and models:

- [Donut](https://github.com/clovaai/donut)
- [YOLOv11](https://github.com/ultralytics/ultralytics)
- [DocAligner](https://github.com/DocsaidLab/DocAligner)

Special thanks to the creators of these tools for their contributions to the open-source community.

---

For any questions or issues, please open an [issue](https://github.com/ikmalalfaozi/KTP-Info-Extractor/issues).

