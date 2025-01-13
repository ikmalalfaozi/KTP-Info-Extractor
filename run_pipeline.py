import json
import os
from PIL import Image
import cv2
from docaligner import ModelType
from capybara import Backend
from src import KTPDetector
from src import CornerDetector
from src import ImageWarper
from src import ImageOrientation
from src.extract_info import KTPInfoExtractor


def run_pipeline(input_folder: str,
                 output_folder: str,
                 segment_model_path: str,
                 corner_model_config: dict,
                 cls_model_path: str,
                 extractor_model_name: str):
    """
    Run the complete pipeline to extract information from KTP images.

    Parameters:
    input_folder (str): Path to the folder containing input KTP images.
    output_folder (str): Path to the folder where results will be saved.
    segment_model_path (str): Path to YOLO model for KTP detection.
    corner_model_config (dict): Configuration for corner detector.
    cls_model_path (str): Path to YOLO model for KTP orientation classification.
    extractor_model_name (str): Name of the Donut model to use for information extraction.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    ktp_detector = KTPDetector(segment_model_path)

    # Initialize the corner detector
    corner_detector = CornerDetector(
        model_type=corner_model_config['model_type'],
        model_cfg=corner_model_config['model_cfg'],
        backend=corner_model_config['backend']
    )

    # Initialize the KTP orientation classifier
    ktp_oc = ImageOrientation(cls_model_path)

    # Initialize the KTP information extractor
    extractor = KTPInfoExtractor(model_name=extractor_model_name)

    # Iterate through all images in the input folder
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        if not filename.lower().endswith(('jpg', 'jpeg', 'png')):
            print(f"Skipping non-image file: {filename}")
            continue

        try:
            # Open the image
            img = cv2.imread(input_path)
            if img is None:
                print(f"Failed to load image: {img}")
                continue

            # Detect KTP in the image
            try:
                ktp_regions = ktp_detector.detect_and_segment(input_path, conf=0.9)
            except ValueError as e:
                print(f"{input_path}: {e}")
                continue

            # Define a padding value to avoid the bounding box being too tight
            padding = 10

            results = []

            for i, ktp_region in enumerate(ktp_regions):
                bbox = ktp_region['bbox']
                mask = ktp_region['mask']

                #  Extract the region of interest (ROI) based on segmentation mask
                x1, y1, x2, y2 = bbox
                x_min = max(0, x1 - padding)
                y_min = max(0, y1 - padding)
                x_max = min(img.shape[1], x2 + padding)
                y_max = min(img.shape[0], y2 + padding)
                roi = img[y_min:y_max, x_min:x_max]
                mask_roi = mask[y_min:y_max, x_min:x_max] * 255

                # Detect corners within the ROI
                corners = corner_detector.detect_corners(roi, mask_roi)

                if len(corners) != 4:
                    print(f"Invalid corners detected in image {input_path}, mask {i}. Skipping.")
                    continue

                # Warp the image using detected corners
                warped_image = ImageWarper.warp_image(roi, corners, output_size=(856, 540))

                # Detect orientation
                predicted_orientation = ktp_oc.detect_orientation(warped_image)
                # Correct orientation
                corrected_image = ktp_oc.correct_orientation(warped_image, predicted_orientation)
                # Check image size
                h, w = corrected_image.shape[:2]
                if h > w:
                    corrected_image = cv2.resize(corrected_image, (h, w), interpolation=cv2.INTER_AREA)

                # Extract information from the warped image
                result = extractor.predict(Image.fromarray(cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB)))
                results.append(result)

            # Save the extracted information to a JSON file
            output_file = os.path.join(output_folder, os.path.splitext(filename)[0] + ".json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

            print(f"Processed and saved results for: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    # Configuration for the corner detector
    corner_model_config = {
        'model_type': ModelType.heatmap,  # Options: 'heatmap', 'point'
        'model_cfg': 'fastvit_t8',  # Options depend on model_type
        'backend': Backend.cpu  # Options: 'cpu', 'cuda'
    }
    # Model path
    segment_model_path = "models/YOLO11n-seg-ktp.pt"
    cls_model_path = "models/YOLO11n-ktp-oc.pt"

    # Extractor model name
    extractor_model_name = "ikmalalfaozi/donut-base-finetuned-ktp-v1"

    input_folder = "ktp/images"
    output_folder = "ktp/results"

    run_pipeline(input_folder, output_folder,
                 segment_model_path, corner_model_config,
                 cls_model_path, extractor_model_name)
