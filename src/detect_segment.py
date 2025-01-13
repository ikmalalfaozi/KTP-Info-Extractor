import cv2
import numpy as np
from ultralytics import YOLO


class KTPDetector:
    def __init__(self, model_path, device="cpu"):
        """
        Initialize the KTP detector.

        Args:
            model_path (str): Path to the YOLOv11-seg model.
            device (str): Device to use ("cuda" or "cpu").
        """
        self.device = device
        self.model = YOLO(model_path)

    def detect_and_segment(self, image_path, conf=0.5):
        """
        Perform detection and segmentation of the KTP area.

        Args:
            image_path (str): Path to the input image.
            conf (float): Confidence score of prediction [0,1].

        Returns:
            dict: Dictionary containing detection and segmentation results.
                - 'bbox': Coordinates of the bounding box [x_min, y_min, x_max, y_max]
                - 'mask': Binary segmentation mask (if available)
        """
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found: {image_path}")

        # Perform detection
        results = self.model.predict(source=image, conf=conf, device=self.device)
        if len(results) == 0:
            raise ValueError("No KTP detected.")

        # Extract bounding boxes and masks for all detections
        detections = []
        for detection in results:
            boxes = detection.boxes.xyxy.cpu().numpy()  # Bounding boxes
            masks = detection.masks.data.cpu().numpy() if detection.masks else None  # Masks

            for i, box in enumerate(boxes):
                bbox = box.astype(int)
                mask = masks[i] if masks is not None else None
                detections.append({
                    'bbox': bbox,
                    'mask': cv2.resize(mask, image.shape[:2][::-1])
                })

            return detections


if __name__ == "__main__":
    # Paths to the model and image
    model_path = "models/YOLO11n-seg-ktp.pt"
    image_path = "data/raw/5.jpg"

    # Initialize the detector
    detector = KTPDetector(model_path)

    # Perform detection and segmentation
    try:
        results = detector.detect_and_segment(image_path)

        # Load the image for visualization
        image = cv2.imread(image_path)
        image_vis = image.copy()

        # Define a padding value to avoid the bounding box being too tight
        padding = 20

        for i, result in enumerate(results):
            bbox = result['bbox']
            mask = result['mask']

            # Draw the bounding box (without padding in the visualization)
            cv2.rectangle(image_vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)

            if mask is not None:
                # Overlay the mask on the image for visualization
                mask_overlay = np.zeros_like(image, dtype=np.uint8)
                mask_overlay[mask == 1] = (0, 255, 0)  # Green color for the mask
                image_vis = cv2.addWeighted(image_vis, 0.7, mask_overlay, 0.3, 0)

                # Apply padding only to the individual KTP image and mask
                x1, y1, x2, y2 = bbox
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(image.shape[1], x2 + padding)
                y2 = min(image.shape[0], y2 + padding)

                # Save the individual KTP image (cropped with padding)
                ktp_image = image[y1:y2, x1:x2]
                ktp_image_path = f"data/processed/ktp_{i}.jpg"
                cv2.imwrite(ktp_image_path, ktp_image)
                print(f"Individual KTP image saved at: {ktp_image_path}")

                # Save the individual mask image (cropped with padding)
                mask_image = mask[y1:y2, x1:x2]
                mask_image_path = f"data/processed/mask_{i}.png"
                cv2.imwrite(mask_image_path, mask_image * 255)  # Convert binary mask to an image
                print(f"Individual mask saved at: {mask_image_path}")

        # Save the visualization result (without padding for the bounding box)
        output_path = "data/processed/ktp_detected.jpg"
        cv2.imwrite(output_path, image_vis)
        print(f"Detection results saved at: {output_path}")

    except ValueError as e:
        print(e)

