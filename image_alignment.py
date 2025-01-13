import os
import cv2
from docaligner import ModelType
from capybara import Backend
from src import KTPDetector
from src import CornerDetector
from src import ImageWarper
from src import ImageOrientation


class KTPImageProcessor:
    def __init__(self, segment_model_path, corner_model_config, cls_model_path):
        """
        Initialize the KTP Processor with models for detection and corner detection.

        Args:
            segment_model_path (str): Path to YOLO model for KTP detection.
            corner_model_config (dict): Configuration for corner detector.
            cls_model_path (str): Path to YOLO model for KTP orientation classification.
        """
        # Initialize the KTP detector
        self.ktp_detector = KTPDetector(segment_model_path)

        # Initialize the corner detector
        self.corner_detector = CornerDetector(
            model_type=corner_model_config['model_type'],
            model_cfg=corner_model_config['model_cfg'],
            backend=corner_model_config['backend']
        )

        # Initialize the KTP orientation classifier
        self.ktp_oc = ImageOrientation(cls_model_path)

    def process_image(self, image_path, output_folder):
        """
        Process a single image: detect KTP, extract corners, warp the image, and save the result.

        Args:
            image_path (str): Path to the input image.
            output_folder (str): Path to save the warped images.
        """
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return

        # Detect KTP in the image
        try:
            ktp_regions = self.ktp_detector.detect_and_segment(image_path, conf=0.9)
        except ValueError as e:
            print(f"{image_path}: {e}")
            return

        # Define a padding value to avoid the bounding box being too tight
        padding = 10

        for i, ktp_region in enumerate(ktp_regions):
            bbox = ktp_region['bbox']
            mask = ktp_region['mask']

            #  Extract the region of interest (ROI) based on segmentation mask
            x1, y1, x2, y2 = bbox
            x_min = max(0, x1 - padding)
            y_min = max(0, y1 - padding)
            x_max = min(image.shape[1], x2 + padding)
            y_max = min(image.shape[0], y2 + padding)
            roi = image[y_min:y_max, x_min:x_max]
            mask_roi = mask[y_min:y_max, x_min:x_max] * 255

            # Detect corners within the ROI
            corners = self.corner_detector.detect_corners(roi, mask_roi)

            if len(corners) != 4:
                print(f"Invalid corners detected in image {image_path}, mask {i}. Skipping.")
                continue

            # Warp the image using detected corners
            warped_image = ImageWarper.warp_image(roi, corners, output_size=(856, 540))

            # Detect orientation
            predicted_orientation = self.ktp_oc.detect_orientation(warped_image)
            # Correct orientation
            corrected_image = self.ktp_oc.correct_orientation(warped_image, predicted_orientation)
            # Check image size
            h, w = corrected_image.shape[:2]
            if h > w:
                corrected_image = cv2.resize(corrected_image, (h, w), interpolation=cv2.INTER_AREA)

            # Display the corrected image for user inspection
            # cv2.imshow("Corrected Image", corrected_image)
            # key = cv2.waitKey(0)
            #
            # if key == ord('s'):  # Save the image if 's' is pressed
            #     output_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}_ktp_{i}.jpg")
            #     cv2.imwrite(output_path, corrected_image)
            #     print(f"Saved corrected image to: {output_path}")
            # elif key == ord('q'):  # Quit processing
            #     print("Processing aborted by user.")
            #     cv2.destroyAllWindows()
            #     return
            #
            # cv2.destroyAllWindows()

            # save corrected image
            output_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}_ktp_{i}.jpg")
            cv2.imwrite(output_path, corrected_image)
            print(f"Saved corrected image to: {output_path}")

    def process_folder(self, input_folder, output_folder):
        """
        Process all images in a folder.

        Args:
            input_folder (str): Path to the folder containing input images.
            output_folder (str): Path to save the warped images.
        """
        os.makedirs(output_folder, exist_ok=True)
        image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

        for image_file in image_files:
            image_path = os.path.join(input_folder, image_file)
            self.process_image(image_path, output_folder)


if __name__ == "__main__":
    # Configuration for the corner detector
    corner_model_config = {
        'model_type': ModelType.heatmap,    # Options: 'heatmap', 'point'
        'model_cfg': 'fastvit_t8',          # Options depend on model_type
        'backend': Backend.cpu              # Options: 'cpu', 'cuda'
    }

    # Paths input and output folder
    input_folder = "datasets/ktp_original"
    output_folder = "datasets/ktp_warped"

    # Initialize the processor
    processor = KTPImageProcessor(segment_model_path="models/YOLO11n-seg-ktp.pt",
                                  corner_model_config=corner_model_config,
                                  cls_model_path="models/YOLO11n-ktp-oc.pt")

    # Process the folder
    processor.process_folder(input_folder, output_folder)
