import cv2
import numpy as np
from docaligner import DocAligner, ModelType
from capybara import Backend


class CornerDetector:
    def __init__(self, model_type=ModelType.heatmap, model_cfg=None, backend=Backend.cpu):
        """
        Initialize the corner detector using the DocAligner library.

        Args:
            model_type (ModelType): Type of model to use (heatmap or point).
            model_cfg (str): Configuration for the model (e.g., 'lcnet100', 'fastvit_t8').
            backend (Backend): Computational backend ('cpu' or 'cuda').
        """
        if model_cfg is None:
            if model_type == ModelType.heatmap:
                model_cfg = 'fastvit_t8'  # Default for heatmap
            elif model_type == ModelType.point:
                model_cfg = 'lcnet050'  # Default for point
            else:
                raise ValueError("Unsupported model type.")

        self.model = DocAligner(model_type=model_type, model_cfg=model_cfg, backend=backend)

    def detect_corners(self, image, mask=None):
        """
        Detect the four corners of a document in the image.

        Args:
            image (numpy.ndarray): Input image.
            mask (numpy.ndarray, optional): Binary segmentation mask of the document.

        Returns:
            list: List of corner points [(x1, y1), (x2, y2), (x3, y3), (x4, y4)].
        """
        corners = self.model(image)

        if corners is None or len(corners) != 4:
            # Use mask based fall back logic if corners are not detected or are invalid
            if mask is None:
                raise ValueError("Failed to detect exactly 4 corners, and no mask provided for fallback.")

            # Add padding to image
            h, w = mask.shape[:2]
            mask_extended = np.zeros((2 * h, 2 * w), dtype=mask.dtype)
            mask_extended[(h // 2):(h + h // 2), (w // 2):(w + w // 2)] = mask.copy()

            # Edge Detection
            canny = cv2.Canny(mask_extended.astype(np.uint8), 255, 255)
            canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            contour, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            page = sorted(contour, key=cv2.contourArea, reverse=True)[0]

            epsilon = 0.02 * cv2.arcLength(page, True)
            corners = cv2.approxPolyDP(page, epsilon, True)
            corners = np.concatenate(corners).astype(np.float32)
            corners[:, 0] -= (w // 2)
            corners[:, 1] -= (h // 2)

            corners = corners.tolist()
            corners = self.order_points(corners)

        return [(int(point[0]), int(point[1])) for point in corners]

    def order_points(self, corners):
        """
        Order points in top-left, top-right, bottom-right, bottom-left order.

        Args:
            corners (list): List of corner points.

        Returns:
            list: Ordered corner points.
        """
        corners = sorted(corners, key=lambda x: (x[1], x[0]))  # Sort by y, then x
        top = sorted(corners[:2], key=lambda x: x[0])
        bottom = sorted(corners[2:], key=lambda x: x[0])
        return top + bottom


if __name__ == "__main__":
    # Path to the image and mask
    image_path = "data/processed/ktp_5.jpg"
    mask_path = "data/processed/mask_5.png"

    # Initialize the corner detector
    detector = CornerDetector()

    # Read and preprocess the image
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if mask_path else None

    try:
        # Detect corners
        corners = detector.detect_corners(image, mask)
        print(corners)

        # Draw the corners on the image
        for corner in corners:
            cv2.circle(image, corner, radius=5, color=(0, 0, 255), thickness=-1)

        cv2.imshow("image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save the visualization
        output_path = "data/processed/ktp_corners_detected.jpg"
        cv2.imwrite(output_path, image)
        print(f"Corner detection results saved at: {output_path}")

    except ValueError as e:
        print(e)
