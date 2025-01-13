import cv2
import numpy as np
from ultralytics import YOLO


class ImageOrientation:
    def __init__(self, model_path, device='cpu'):
        """
        Initialize the orientation detection model.

        Args:
            model_path (str): Path to the YOLO classification model.
            device (str): Device to run the model on ('cpu' or 'cuda').
        """
        self.model = YOLO(model_path)
        self.device = device

    def detect_orientation(self, image):
        """
        Detect the orientation of the image.

        Args:
            image (numpy.ndarray): Input image.

        Returns:
            str: Predicted orientation ('0', '90', '180', '270').
        """
        results = self.model.predict(image, device=self.device, verbose=False)
        predicted_label = results[0].names[results[0].probs.top1]
        return predicted_label


    def correct_orientation(self, image, orientation):
        """
        Rotate the image to the correct orientation based on the predicted label.

        Args:
            image (numpy.ndarray): Input image.
            orientation (str): Predicted orientation ('0', '90', '180', '270'), clockwise.

        Returns:
            numpy.ndarray: Correctly oriented image.
        """
        if orientation == '0':
            return image  # No rotation needed
        elif orientation == '90':
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif orientation == '180':
            return cv2.rotate(image, cv2.ROTATE_180)
        elif orientation == '270':
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        else:
            raise ValueError(f"Invalid orientation: {orientation}")


# Example usage
if __name__ == "__main__":
    model_path = "models/YOLO11n-ktp-oc.pt"  # Replace with your model path
    image_path = "data/processed/ktp_warped_2.jpg"  # Replace with your image path

    # Initialize orientation detector
    orientation_detector = ImageOrientation(model_path, device='cpu')

    # Read the input image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Detect orientation
    predicted_orientation = orientation_detector.detect_orientation(image)
    print(f"Predicted Orientation: {predicted_orientation}")

    # Correct orientation
    corrected_image = orientation_detector.correct_orientation(image, predicted_orientation)

    # Display and save the corrected image
    cv2.imshow("Corrected Image", corrected_image)
    corrected_image_path = "corrected_image.jpg"  # Replace with your desired output path
    cv2.imwrite(corrected_image_path, corrected_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the result
    output_path = "data/processed/corrected_ktp_orientation.jpg"
    cv2.imwrite(output_path, corrected_image)
    print(f"Warped image results saved at: {output_path}")
