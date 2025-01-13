import cv2
import numpy as np


class ImageWarper:
    @staticmethod
    def order_points(pts):
        """
        Order the points in clockwise order: top-left, top-right, bottom-right, bottom-left.

        Args:
            pts (numpy.ndarray): Array of four points.

        Returns:
            numpy.ndarray: Ordered points.
        """
        rect = np.zeros((4, 2), dtype="float32")

        # Sum and difference of points to determine top-left and bottom-right
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left
        rect[2] = pts[np.argmax(s)]  # Bottom-right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right
        rect[3] = pts[np.argmax(diff)]  # Bottom-left

        return rect

    @staticmethod
    def warp_image(image, corners, output_size=(856, 540), buffer=10):
        """
        Warp the input image to a top-down view using the detected corners.

        Args:
            image (numpy.ndarray): Input image.
            corners (list or numpy.ndarray): Detected corner points [(x1, y1), (x2, y2), (x3, y3), (x4, y4)].
            output_size (tuple): Desired output size (width, height).
            buffer (int): Buffer size for padding when corners are outside the image.

        Returns:
            numpy.ndarray: Warped image.
        """
        if len(corners) != 4:
            raise ValueError("Exactly 4 corners are required for warping.")

        imH, imW = image.shape[:2]

        # Check if corners are inside the image boundaries
        if not (np.all(np.array(corners).min(axis=0) >= (0, 0)) and np.all(np.array(corners).max(axis=0) <= (imW, imH))):
            left_pad, top_pad, right_pad, bottom_pad = 0, 0, 0, 0

            rect = cv2.minAreaRect(np.array(corners).reshape((-1, 1, 2)))
            box = cv2.boxPoints(rect)
            box_corners = np.int32(box)

            box_x_min = np.min(box_corners[:, 0])
            box_x_max = np.max(box_corners[:, 0])
            box_y_min = np.min(box_corners[:, 1])
            box_y_max = np.max(box_corners[:, 1])

            # Calculate padding
            if box_x_min < 0:
                left_pad = abs(box_x_min) + buffer

            if box_x_max > imW:
                right_pad = (box_x_max - imW) + buffer

            if box_y_min < 0:
                top_pad = abs(box_y_min) + buffer

            if box_y_max > imH:
                bottom_pad = (box_y_max - imH) + buffer

            # Extend the image with padding
            image_extended = np.zeros((top_pad + bottom_pad + imH, left_pad + right_pad + imW, 3), dtype=image.dtype)
            image_extended[top_pad:top_pad + imH, left_pad:left_pad + imW, :] = image

            # Shift corners to match the extended image
            corners = np.array(corners, dtype=np.float32)
            corners[:, 0] += left_pad
            corners[:, 1] += top_pad

            image = image_extended

        # Order the points
        rect = ImageWarper.order_points(np.array(corners))

        (tl, tr, br, bl) = rect

        # Calculate the width of the new image
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))

        # Calculate the height of the new image
        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))

        # Set the destination points for the "birds-eye" view
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype="float32")

        # Compute the perspective transform matrix
        matrix = cv2.getPerspectiveTransform(rect, dst)

        # Perform the warp perspective
        warped = cv2.warpPerspective(image, matrix, (max_width, max_height))

        # Resize to output size
        if max_height > max_width and output_size[0] > output_size[1]:
            output_size = output_size[::-1]
        warped = cv2.resize(warped, output_size, interpolation=cv2.INTER_AREA)

        return warped


# Example usage
if __name__ == "__main__":
    # Read input image
    image_path = "data/processed/ktp_5.jpg"
    image = cv2.imread(image_path)

    # Example corners detected (replace with actual detected corners)
    corners = [(15, 19), (184, 30), (179, 134), (5, 125)]

    # Initialize warper and warp image
    warper = ImageWarper()
    warped_image = warper.warp_image(image, corners, output_size=(856, 540))

    # Display the result
    cv2.imshow("Warped Image", warped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the result
    output_path = "data/processed/ktp_warped.jpg"
    cv2.imwrite(output_path, warped_image)
    print(f"Warped image results saved at: {output_path}")
