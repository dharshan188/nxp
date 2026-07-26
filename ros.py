# Copyright 2024-2026 NXP
# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2
import math
import time
from synapse_msgs.msg import EdgeVectors

QOS_PROFILE_DEFAULT = 10
PI = math.pi

RED_COLOR = (0, 0, 255)
BLUE_COLOR = (255, 0, 0)
GREEN_COLOR = (0, 255, 0)

# HINT: You can adjust what percentage of the image from the bottom is analyzed.
# Lower portions are closer to the buggy, while upper portions see further ahead.
VECTOR_IMAGE_HEIGHT_PERCENTAGE = 0.55
VECTOR_MAGNITUDE_MINIMUM = 2.25


class EdgeVectorsPublisher(Node):
    """
    ROS 2 Node that processes raw camera images to detect the lane edges (left/right bounds).
    It publishes the detected boundaries as synapse_msgs/EdgeVectors.
    """
    def __init__(self):
        super().__init__('edge_vectors_publisher')

        # Subscription for camera images.
        self.subscription_camera = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.camera_image_callback,
            QOS_PROFILE_DEFAULT)

        # Publisher for edge vectors.
        self.publisher_edge_vectors = self.create_publisher(
            EdgeVectors,
            '/edge_vectors',
            QOS_PROFILE_DEFAULT)

        # Publisher for thresh image (for debugging thresholding/segmentation).
        self.publisher_thresh_image = self.create_publisher(
            CompressedImage,
            "/debug_images/thresh_image",
            QOS_PROFILE_DEFAULT)

        # Publisher for vector image (for debugging vector drawing).
        self.publisher_vector_image = self.create_publisher(
            CompressedImage,
            "/debug_images/vector_image",
            QOS_PROFILE_DEFAULT)

        self.image_height = 0
        self.image_width = 0
        self.lower_image_height = 0
        self.upper_image_height = 0

        # ------------------ Thresholding parameters ------------------
        self.declare_parameter('threshold_black_fallback', 90)
        self.declare_parameter('use_otsu', True)
        self.declare_parameter('min_contour_area', 15.0)
        self.declare_parameter('max_aspect_ratio_for_noise', 0.0)
        self.declare_parameter('debug_log_period', 1.0)

        self.threshold_black_fallback = self.get_parameter('threshold_black_fallback').value
        self.use_otsu = self.get_parameter('use_otsu').value
        self.min_contour_area = self.get_parameter('min_contour_area').value
        self.debug_log_period = self.get_parameter('debug_log_period').value

        self.last_debug_log_time = 0.0

        self.get_logger().info(
            "EdgeVectorsPublisher initialized. "
            f"use_otsu={self.use_otsu} "
            f"threshold_black_fallback={self.threshold_black_fallback} "
            f"min_contour_area={self.min_contour_area}"
        )

    def publish_debug_image(self, publisher, image):
        """Helper function to publish OpenCV debug images to ROS topics."""
        message = CompressedImage()
        _, encoded_data = cv2.imencode('.jpg', image)
        message.format = "jpeg"
        message.data = encoded_data.tobytes()
        publisher.publish(message)

    def get_vector_angle_in_radians(self, vector):
        """Calculates the slope angle of a vector in radians."""
        if ((vector[0][0] - vector[1][0]) == 0):  # Prevent division by zero
            theta = PI / 2
        else:
            slope = (vector[1][1] - vector[0][1]) / (vector[0][0] - vector[1][0])
            theta = math.atan(slope)
        return theta

    def compute_binary_threshold(self, gray):
        """
        Produces a binary image where lane-boundary (black) pixels are 255.
        Uses Otsu's method for adaptive thresholding with morphological cleanup.
        """
        if self.use_otsu:
            otsu_val, thresh = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Guard against degenerate Otsu values
            if otsu_val < 10 or otsu_val > 245:
                thresh = cv2.threshold(
                    gray, self.threshold_black_fallback, 255, cv2.THRESH_BINARY_INV)[1]
        else:
            thresh = cv2.threshold(
                gray, self.threshold_black_fallback, 255, cv2.THRESH_BINARY_INV)[1]

        # Morphological cleanup
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        return thresh

    def compute_vectors_from_image(self, image, thresh):
        """
        Analyzes the binary threshold image and extracts left and right lane edge vectors.
        This uses basic contour finding and coordinates calculations.
        """
        # Find contours around black edge stripes detected in the binary threshold image.
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

        num_contours = len(contours)
        num_valid_vectors = 0

        vectors = []
        for i in range(num_contours):
            coordinates = contours[i][:, 0, :]

            # Skip if contour has no valid coordinates
            if len(coordinates) == 0:
                continue

            # Improved contour width filter
            x_min = np.min(coordinates[:, 0])
            x_max = np.max(coordinates[:, 0])
            width = x_max - x_min
            height = np.max(coordinates[:, 1]) - np.min(coordinates[:, 1])

            # Reject contours that are extremely wide compared to their height
            if width > 0.75 * self.image_width and width > height:
                continue

            # ------------------ Area filter ------------------
            area = cv2.contourArea(contours[i])
            if area < self.min_contour_area:
                continue

            # Get coordinates representing the boundaries of the contour
            min_y_value = np.min(coordinates[:, 1])
            max_y_value = np.max(coordinates[:, 1])

            min_y_coords = np.array(coordinates[coordinates[:, 1] == min_y_value])
            max_y_coords = np.array(coordinates[coordinates[:, 1] == max_y_value])

            min_y_coord = min_y_coords[0].copy()
            max_y_coord = max_y_coords[0].copy()

            # Calculate contour vector magnitude
            magnitude = np.linalg.norm(min_y_coord - max_y_coord)
            if magnitude > VECTOR_MAGNITUDE_MINIMUM:
                num_valid_vectors += 1

                # Calculate distance from the camera center at the bottom of the crop
                rover_point = [self.image_width / 2, self.lower_image_height]
                middle_point = (min_y_coord + max_y_coord) / 2
                distance = np.linalg.norm(middle_point - rover_point)

                # Correct point coordinates based on vector slope angle
                angle = self.get_vector_angle_in_radians([min_y_coord, max_y_coord])
                if angle > 0:
                    min_y_coord[0] = np.max(min_y_coords[:, 0])
                else:
                    max_y_coord[0] = np.max(max_y_coords[:, 0])

                # Store vectors with distance metadata for sorting
                vectors.append([list(min_y_coord), list(max_y_coord), distance])

                # Draw all detected raw (filtered-in) vectors in blue on the debug image.
                cv2.line(image, tuple(min_y_coord), tuple(max_y_coord), BLUE_COLOR, 2)

        return vectors, image, num_contours, num_valid_vectors

    def process_image_for_edge_vectors(self, image):
        """
        Applies preprocessing (Grayscale + robust Thresholding + Morphology) and
        extracts lane vectors, with debug logging at each stage.
        """
        self.image_height, self.image_width, _ = image.shape
        self.lower_image_height = int(self.image_height * VECTOR_IMAGE_HEIGHT_PERCENTAGE)
        self.upper_image_height = int(self.image_height - self.lower_image_height)

        # 1. Convert to Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 2. Robust binary thresholding + morphology
        thresh = self.compute_binary_threshold(gray)

        # 3. Crop the image to focus on the lower section close to the buggy.
        thresh_cropped = thresh[self.image_height - self.lower_image_height:]
        image_cropped = image[self.image_height - self.lower_image_height:].copy()

        white_pixel_count = int(np.count_nonzero(thresh_cropped))

        # 4. Compute vectors from the binary image contours
        vectors, debug_img, num_contours, num_valid_vectors = self.compute_vectors_from_image(
            image_cropped, thresh_cropped)

        # 5. Sort vectors based on distance from buggy
        vectors = sorted(vectors, key=lambda x: x[2])

        # 6. Split vectors based on left/right halves of the image
        half_width = self.image_width / 2
        vectors_left = [v for v in vectors if ((v[0][0] + v[1][0]) / 2) < half_width]
        vectors_right = [v for v in vectors if ((v[0][0] + v[1][0]) / 2) >= half_width]

        final_vectors = []
        selected_left = None
        selected_right = None

        # Select the closest vector from each side (left/right)
        for side_name, side_vectors in [('left', vectors_left), ('right', vectors_right)]:
            if len(side_vectors) > 0:
                best_vector = side_vectors[0]
                # Draw the selected key lane vector in green on the debug image
                cv2.line(debug_img, tuple(best_vector[0]), tuple(best_vector[1]), GREEN_COLOR, 2)

                # Transform coordinates back to the original uncropped image space
                best_vector[0][1] += self.upper_image_height
                best_vector[1][1] += self.upper_image_height
                final_vectors.append(best_vector[:2])

                if side_name == 'left':
                    selected_left = best_vector[:2]
                else:
                    selected_right = best_vector[:2]

        # Publish visual debugging images
        self.publish_debug_image(self.publisher_thresh_image, thresh_cropped)
        self.publish_debug_image(self.publisher_vector_image, debug_img)

        # ------------------ Debug logging (throttled) ------------------
        now = time.time()
        if now - self.last_debug_log_time >= self.debug_log_period:
            self.last_debug_log_time = now
            self.get_logger().info(
                f"white_pixels={white_pixel_count} "
                f"contours={num_contours} "
                f"valid_vectors={num_valid_vectors} "
                f"final_vectors={len(final_vectors)} "
                f"selected_left={selected_left is not None} "
                f"selected_right={selected_right is not None}"
            )
            if white_pixel_count == 0:
                self.get_logger().warn(
                    "Thresholded image has 0 white pixels -- lane boundaries not "
                    "being isolated. Check lighting/threshold parameters.")
            elif num_contours == 0:
                self.get_logger().warn(
                    "White pixels present but 0 contours found -- check ROI crop "
                    "and morphology kernel size.")
            elif num_valid_vectors == 0:
                self.get_logger().warn(
                    "Contours found but none passed magnitude/area filtering -- "
                    "consider lowering VECTOR_MAGNITUDE_MINIMUM or min_contour_area.")

        return final_vectors

    def camera_image_callback(self, message):
        """Processes incoming camera frames and publishes detected EdgeVectors."""
        # Convert compressed image message to OpenCV format
        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            self.get_logger().warn("Failed to decode incoming compressed image.")
            return

        vectors = self.process_image_for_edge_vectors(image)

        # Construct and publish the ROS 2 EdgeVectors message
        vectors_message = EdgeVectors()
        vectors_message.image_height = image.shape[0]
        vectors_message.image_width = image.shape[1]
        vectors_message.vector_count = 0

        # Vector 1 (usually representing Left boundary)
        if len(vectors) > 0:
            vectors_message.vector_1[0].x = float(vectors[0][0][0])
            vectors_message.vector_1[0].y = float(vectors[0][0][1])
            vectors_message.vector_1[1].x = float(vectors[0][1][0])
            vectors_message.vector_1[1].y = float(vectors[0][1][1])
            vectors_message.vector_count += 1

        # Vector 2 (usually representing Right boundary)
        if len(vectors) > 1:
            vectors_message.vector_2[0].x = float(vectors[1][0][0])
            vectors_message.vector_2[0].y = float(vectors[1][0][1])
            vectors_message.vector_2[1].x = float(vectors[1][1][0])
            vectors_message.vector_2[1].y = float(vectors[1][1][1])
            vectors_message.vector_count += 1

        self.publisher_edge_vectors.publish(vectors_message)


def main(args=None):
    rclpy.init(args=args)
    node = EdgeVectorsPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
