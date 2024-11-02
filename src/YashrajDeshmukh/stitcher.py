import cv2
import os
import glob
import numpy as np
import logging

# Set up logging configuration
logging.basicConfig(
    filename='panorama_stitcher.log',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s:%(message)s'
)

def resize_image(image):
    """
    Resize the image based on its shape.
    
    Args:
    - image (numpy.ndarray): The input image.
    
    Returns:
    - numpy.ndarray: The resized image.
    """
    height, width, channels = image.shape
    
    if (height, width) == (2448, 3264) or (height, width) == (2658, 4000):
        # Reduce the size by a factor of 4
        new_height = height // 4
        new_width = width // 4
    elif (height, width) == (1329, 2000):
        # Reduce the size by half
        new_height = height // 2
        new_width = width // 2
    else:
        # Pass the image as it is
        return image
    
    # Log change in size
    logging.info("Resizing image from (%d, %d) to (%d, %d)", height, width, new_height, new_width)
    
    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return resized_image

class PanaromaStitcher():
    def __init__(self):
        pass

    def detect_and_match_features(self, image1, image2, nfeatures=100):
        """
        Detect keypoints and descriptors using SIFT and match them between two images.
        """
        try:
            sift = cv2.SIFT_create()  # Initialize SIFT detector
            keypoints_left, descriptors_left = sift.detectAndCompute(image1, None)
            keypoints_right, descriptors_right = sift.detectAndCompute(image2, None)

            # Initialize the BFMatcher
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            # Match descriptors
            matches = bf.match(descriptors_left, descriptors_right)
            # Sort matches based on distance
            matches = sorted(matches, key=lambda x: x.distance)

            # Get the keypoints corresponding to the matches
            correspondences = []
            for match in matches:
                correspondences.append((keypoints_left[match.queryIdx].pt, keypoints_right[match.trainIdx].pt))

            return np.array(correspondences)

        except Exception as e:
            logging.error("Error in detect_and_match_features: %s", e)
            raise

    def _compute_homography(self, correspondences):
        """
        Computes the homography matrix from 4 pairs of correspondences.
        """
        A = []
        for (x, y), (x_prime, y_prime) in correspondences:
            A.append([-x, -y, -1, 0, 0, 0, x_prime * x, x_prime * y, x_prime])
            A.append([0, 0, 0, -x, -y, -1, y_prime * x, y_prime * y, y_prime])

        A = np.array(A)

        # Compute SVD of A
        _, _, Vt = np.linalg.svd(A)
        # The last row of Vt (or last column of V) gives the solution
        H = Vt[-1].reshape(3, 3)
        return H / H[-1, -1]  # Normalize so that H[2, 2] = 1
    
    def get_homography_via_RANSAC(self, correspondences, trials=1000, threshold=5.0):
        """
        Estimates the homography matrix using RANSAC.
        
        Parameters:
            correspondences (ndarray): Array of point pairs [(x1, y1), (x2, y2)].
            trials (int): Number of RANSAC iterations to perform.
            threshold (float): Distance threshold to consider a point an inlier.
            
        Returns:
            best_H (ndarray): Best homography matrix found.
            best_inliers (list): List of inliers that agree with best_H.
        """
        try:
            if len(correspondences) < 4:
                raise ValueError("Not enough matches to compute homography.")

            max_inliers = []
            best_H = None

            num_points = correspondences.shape[0]

            for _ in range(trials):
                # Randomly select 4 correspondences to compute the homography
                sample_indices = np.random.choice(num_points, 4, replace=False)
                sample_correspondences = correspondences[sample_indices]

                # Compute homography for the sample
                H = self._compute_homography(sample_correspondences)

                # Calculate the number of inliers
                inliers = []
                for (x, y), (x_prime, y_prime) in correspondences:
                    # Transform (x, y) using the estimated homography
                    transformed_point = np.dot(H, np.array([x, y, 1]))
                    transformed_point /= transformed_point[2]  # Normalize

                    # Calculate the Euclidean distance between the transformed and actual points
                    distance = np.linalg.norm(transformed_point[:2] - np.array([x_prime, y_prime]))

                    if distance < threshold:
                        inliers.append(((x, y), (x_prime, y_prime)))

                # Update the best homography if the current set has more inliers
                if len(inliers) > len(max_inliers):
                    max_inliers = inliers
                    best_H = H

            # Recompute the homography using all inliers of the best model
            if best_H is not None:
                best_H = self._compute_homography(max_inliers)

            return best_H

        except Exception as e:
            logging.error("Error in get_homography: %s", e)
            raise

    def _bilinear_interpolation(self, image, x, y):
        """
        Performs bilinear interpolation for the given (x, y) coordinates.

        Parameters:
            image (ndarray): The input image.
            x (float): The x-coordinate (non-integer) in the image.
            y (float): The y-coordinate (non-integer) in the image.

        Returns:
            interpolated_value (ndarray): The interpolated RGB value.
        """
        x1, y1 = int(x), int(y)
        x2, y2 = min(x1 + 1, image.shape[1] - 1), min(y1 + 1, image.shape[0] - 1)

        # Calculate the distances
        dx, dy = x - x1, y - y1

        # Get pixel values from the four corners
        top_left = image[y1, x1]
        top_right = image[y1, x2]
        bottom_left = image[y2, x1]
        bottom_right = image[y2, x2]

        # Interpolate along x for the top and bottom edges
        top = (1 - dx) * top_left + dx * top_right
        bottom = (1 - dx) * bottom_left + dx * bottom_right

        # Interpolate along y between the top and bottom edges
        interpolated_value = (1 - dy) * top + dy * bottom

        return np.clip(interpolated_value, 0, 255).astype(np.uint8)
    
    def _compute_canvas_and_offset(self, image_left, image_right, H):
        """
        Computes a canvas size and offset for placing the transformed left image and the right image.

        Parameters:
            image_left (ndarray): The reference (left) image.
            image_right (ndarray): The right image.
            H (ndarray): The homography matrix mapping left image to the right image's coordinates.

        Returns:
            canvas_width (int): Width of the canvas to fit both images.
            canvas_height (int): Height of the canvas to fit both images.
            x_offset (int): X offset to translate images to non-negative coordinates.
            y_offset (int): Y offset to translate images to non-negative coordinates.
        """
        h_left, w_left = image_left.shape[:2]
        h_right, w_right = image_right.shape[:2]

        # Corners of the left image (to be transformed)
        left_corners = np.array([
            [0, 0, 1],                      # Top-left
            [w_left - 1, 0, 1],             # Top-right
            [w_left - 1, h_left - 1, 1],    # Bottom-right
            [0, h_left - 1, 1]              # Bottom-left
        ])

        # Apply homography to left image corners
        transformed_left_corners = []
        for corner in left_corners:
            transformed_corner = np.dot(H, corner)
            transformed_corner /= transformed_corner[2]             # Normalize by third coordinate
            transformed_left_corners.append(transformed_corner[:2])
        transformed_left_corners = np.array(transformed_left_corners)

        # Calculate bounding box for transformed left image corners
        x_min, y_min = np.min(transformed_left_corners, axis=0)
        x_max, y_max = np.max(transformed_left_corners, axis=0)

        # Use ceiling for offsets to handle fractional minimum values
        x_offset = int(np.ceil(-x_min) if x_min < 0 else 0)
        y_offset = int(np.ceil(-y_min) if y_min < 0 else 0)

        # Determine canvas width and height, including the dimensions of the right image
        canvas_width = int(np.ceil(max(x_max + x_offset, w_right + x_offset)))
        canvas_height = int(np.ceil(max(y_max + y_offset, h_right + y_offset)))

        return canvas_width, canvas_height, x_offset, y_offset

    def warp_image(self, image, H, output_shape, x_offset, y_offset):
        """
        Warps the input image using inverse mapping and computes the output shape based on the bounding box.

        Parameters:
            image (ndarray): The input image to be warped.
            H (ndarray): The homography matrix.
            output_shape (tuple): The dimensions (height, width) of the output canvas.
            x_offset (int): X offset to translate image coordinates.
            y_offset (int): Y offset to translate image coordinates.

        Returns:
            warped_image (ndarray): The warped image on the output canvas with transformed coordinates.
        """
        h_out, w_out = output_shape
        warped_image = np.zeros((h_out, w_out, 3), dtype=np.uint8)
        H_inv = np.linalg.inv(H)

        for y_out in range(h_out):
            for x_out in range(w_out):
                src_point = np.dot(H_inv, np.array([x_out - x_offset, y_out - y_offset, 1]))
                src_point /= src_point[2]

                x_src, y_src = src_point[:2]
                if 0 <= x_src < image.shape[1] and 0 <= y_src < image.shape[0]:
                    warped_image[y_out, x_out] = self._bilinear_interpolation(image, x_src, y_src)

        return warped_image

    def make_panaroma_for_images_in(self, path):
        """
        Load all images from the provided path and create a panorama.
        """
        try:
            all_images = sorted(glob.glob(path + os.sep + '*'))
            logging.info('Found %d images for stitching', len(all_images))

            if len(all_images) < 2:
                raise ValueError("Need at least two images to create a panorama.")

            base_image = resize_image(cv2.imread(all_images[0]))
            homography_matrix_list = []

            for i in range(1, len(all_images)):
                next_image = resize_image(cv2.imread(all_images[i]))

                # Detect and match features
                good_matches = self.detect_and_match_features(base_image, next_image)
                H = self.get_homography_via_RANSAC(good_matches)
                homography_matrix_list.append(H)

                # Compute canvas and offset
                canvas_width, canvas_height, x_offset, y_offset = self._compute_canvas_and_offset(base_image, next_image, H)
                output_shape = (canvas_height, canvas_width)

                # Warp base_image onto the larger canvas
                warped_base_image = self.warp_image(base_image, H, output_shape, x_offset, y_offset)

                # Place the right image onto the canvas
                panorama = warped_base_image
                panorama[y_offset:y_offset + next_image.shape[0], x_offset:x_offset + next_image.shape[1]] = next_image

                # Update base image for the next iteration
                base_image = panorama

            return base_image, homography_matrix_list

        except Exception as e:
            logging.error("Error in make_panaroma_for_images_in: %s", e)
            raise
