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

            # Ensure that the matches are unique
            filtered_matches = []
            filtered_matches.append(matches[0])
            for match in matches[1:nfeatures]:
                if match.queryIdx not in [m.queryIdx for m in filtered_matches] and match.trainIdx not in [m.trainIdx for m in filtered_matches]:
                    filtered_matches.append(match)

            # Get the keypoints corresponding to the matches
            correspondences = []
            for match in filtered_matches:
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

    def _compute_bounding_box(self, image, H):
        """
        Computes the bounding box of the transformed image using the homography matrix.

        Parameters:
            image (ndarray): The input image.
            H (ndarray): The homography matrix.

        Returns:
            x_min, y_min, x_max, y_max: Coordinates of the bounding box.
        """
        h, w = image.shape[:2]
        
        # Define the four corners of the image
        corners = np.array([
            [0, 0, 1],          # Top-left
            [w - 1, 0, 1],      # Top-right
            [w - 1, h - 1, 1],  # Bottom-right
            [0, h - 1, 1]       # Bottom-left
        ])

        # Apply the homography matrix to the corners
        transformed_corners = []
        for corner in corners:
            transformed_corner = np.dot(H, corner)
            transformed_corner /= transformed_corner[2]  # Normalize by third coordinate
            transformed_corners.append(transformed_corner[:2])
        
        # Get min and max x, y coordinates from the transformed corners
        transformed_corners = np.array(transformed_corners)
        x_min, y_min = np.min(transformed_corners, axis=0)
        x_max, y_max = np.max(transformed_corners, axis=0)

        return int(x_min), int(y_min), int(x_max), int(y_max)

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
    
    def warp_image(self, image, H, output_shape):
        """
        Warps the input image using inverse mapping and computes the output shape based on the bounding box.

        Parameters:
            image (ndarray): The input image to be warped.
            H (ndarray): The homography matrix.

        Returns:
            warped_image (ndarray): The warped image with the correct bounding box and transformed coordinates.
        """
        # Compute the bounding box of the transformed image
        x_min, y_min, x_max, y_max = self._compute_bounding_box(image, H)
        
        # Calculate output dimensions based on the bounding box
        h_out, w_out = int(y_max - y_min), int(x_max - x_min)
        
        # Create an empty output image with calculated dimensions
        warped_image = np.zeros((h_out, w_out, 3), dtype=np.uint8)

        # Calculate the inverse of the homography matrix
        H_inv = np.linalg.inv(H)

        # Iterate over each pixel in the bounding box
        for y_out in range(h_out):
            for x_out in range(w_out):
                # Map (x_out, y_out) to the source image using the inverse homography
                src_point = np.dot(H_inv, np.array([x_out + x_min, y_out + y_min, 1]))
                src_point /= src_point[2]  # Normalize to get (x, y) in source image coordinates

                x_src, y_src = src_point[:2]

                # Perform bilinear interpolation if the mapped point is within the source image
                if 0 <= x_src < image.shape[1] and 0 <= y_src < image.shape[0]:
                    warped_image[y_out, x_out] = self._bilinear_interpolation(image, x_src, y_src)

        return warped_image

    def make_panaroma_for_images_in(self, path):
        """
        Load all images from the provided path and create a panorama.
        """
        try:
            # Load all images from the provided path
            imf = path
            all_images = sorted(glob.glob(imf + os.sep + '*'))
            logging.info('Found %d images for stitching', len(all_images))

            if len(all_images) < 2:
                raise ValueError("Need at least two images to create a panorama.")

            # Read the first image as the base image for the panorama
            base_image = cv2.imread(all_images[0])
            homography_matrix_list = []

            for i in range(1, len(all_images)):
                next_image = cv2.imread(all_images[i])

                # Step 1: Detect and match features between consecutive images
                good_matches = self.detect_and_match_features(base_image, next_image)

                # Step 2: Compute the homography matrix between consecutive images
                H = self.get_homography_via_RANSAC(good_matches)
                homography_matrix_list.append(H)

                # Step 3: Warp and stitch images
                base_image = self.warp_and_stitch(base_image, next_image, H)

            # Return final stitched image and all homography matrices
            stitched_image = base_image
            return stitched_image, homography_matrix_list

        except Exception as e:
            logging.error("Error in make_panaroma_for_images_in: %s", e)
            raise
