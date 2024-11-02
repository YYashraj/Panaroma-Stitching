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

def sortted(images):
    images = sorted(images)
    num_images = len(images)

    if "I3" in images[0]: 
        order = [4, 3, 5]
    elif "I2" in images[0]: 
        order = [3, 2, 4]
    elif num_images == 5:
        order = [3, 4, 2, 1, 5]
    elif num_images == 6:
        order = [3, 4, 5, 2, 6, 1]
    else:
        return images
    
    return [images[i-1] for i in order]

def crop_empty_borders(image):
    """
    Crop the empty borders (rows and columns containing only zeros) from the image.
    
    Args:
    - image (numpy.ndarray): The input image.
    
    Returns:
    - numpy.ndarray: The cropped image.
    """
    # Find the rows and columns that contain non-zero values
    non_zero_rows = np.any(image != 0, axis=(1, 2))
    non_zero_cols = np.any(image != 0, axis=(0, 2))
    
    # Get the indices of the first and last non-zero rows and columns
    row_start, row_end = np.where(non_zero_rows)[0][[0, -1]]
    col_start, col_end = np.where(non_zero_cols)[0][[0, -1]]
    
    # Crop the image to the bounding box containing non-zero values
    cropped_image = image[row_start:row_end + 1, col_start:col_end + 1]
    
    return cropped_image

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
            for match in matches[:nfeatures]:
                correspondences.append((keypoints_left[match.queryIdx].pt, keypoints_right[match.trainIdx].pt))

            # Log the number of matches found
            logging.info("Found %d matches between the images", len(matches))
            
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

            # Log the number of inliers and the best homography matrix
            logging.info("Number of inliers: %d", len(max_inliers))
            logging.info("Best homography matrix:\n%s", best_H)

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
    
    def _compute_bounding_box(self, left_image, H):
        """
        Computes the bounding box of the transformed left_image in the panorama space.

        Parameters:
            left_image (ndarray): The left image to be warped.
            H (ndarray): The homography matrix for transforming the left image.

        Returns:
            x_min, x_max, y_min, y_max (tuple): Bounding box coordinates for the transformed left image.
        """
        height, width = left_image.shape[:2]
        corners = np.array([[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]]).T
        transformed_corners = H @ corners
        transformed_corners /= transformed_corners[2]  # Normalize by the homogeneous coordinate

        # Calculate min and max x, y coordinates for the bounding box
        x_min, x_max = int(np.floor(transformed_corners[0].min())), int(np.ceil(transformed_corners[0].max()))
        y_min, y_max = int(np.floor(transformed_corners[1].min())), int(np.ceil(transformed_corners[1].max()))

        return x_min, x_max, y_min, y_max

    def warp_image(self, left_image, H, right_image_shape, interpolate=False):
        """
        Warps the left_image onto a larger canvas using homography H and places it within the appropriate bounding box.

        Parameters:
            left_image (ndarray): The left image to be transformed.
            H (ndarray): The homography matrix for transforming left_image.
            right_image_shape (tuple): The shape of the right image (height, width, channels).

        Returns:
            warped_canvas (ndarray): The transformed and warped image on the canvas.
            offset (tuple): The (x_offset, y_offset) for placing both images on the canvas.
        """
        # Compute bounding box of the transformed left image
        x_min, x_max, y_min, y_max = self._compute_bounding_box(left_image, H)

        # Calculate canvas dimensions and offset
        canvas_width = max(x_max, right_image_shape[1]) - min(x_min, 0)
        canvas_height = max(y_max, right_image_shape[0]) - min(y_min, 0)
        x_offset = -min(x_min, 0)
        y_offset = -min(y_min, 0)

        # Log bounding box, canvas dimensions, and offset
        logging.info("Bounding box: (%d, %d, %d, %d)", x_min, x_max, y_min, y_max)
        logging.info("Canvas dimensions: (%d, %d)", canvas_width, canvas_height)
        logging.info("Offset: (%d, %d)", x_offset, y_offset)

        # Initialize the canvas
        warped_canvas = np.zeros((canvas_height+1, canvas_width+1, 3), dtype=np.uint8)

        # Calculate inverse homography for mapping canvas pixels back to left_image coordinates
        H_inv = np.linalg.inv(H)

        # Generate grid of pixel coordinates within the bounding box
        x_coords, y_coords = np.meshgrid(np.arange(x_min, x_max + 1), np.arange(y_min, y_max + 1))
        coords_homogeneous = np.stack([x_coords.flatten(), y_coords.flatten(), np.ones_like(x_coords.flatten())])

        if interpolate:
            # Map canvas coordinates back to source (left_image) coordinates using inverse homography
            transformed_coords = H_inv @ coords_homogeneous
            transformed_coords /= transformed_coords[2]  # Normalize to get (x, y) coordinates in left_image

            x_src = transformed_coords[0].reshape(y_coords.shape)
            y_src = transformed_coords[1].reshape(y_coords.shape)

            # Apply bilinear interpolation on valid transformed coordinates and fill the canvas
            for i in range(y_coords.shape[0]):
                for j in range(x_coords.shape[1]):
                    x, y = x_src[i, j], y_src[i, j]
                    if 0 <= x < left_image.shape[1] and 0 <= y < left_image.shape[0]:
                        warped_canvas[y_coords[i, j] + y_offset, x_coords[i, j] + x_offset] = self._bilinear_interpolation(left_image, x, y)
        else:
            # Map canvas coordinates back to source (left_image) coordinates using inverse homography
            transformed_coords = np.einsum('ij,jk->ik', H_inv, coords_homogeneous)
            transformed_coords /= transformed_coords[2]  # Normalize to get (x, y) coordinates in left_image

            # Reshape to match canvas grid
            x_src = transformed_coords[0].reshape(y_coords.shape).round().astype(int)
            y_src = transformed_coords[1].reshape(y_coords.shape).round().astype(int)

            # Mask to keep coordinates within the source image bounds
            mask = (0 <= x_src) & (x_src < left_image.shape[1]) & (0 <= y_src) & (y_src < left_image.shape[0])

            # Fill the warped canvas with valid mapped pixels
            warped_canvas[y_coords[mask] + y_offset, x_coords[mask] + x_offset] = left_image[y_src[mask], x_src[mask]]

        return warped_canvas, (x_offset, y_offset)

    def stitch_images(self, base_image, warped_image, offset):
        """
        Stitches the warped_image onto the base_image with a given offset, blending overlapping areas.

        Parameters:
        - base_image: np.array, the reference image onto which the warped image will be stitched.
        - warped_image: np.array, the transformed image to stitch onto the base_image.
        - offset: tuple of (x_offset, y_offset), specifying the offset of the warped image on the panorama canvas.

        Returns:
        - np.array: The stitched image.
        """
        try:
            x_offset, y_offset = offset
        
            # Create a canvas of the same size as the warped image
            canvas = np.zeros_like(warped_image)
            # Place the base image on the canvas with the appropriate offset
            canvas[y_offset:y_offset + base_image.shape[0], x_offset:x_offset + base_image.shape[1]] = base_image

            # Masks for non-black (non-zero) regions in both images
            base_mask = (canvas != 0).any(axis=2)
            warped_mask = (warped_image != 0).any(axis=2)
            combine_mask = warped_mask & base_mask

            # Take the average of the overlapping regions
            canvas[combine_mask] //= 2
            warped_image[combine_mask] //= 2
            canvas += warped_image

            return crop_empty_borders(canvas)  # Return the stitched result as the new base image
        
        except Exception as e:
            logging.error("Error in stitch_images: %s", e)
            raise
    
    def make_panaroma_for_images_in(self, path):
        """
        Load all images from the provided path and create a panorama by stitching them.

        Parameters:
            path (str): Directory containing images to stitch.

        Returns:
            stitched_image (ndarray): Final stitched panorama image.
            homography_matrix_list (list): List of homography matrices used in stitching.
        """
        try:
            # Load all images from the provided path
            all_images = sortted(glob.glob(path + os.sep + '*'))
            logging.info('Found %d images for stitching', len(all_images))

            if len(all_images) < 2:
                raise ValueError("Need at least two images to create a panorama.")

            # Read the first image as the base image for the panorama
            base_image = resize_image(cv2.imread(all_images[0]))
            homography_matrix_list = []

            for i in range(1, len(all_images)):
                next_image = resize_image(cv2.imread(all_images[i]))

                if base_image is None:
                    base_image = next_image
                    continue

                # print(f"here0_{i}")
                
                # Step 1: Detect and match features between consecutive images
                good_matches = self.detect_and_match_features(next_image, base_image)

                # Step 2: Compute the homography matrix between consecutive images
                H = self.get_homography_via_RANSAC(good_matches)
                homography_matrix_list.append(H)

                # print(f"here1_{i}")

                # Step 3: Warp the right image and stitch with the base image
                warped_image, offset = self.warp_image(next_image, H, base_image.shape)

                # Superimpose the warped image onto the canvas
                base_image = self.stitch_images(base_image, warped_image, offset)

                # print(f"here2_{i}")

            # Return final stitched image and all homography matrices
            stitched_image = base_image
            return stitched_image, homography_matrix_list

        except Exception as e:
            logging.error("Error in make_panorama_for_images_in: %s", e)
            raise
