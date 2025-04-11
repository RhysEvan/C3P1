from Classes.Structured_Light import Structured_Light
from StructuredLight.Graycode.decoder import *
import glob
import h5py
import numpy as np
import os
from scipy.spatial import cKDTree
from core_toolbox_python.Plucker.Line import *
import math
import logging
import sys
sys.path.append('Classes/GrayCode/process_frames_cpp')
from Classes.proces_frames_cpp import process_frames_module as process_frames_module
from Classes.correspondence import correspondence as correspondence
from Classes.combined_corr_process import combined as combined

import multiprocessing
#from DataCapture import graycode_data_capture
from StructuredLight.Graycode.projector_image import *
from Classes.GrayCode_additional_methods import _process_neighbor_search_numba, process_frames_numba,find_valid_neighbor_numba
from Classes.GrayCode_additional_methods_torch import process_frames_optimized_pytorch


class GrayCodeMultiCam(Structured_Light):
    """
    \ingroup structured_light
    \brief A class to represent a Gray Code structured light system for multiple cameras.

    This class is a child of the Structured_Light class and includes methods for generating patterns, decoding patterns, and ray matching.
    """
    process_frames_numba = process_frames_numba
    _process_neighbor_search_numba = _process_neighbor_search_numba
    find_valid_neighbor_numba = find_valid_neighbor_numba
    process_frames_optimized_pytorch = process_frames_optimized_pytorch

    def __init__(self):
        """
        \ingroup structured_light
        \brief Initializes the GrayCodeMultiCam class.
        """
        super().__init__()
        print('GrayCodeMultiCam initialized')
        self.Decoder = Decode_Gray()
        self.Images = []

    def generate_patterns(self,width=720, height=1280, projector_directory="projector_pattern"):
        """
        \ingroup structured_light
        \brief Generates Gray Code patterns for projection.

        \return A list of generated patterns.
        """
        # Implement pattern generation logic here

        #check if the directory exists
        if not os.path.exists(projector_directory):
            os.makedirs(projector_directory)
        graycode = ProjectionPattern(width, height, projector_directory)
        graycode.generate_images()


        pass

    def decode_patterns_from_image_folder(self, folder_path):
        """
        \ingroup structured_light
        \brief Decodes the captured Gray Code patterns to obtain the corresponding code for each pixel.

        :param folder_path: The path to the folder containing the captured images.
        :return: Decoded patterns.
        """
        # Implement pattern decoding logic here
        self.Images = self.load_images_into_dict(folder_path)

        self.decode_patterns()
        return self.left_horizontal_decoded_image, self.left_vertical_decoded_image, self.right_horizontal_decoded_image, self.right_vertical_decoded_image
    def decode_patterns_from_file(self, captured_images_path):
        """
        \ingroup structured_light
        \brief Decodes the captured Gray Code patterns to obtain the corresponding code for each pixel.

        \param file_path: The path to the captured images after projecting the patterns.
        \return Decoded patterns.
        """
        # Implement pattern decoding logic here

        self.Images = self.load_h5(captured_images_path)
        self.decode_patterns()
        return self.left_horizontal_decoded_image, self.left_vertical_decoded_image, self.right_horizontal_decoded_image, self.right_vertical_decoded_image


        pass

    def decode_patterns(self, captured_images = None):
        """
        \ingroup structured_light
        \brief Decodes the captured Gray Code patterns to obtain the corresponding code for each pixel.

        \param captured_images: A list of captured images after projecting the patterns.
        \return Decoded patterns.
        """
        # Implement pattern decoding logic here
        if captured_images is None:
            captured_images = self.Images
        for i, item in enumerate(captured_images ):
            logging.info("decoding scene ... " + str(i))
            self.left_horizontal_decoded_image, self.left_vertical_decoded_image = self.Decoder.scene_decoder(item, "L")
            self.right_horizontal_decoded_image, self.right_vertical_decoded_image = self.Decoder.scene_decoder(item, "R")
            logging.info("decoding done")
        pass
        return self.left_horizontal_decoded_image, self.left_vertical_decoded_image, self.right_horizontal_decoded_image, self.right_vertical_decoded_image

    def process_frames(self,inverse_left_cam, inverse_right_cam, shape_left, shape_right):
        shape_invers_left = inverse_left_cam.shape
        shape_invers_right = inverse_right_cam.shape
        min_rows = min(shape_invers_left[0], shape_invers_right[0]) - 1
        min_cols = min(shape_invers_left[1], shape_invers_right[1]) - 1

        index = 0
        indexl = []
        indexr = []
        for ii in range(0, min(shape_invers_left[0], shape_invers_right[0]) - 1):
            for jj in range(0, min(shape_invers_left[1], shape_invers_right[1]) - 1):
                flag = False
                index = index + 1
                leftframe = inverse_left_cam[ii][jj]
                rightframe = inverse_right_cam[ii][jj]

                if leftframe[0] != 0 and leftframe[1] != 0 and rightframe[0] != 0 and rightframe[1] != 0:
                    width = shape_left[1]
                    indexl.append((leftframe[1]) * width + (leftframe[0]))
                    width = shape_right[1]
                    indexr.append((rightframe[1]) * width + (rightframe[0] ))                    # self.array_colorizer(img, leftframe)

                elif leftframe[0] != 0 and leftframe[1] != 0 and rightframe[0] == 0 and rightframe[1] == 0:

                    rightframe = inverse_right_cam[ii][jj + 1]

                    if rightframe[0] != 0 and rightframe[1] != 0:
                        width = shape_left[1]
                        indexl.append((leftframe[1]) * width + (leftframe[0]))
                        width = shape_right[1]
                        indexr.append((rightframe[1]) * width + (rightframe[0]))
                        flag = True
                        # self.array_colorizer(img, leftframe)

                    rightframe = inverse_right_cam[ii][jj - 1]

                    if rightframe[0] != 0 and rightframe[1] != 0 and flag == False:
                        width = shape_left[1]
                        indexl.append((leftframe[1]) * width + (leftframe[0]))
                        width = shape_right[1]
                        indexr.append((rightframe[1]) * width + (rightframe[0]))
                        flag = True
                        # self.array_colorizer(img, leftframe)

                    rightframe = inverse_right_cam[ii + 1][jj]

                    if rightframe[0] != 0 and rightframe[1] != 0 and flag == False:
                        width = shape_left[1]
                        indexl.append((leftframe[1]) * width + (leftframe[0]))
                        width = shape_right[1]
                        indexr.append((rightframe[1]) * width + (rightframe[0]))
                        flag = True
                        # self.array_colorizer(img, leftframe)

                    rightframe = inverse_right_cam[ii - 1][jj]

                    if rightframe[0] != 0 and rightframe[1] != 0 and flag == False:
                        width = shape_left[1]
                        indexl.append((leftframe[1]) * width + (leftframe[0]))
                        width = shape_right[1]
                        indexr.append((rightframe[1]) * width + (rightframe[0]))
                        flag = True
                        # self.array_colorizer(img, leftframe)

                elif leftframe[0] == 0 and leftframe[1] == 0 and rightframe[0] != 0 and rightframe[1] != 0:

                    leftframe = inverse_left_cam[ii][jj + 1]

                    if leftframe[0] != 0 and leftframe[1] != 0:
                        width = shape_left[1]
                        indexl.append((leftframe[1]) * width + (leftframe[0]))
                        width = shape_right[1]
                        indexr.append((rightframe[1]) * width + (rightframe[0]))
                        flag = True
                        # self.array_colorizer(img, leftframe)

                    leftframe = inverse_left_cam[ii][jj - 1]

                    if leftframe[0] != 0 and leftframe[1] != 0 and flag == False:
                        width = shape_left[1]
                        indexl.append((leftframe[1]) * width + (leftframe[0]))
                        width = shape_right[1]
                        indexr.append((rightframe[1]) * width + (rightframe[0]))
                        flag = True
                        # self.array_colorizer(img, leftframe)

                    leftframe = inverse_left_cam[ii + 1][jj]

                    if leftframe[0] != 0 and leftframe[1] != 0 and flag == False:
                        width = shape_left[1]
                        indexl.append((leftframe[1]) * width + (leftframe[0]))
                        width = shape_right[1]
                        indexr.append((rightframe[1]) * width + (rightframe[0]))
                        flag = True
                        # self.array_colorizer(img, leftframe)

                    leftframe = inverse_left_cam[ii - 1][jj]

                    if leftframe[0] != 0 and leftframe[1] != 0 and flag == False:
                        width = shape_left[1]
                        indexl.append((leftframe[1]) * width + (leftframe[0]))
                        width = shape_right[1]
                        indexr.append((rightframe[1]) * width + (rightframe[0]))
                        flag = True
                        # self.array_colorizer(img, leftframe)
        return indexl, indexr

    def process_frames2(self, inverse_left_cam, inverse_right_cam, shape_left, shape_right):
        shape_invers_left = inverse_left_cam.shape
        shape_invers_right = inverse_right_cam.shape
        min_rows = min(shape_invers_left[0], shape_invers_right[0]) - 1
        min_cols = min(shape_invers_left[1], shape_invers_right[1]) - 1

        indexl = []
        indexr = []

        for ii in range(min_rows):
            for jj in range(min_cols):
                leftframe = inverse_left_cam[ii, jj]
                rightframe = inverse_right_cam[ii, jj]

                if leftframe[0] != 0 and leftframe[1] != 0 and rightframe[0] != 0 and rightframe[1] != 0:
                    width_left = shape_left[1]
                    width_right = shape_right[1]
                    indexl.append(leftframe[1] * width_left + leftframe[0])
                    indexr.append(rightframe[1] * width_right + rightframe[0])
                elif leftframe[0] != 0 and leftframe[1] != 0:
                    rightframe = self.find_non_zero_neighbor(inverse_right_cam, ii, jj)
                    if rightframe is not None:
                        width_left = shape_left[1]
                        width_right = shape_right[1]
                        indexl.append(leftframe[1] * width_left + leftframe[0])
                        indexr.append(rightframe[1] * width_right + rightframe[0])
                elif rightframe[0] != 0 and rightframe[1] != 0:
                    leftframe = self.find_non_zero_neighbor(inverse_left_cam, ii, jj)
                    if leftframe is not None:
                        width_left = shape_left[1]
                        width_right = shape_right[1]
                        indexl.append(leftframe[1] * width_left + leftframe[0])
                        indexr.append(rightframe[1] * width_right + rightframe[0])

        return indexl, indexr

        return indexl, indexr

    def find_valid_neighbor(self, inverse_map: np.ndarray, r: int, c: int, sentinel_value: int):
        """
        Searches the 8 neighbors of (r, c) in the inverse_map for a valid entry.

        Args:
            inverse_map: The 3D inverse map array (H', W', 2).
            r: Row index in the inverse map.
            c: Column index in the inverse map.
            sentinel_value: The value indicating an invalid/unmapped entry.

        Returns:
            The [col, row] ndarray of the first valid neighbor found, or None.
        """
        map_rows, map_cols = inverse_map.shape[:2]
        # Define neighbor offsets (8-connectivity, order can matter if multiple are valid)
        # Check immediate neighbors first
        neighbor_offsets = [
            (-1, 0), (1, 0), (0, -1), (0, 1), # Cardinal
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # Diagonal
        ]

        for dr, dc in neighbor_offsets:
            nr, nc = r + dr, c + dc
            # Check bounds
            if 0 <= nr < map_rows and 0 <= nc < map_cols:
                neighbor_val = inverse_map[nr, nc]
                # Check if valid (checking row or col is sufficient if [sentinel, sentinel] is used)
                if neighbor_val[1] != sentinel_value: # Assuming checking row is enough
                    return neighbor_val # Return the [col, row] pair
        return None # No valid neighbor found

    def process_frames_optimized(self,
                                inverse_left_cam: np.ndarray,
                                inverse_right_cam: np.ndarray,
                                shape_left,
                                shape_right,
                                sentinel_value: int = 0) :
        """
        Processes inverse maps to find corresponding flattened indices in original frames,
        using neighbor search for missing correspondences, optimized with NumPy.

        Args:
            inverse_left_cam: Inverse map for the left camera [col, row].
            inverse_right_cam: Inverse map for the right camera [col, row].
            shape_left: Original shape (H, W) of the left image.
            shape_right: Original shape (H, W) of the right image.
            sentinel_value: Value indicating unmapped points in the inverse maps.

        Returns:
            A tuple containing:
            - final_indexl: NumPy array of flattened indices for the left original image.
            - final_indexr: NumPy array of flattened indices for the right original image.
        """
        shape_invers_left = inverse_left_cam.shape
        shape_invers_right = inverse_right_cam.shape

        # Determine the common region to process based on inverse map dimensions
        min_rows = min(shape_invers_left[0], shape_invers_right[0])
        min_cols = min(shape_invers_left[1], shape_invers_right[1])

        # --- Crop maps to the common region ---
        left_map_crop = inverse_left_cam[:min_rows, :min_cols]
        right_map_crop = inverse_right_cam[:min_rows, :min_cols]

        # --- Extract original column and row, handling the sentinel ---
        # Shape: (min_rows, min_cols)
        left_orig_col = left_map_crop[..., 0]
        left_orig_row = left_map_crop[..., 1]
        right_orig_col = right_map_crop[..., 0]
        right_orig_row = right_map_crop[..., 1]

        # --- Create masks for valid entries ---
        # Valid if the row value is not the sentinel (assuming [sentinel, sentinel] for invalid)
        valid_left_mask = left_orig_row != sentinel_value
        valid_right_mask = right_orig_row != sentinel_value

        # --- Case 1: Both left and right mappings are valid ---
        both_valid_mask = valid_left_mask & valid_right_mask

        # Extract original coordinates where both are valid
        left_orig_row_bv = left_orig_row[both_valid_mask]
        left_orig_col_bv = left_orig_col[both_valid_mask]
        right_orig_row_bv = right_orig_row[both_valid_mask]
        right_orig_col_bv = right_orig_col[both_valid_mask]

        # Calculate flattened indices for these points
        width_left = shape_left[1]
        width_right = shape_right[1]

        flat_indices_l_bv = left_orig_row_bv * width_left + left_orig_col_bv
        flat_indices_r_bv = right_orig_row_bv * width_right + right_orig_col_bv

        # --- Cases 2 & 3: One mapping valid, requires neighbor search for the other ---
        # Initialize lists to store results from neighbor searches
        indexl_neighbor = []
        indexr_neighbor = []

        # Case 2: Left valid, Right invalid -> Search neighbors in Right map
        left_only_mask = valid_left_mask & ~valid_right_mask
        rows_lo, cols_lo = np.where(left_only_mask) # Coordinates (ii, jj) needing search

        for r, c in zip(rows_lo, cols_lo):
            leftframe = left_map_crop[r, c] # This one is valid by definition
            # Find a valid neighbor in the *right* map at the *same* (r, c) location
            rightframe_neighbor = self.find_valid_neighbor(inverse_right_cam, r, c, sentinel_value) # Use full map for search
            if rightframe_neighbor is not None:
                # Calculate flattened indices using original left and found right
                indexl_neighbor.append(leftframe[1] * width_left + leftframe[0])
                indexr_neighbor.append(rightframe_neighbor[1] * width_right + rightframe_neighbor[0])

        # Case 3: Right valid, Left invalid -> Search neighbors in Left map
        right_only_mask = ~valid_left_mask & valid_right_mask
        rows_ro, cols_ro = np.where(right_only_mask) # Coordinates (ii, jj) needing search

        for r, c in zip(rows_ro, cols_ro):
            rightframe = right_map_crop[r, c] # This one is valid by definition
            # Find a valid neighbor in the *left* map at the *same* (r, c) location
            leftframe_neighbor = self.find_valid_neighbor(inverse_left_cam, r, c, sentinel_value) # Use full map for search
            if leftframe_neighbor is not None:
                 # Calculate flattened indices using found left and original right
                indexl_neighbor.append(leftframe_neighbor[1] * width_left + leftframe_neighbor[0])
                indexr_neighbor.append(rightframe[1] * width_right + rightframe[0])

        # --- Combine results ---
        # Convert neighbor search results to numpy arrays
        indexl_neighbor_arr = np.array(indexl_neighbor, dtype=flat_indices_l_bv.dtype)
        indexr_neighbor_arr = np.array(indexr_neighbor, dtype=flat_indices_r_bv.dtype)

        # Concatenate results from direct matches and neighbor searches
        final_indexl = np.concatenate((flat_indices_l_bv, indexl_neighbor_arr))
        final_indexr = np.concatenate((flat_indices_r_bv, indexr_neighbor_arr))

        return final_indexl, final_indexr
    def find_non_zero_neighbor(self, frame, ii, jj):
        neighbors = [(ii, jj + 1), (ii, jj - 1), (ii + 1, jj), (ii - 1, jj)]
        for ni, nj in neighbors:
            if 0 <= ni < frame.shape[0] and 0 <= nj < frame.shape[1]:
                neighbor = frame[ni, nj]
                if neighbor[0] != 0 and neighbor[1] != 0:
                    return neighbor
        return None
    def triangulate(self):
        """
        \ingroup structured_light
        \brief Triangulates the matched 3D points to obtain the 3D point cloud.

        \return A 3D point cloud.
        """
        # Implement triangulation logic here
        logging.info('finding correspondences...')
        inverse_left_cam, inverse_right_cam = self.find_correspondence_optimized()
        logging.info('finding correspondences cpp..')
        # Call the C++ function
        inverse_matrix_left_cam, inverse_matrix_right_cam = correspondence.find_correspondence_optimized(
            self.left_vertical_decoded_image,
            self.left_horizontal_decoded_image,
            self.right_vertical_decoded_image,
            self.right_horizontal_decoded_image
        )

        #logging.info('processing...')

        #indexl, indexr = self.process_frames_optimized(inverse_left_cam, inverse_right_cam,self.left_horizontal_decoded_image.shape,self.right_horizontal_decoded_image.shape )
        #logging.info('processing numba...')

        indexl, indexr = self.process_frames_numba(inverse_left_cam, inverse_right_cam,self.left_horizontal_decoded_image.shape,self.right_horizontal_decoded_image.shape )


        #logging.info('processing cpp...')

        #indexl, indexr = process_frames_module.process_frames(inverse_left_cam, inverse_right_cam, self.left_horizontal_decoded_image.shape[0], self.left_horizontal_decoded_image.shape[0],
         #                                    0)





        logging.info('line_filter...')

        #width = self.array_hor_l_masked.shape[1]
        #indices_1dl = [v * width + u for u, v in inverse_left_cam.ravel()]
        # convert list to numpy array
        #indices_1dl = np.array(indices_1dl).astype(np.uint32)
        indices_1dl = np.array(indexl).astype(np.uint32)
        #width = self.array_hor_r_masked.shape[1]
        #indices_1dr = [v * width + u for u, v in inverse_right_cam.ravel()]
        #indices_1dr = np.array(indices_1dr).astype(np.uint32)
        indices_1dr = np.array(indexr).astype(np.uint32)
        #nn,dn = self.ray_matching()
        rays_cam1 = self.camera_parameters[0].generate_rays()
        rays_cam2 = self.camera_parameters[1].generate_rays()
        self.H[0].invert()
        rays_cam2.TransformLines(self.H[0])
        #rays_cam2_selected = Line()
        #rays_cam2_selected.Ps = rays_cam2.Ps[nn]
        #rays_cam2_selected.V = rays_cam2.V[nn]

        rays_cam2_selected = Line()
        rays_cam2_selected.Ps = rays_cam2.Ps[indices_1dr]
        rays_cam2_selected.V = rays_cam2.V[indices_1dr]
        rays_cam1_selected = Line()
        rays_cam1_selected.Ps = rays_cam1.Ps[indices_1dl]
        rays_cam1_selected.V = rays_cam1.V[indices_1dl]
        logging.info('Triangulating...')

        ptcloud,d = intersection_between_2_lines(rays_cam1_selected
                                                 , rays_cam2_selected)


        logging.info('Triangulating done...')

        return ptcloud,d




        pass

    def ray_matching(self):
        """
        \ingroup structured_light
        \brief Matches rays from multiple cameras using the decoded Gray Code patterns.

        \param decoded_patterns: Decoded patterns from multiple cameras.
        \return A list of matched 3D points.
        """
        # Implement ray matching logic here

        # Get the shape of the left image
        print('Matching rays...')
        left_horiz = self.left_horizontal_decoded_image
        left_vert = self.left_vertical_decoded_image
        right_horiz = self.right_horizontal_decoded_image
        right_vert = self.right_vertical_decoded_image

        H, W = left_horiz.shape




        # Flatten disparity maps into (N, 2) and (M, 2) coordinate arrays
        left_points = np.stack([left_horiz.ravel(), left_vert.ravel()], axis=1)
        right_points = np.stack([right_horiz.ravel(), right_vert.ravel()], axis=1)

        # Build a KD-Tree for the right camera disparity map
        tree = cKDTree(right_points,leafsize=100)

        # Query the KD-Tree for the nearest neighbor in the right image for each left image pixel
        distances, nearest_indices = tree.query(left_points,p=1, workers=-1)  # workers=-1 uses all CPU cores

        # Convert 1D indices back to 2D (row, col) coordinates in right image
        #nearest_neighbors = np.column_stack(np.unravel_index(nearest_indices, (H, W)))

        # Reshape to match the original left image shape (H, W, 2)
        return nearest_indices, distances





        pass


    def _create_inverse_map(self, vertical_decoded: np.ndarray, horizontal_decoded: np.ndarray, sentinel_value: int = 0) -> np.ndarray:
        """
        Creates an inverse mapping from decoded coordinates to original image coordinates.

        Args:
            vertical_decoded: 2D array where each element is the decoded row index.
            horizontal_decoded: 2D array where each element is the decoded column index.
            sentinel_value: Value to fill unmapped locations in the inverse map.

        Returns:
            A 3D numpy array where inverse_map[decoded_row, decoded_col] = [original_col, original_row].
            Unmapped locations will have [sentinel_value, sentinel_value].
        """
        if vertical_decoded.shape != horizontal_decoded.shape:
            raise ValueError("Vertical and horizontal decoded image shapes must match.")
        if vertical_decoded.ndim != 2:
             raise ValueError("Input arrays must be 2-dimensional.")

        original_height, original_width = vertical_decoded.shape

        # Determine the required size of the inverse map arrays
        # Add a small epsilon before casting to int to handle potential float inputs safely,
        # although indices should ideally be integers already. Handle potential empty arrays.
        max_value_vert = np.amax(vertical_decoded) if vertical_decoded.size > 0 else 0
        max_value_hor = np.amax(horizontal_decoded) if horizontal_decoded.size > 0 else 0

        # Output map dimensions (add 1 because indices are 0-based)
        map_height = int(max_value_vert) + 1
        map_width = int(max_value_hor) + 1

        # Initialize inverse map arrays with the sentinel value
        # Using int32 is often sufficient for indices and saves memory
        inverse_array_row = np.full((map_height, map_width), sentinel_value, dtype=np.int32)
        inverse_array_column = np.full((map_height, map_width), sentinel_value, dtype=np.int32)

        # Create arrays representing the original (i, j) coordinates
        # Using broadcasting avoids explicit meshgrid for assignment
        original_rows_i = np.arange(original_height, dtype=np.int32)[:, None] # Shape (H, 1)
        original_cols_j = np.arange(original_width, dtype=np.int32)         # Shape (W,)

        # Use advanced indexing to populate the inverse maps
        # vertical_decoded and horizontal_decoded act as coordinate pairs for the target arrays
        # original_rows_i broadcasts to (H, W), assigning original 'i' (row)
        # original_cols_j broadcasts to (H, W), assigning original 'j' (column)
        # Only perform assignment if there are values to map
        if vertical_decoded.size > 0:
            inverse_array_row[vertical_decoded, horizontal_decoded] = original_rows_i
            inverse_array_column[vertical_decoded, horizontal_decoded] = original_cols_j

        # Stack the column and row arrays to get (col, row) pairs at each location
        # Shape: (map_height, map_width, 2)
        # Stacking column first matches the original code's temp_matrix structure
        inverse_map = np.stack((inverse_array_column, inverse_array_row), axis=2)

        return inverse_map

    def find_correspondence_optimized(self):
        """
        Finds the inverse mapping for left and right camera decoded images.

        Returns:
            A tuple containing:
            - inverse_matrix_left_cam: Map for the left camera.
            - inverse_matrix_right_cam: Map for the right camera.
            Each map is a 3D array where map[decoded_row, decoded_col] = [original_col, original_row].
        """
        # Process left camera
        inverse_matrix_left_cam = self._create_inverse_map(
            self.left_vertical_decoded_image,
            self.left_horizontal_decoded_image
        )

        # Process right camera
        inverse_matrix_right_cam = self._create_inverse_map(
            self.right_vertical_decoded_image,
            self.right_horizontal_decoded_image
        )

        return inverse_matrix_left_cam, inverse_matrix_right_cam

    def find_correspondence(self):
        ## Find max values to create empty (0) matrices to use
        self.array_hor_l_masked= self.left_horizontal_decoded_image
        self.array_vert_l_masked = self.left_vertical_decoded_image
        self.array_hor_r_masked = self.right_horizontal_decoded_image
        self.array_vert_r_masked= self.right_vertical_decoded_image
        max_value_vert_l_mask = np.amax(self.array_vert_l_masked)
        max_value_hor_l_mask = np.amax(self.array_hor_l_masked)
        max_value_vert_r_mask = np.amax(self.array_vert_r_masked)
        max_value_hor_r_mask = np.amax(self.array_hor_r_masked)


        # inverse_array_row_left = np.zeros((max_value_vert_l_mask + 1,max_value_hor_l_mask + 1))
        # inverse_array_column_left = np.zeros((max_value_vert_l_mask + 1,max_value_hor_l_mask + 1))
        # inverse_array_row_right = np.zeros((max_value_vert_r_mask + 1, max_value_hor_r_mask + 1))
        # inverse_array_column_right = np.zeros((max_value_vert_r_mask + 1, max_value_hor_r_mask + 1))
        #
        # ##Loop to generate invers matrix of Left Camera matrix for pixel correspondence
        # for i in range (self.array_hor_l_masked.shape[0]):
        #     for j in range (self.array_hor_l_masked.shape[1]):
        #         index_row_left = self.array_vert_l_masked[i][j]
        #         index_column_left = self.array_hor_l_masked[i][j]
        #         inverse_array_row_left[index_row_left][index_column_left] = i
        #         inverse_array_column_left[index_row_left][index_column_left] = j


        # Assuming self.array_vert_l_masked and self.array_hor_l_masked are already defined
        index_row_left = self.array_vert_l_masked
        index_column_left = self.array_hor_l_masked

        # Create empty arrays with the required shape
        inverse_array_row_left = np.zeros((np.amax(index_row_left) + 1, np.amax(index_column_left) + 1))
        inverse_array_column_left = np.zeros((np.amax(index_row_left) + 1, np.amax(index_column_left) + 1))

        # indexing to fill the arrays
        inverse_array_row_left[index_row_left, index_column_left] = np.arange(index_row_left.shape[0])[:, None]
        inverse_array_column_left[index_row_left, index_column_left] = np.arange(index_column_left.shape[1])


        temp_matrix_left = np.stack((inverse_array_column_left, inverse_array_row_left), axis=2)
        inverse_matrix_left_cam = temp_matrix_left.view([(f'f{i}', temp_matrix_left.dtype) for i in range(temp_matrix_left.shape[-1])])[
            ..., 0].astype('O')

        # Assuming self.array_vert_r_masked and self.array_hor_r_masked are already defined
        index_row_right = self.array_vert_r_masked
        index_column_right = self.array_hor_r_masked

        # Create empty arrays with the required shape
        inverse_array_row_right = np.zeros((np.amax(index_row_right) + 1, np.amax(index_column_right) + 1))
        inverse_array_column_right = np.zeros((np.amax(index_row_right) + 1, np.amax(index_column_right) + 1))

        # Use advanced indexing to fill the arrays
        inverse_array_row_right[index_row_right, index_column_right] = np.arange(index_row_right.shape[0])[:, None]
        inverse_array_column_right[index_row_right, index_column_right] = np.arange(index_column_right.shape[1])

        temp_matrix_right = np.stack((inverse_array_column_right, inverse_array_row_right),axis =2)
        inverse_matrix_right_cam = temp_matrix_right.view([(f'f{i}', temp_matrix_right.dtype) for i in range(temp_matrix_right.shape[-1])])[
            ..., 0].astype('O')

        return inverse_matrix_left_cam, inverse_matrix_right_cam
if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    import matplotlib
    matplotlib.use('qtagg')

    scanner = GrayCodeMultiCam()
    scanner.Decoder.BLACKTHR = 20
    #scanner.generate_patterns(width=1280, height=720, projector_directory=r"../Examples/static/0_projection_pattern")
    scanner.load_intrinsic_calibration(r"../Examples/static/1_calibration_data/intrinsic")
    scanner.load_extrinsic_calibration(r"../Examples/static/1_calibration_data/extrinsic")
    #scanner.decode_patterns_from_file(captured_images_path=r"..\Examples\static\2_object_data\Seppe/")
    scanner.decode_patterns_from_image_folder(r"../Examples/static/2_object_data3")

    xyz,d = scanner.triangulate()
    scanner.filter_and_plot_pointcloud(xyz,d, treshold= 5)
    scanner.save_as_ply_file('test2.ply', xyz, d,treshold =5)
    pass