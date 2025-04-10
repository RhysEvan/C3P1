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

import multiprocessing
from DataCapture import graycode_data_capture
from StructuredLight.Graycode.projector_image import *



class GrayCodeMultiCam(Structured_Light):
    """
    \ingroup structured_light
    \brief A class to represent a Gray Code structured light system for multiple cameras.

    This class is a child of the Structured_Light class and includes methods for generating patterns, decoding patterns, and ray matching.
    """

    def __init__(self):
        """
        \ingroup structured_light
        \brief Initializes the GrayCodeMultiCam class.
        """
        super().__init__()
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
        inverse_left_cam, inverse_right_cam = self.find_correspondence()
        indexl, indexr = self.process_frames2(inverse_left_cam, inverse_right_cam,self.array_hor_l_masked.shape,self.array_hor_r_masked.shape )
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