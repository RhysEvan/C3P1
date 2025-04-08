import glob
from copy import deepcopy
from core_toolbox_python.Plucker.Line import *

import numpy as np
import h5py
import os
import fnmatch

from Cython.Shadow import nonecheck
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
from PyCamCalib.core.calibration import *
from Classes.GrayCode import GrayCodeMultiCam
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



class GrayCode_mono(GrayCodeMultiCam):
    """
    \ingroup structured_light
    \brief A class to represent a Gray Code structured light system for a single camera with projector intrinsics.

    This class is a child of the GrayCodeMultiCam class and includes additional attributes for projector intrinsics.
    """

    def __init__(self):
        """
        \ingroup structured_light
        \brief Initializes the GrayCode_mono class.
        """
        super().__init__()
        self.projector_parameters = None
        self.projector_dimensions = [720, 1280]



    def calibrate_projector_extrinsics(self,EXRINSIC_CALIBRATION_DATA_DIRECTORY = None, INTRINSIC_CALIBRATION_DATA_DIRECTORY = None):
        """
        Calibrates the projector extrinsics and intrinsics.

        This method captures graycode data and performs stereo calibration to determine the rotations and translations
        between the cameras.


        Returns:
        None
        """
        ## check if there is a custom input
        if INTRINSIC_CALIBRATION_DATA_DIRECTORY is None:
            INTRINSIC_CALIBRATION_DATA_DIRECTORY = self.INTRINSIC_CALIBRATION_DATA_DIRECTORY
        else:
            self.INTRINSIC_CALIBRATION_DATA_DIRECTORY = INTRINSIC_CALIBRATION_DATA_DIRECTORY

        if EXRINSIC_CALIBRATION_DATA_DIRECTORY is None:
            EXRINSIC_CALIBRATION_DATA_DIRECTORY = self.EXRINSIC_CALIBRATION_DATA_DIRECTORY
        else:
            self.EXRINSIC_CALIBRATION_DATA_DIRECTORY = EXRINSIC_CALIBRATION_DATA_DIRECTORY

        ## check if files are available
        calibrated_data = []
        calibration_data = glob.glob(EXRINSIC_CALIBRATION_DATA_DIRECTORY+"/*scan_*.h5")
        if len(calibration_data) == 0:
            raise FileNotFoundError("No extrinsic calibration images found in the directory.")
        calibrated_data = glob.glob(INTRINSIC_CALIBRATION_DATA_DIRECTORY+"/*_intrinsic_parameters.json")
        if len(calibrated_data) == 0:
            # trow error, no calibration images found
            raise FileNotFoundError("No calibration images found in the directory.")

        self.__projector_calibration()
        pass

    def __calibration_mono_loading(self, path, identifier=None):
        files = glob.glob(path)
        # Define the substrings to be excluded
        excluded_substrings = ["pattern_white", "pattern_black"]
        # Initialize a list to hold the arrays from each HDF5 file
        decoded_calibration_hor_scene = []
        decoded_calibration_vert_scene = []
        for file in files: # todo, make this use the standard self.decode functions
            with h5py.File(file, 'r') as h5file:
                # Print all keys in the HDF5 file for debugging
                all_keys = list(h5file.keys())

                if identifier is not None:
                    # Find all keys that match the pattern and do not contain any of the excluded substrings
                    all_matching_keys = [
                        key for key in all_keys
                        if fnmatch.fnmatch(key, identifier + '*')
                    ]
                else:
                    all_matching_keys = all_keys

                scene_data = {}
                for key in all_matching_keys:
                    scene_data[key] = np.array(h5file[key])

                # Iterate through scene_data dictionary
                for key, value in scene_data.items():
                    if 'pattern_black' in key:
                        black_array = value
                    elif 'pattern_white' in key:
                        white_array = value

                matched_keys = [
                    key for key in all_matching_keys
                    if not any(substring in key for substring in excluded_substrings)
                ]

                scene_data_hor = None
                scene_data_vert = None
                # Load the datasets corresponding to the matched
                # keys and stack them into a single NumPy array
                if matched_keys:
                    temp_images = [[], [], [], []]
                    for key in matched_keys:
                        if 'H' in key and not 'H_I' in key:
                            temp_images[2] = np.array(h5file[key])
                            scene_data_hor = np.array(h5file[key])
                        elif 'H_I' in key:
                            temp_images[3] = np.array(h5file[key])
                        elif 'V' in key and not 'V_I' in key:
                            temp_images[0] = np.array(h5file[key])
                            scene_data_vert = np.array(h5file[key])
                        elif 'V_I' in key:
                            temp_images[1] = np.array(h5file[key])
                    # self.save_scene_data_to_png(scene_data,
                    # output_folder=r"C:\Users\Seppe\PycharmProjects\Projector\CPC_CamProCam_UAntwerp\Examples\static\test")
                    temp_hor = self.Decoder.decode_process_calibration(scene_data_hor, white_array,
                                                                       black_array)  # 0705 uitzetten
                    temp_vert = self.Decoder.decode_process_calibration(scene_data_vert, white_array, black_array)
                    # plt.imshow(temp_hor)
                    # plt.show()
                    # plt.imshow(temp_vert)
                    # plt.show()
                    decoded_calibration_hor_scene.append(temp_hor)
                    decoded_calibration_vert_scene.append(temp_vert)
                else:
                    print(f"No keys matched in file: {file}") #todo make this use the standard self.decode function

        return decoded_calibration_hor_scene, decoded_calibration_vert_scene


    def image_to_projector_coordinates(self, uvall,correspondence_map):
        uvProjector = []
        ## lenght of the list uvall

        for j, uv in enumerate(uvall):
            try:
                # Calculate floor and ceil values
                uv_floor = np.floor(uv).astype(np.int32)
                uv_ceil = np.ceil(uv).astype(np.int32)
                weight = uv - uv_floor

                # Lookup projector coordinates for floor and ceil values
                ut = [correspondence_map[0][uv_floor[1], uv_floor[0]],
                      correspondence_map[0][uv_ceil[1], uv_ceil[0]]]
                vt = [correspondence_map[1][uv_floor[1], uv_floor[0]],
                      correspondence_map[1][uv_ceil[1], uv_ceil[0]]]

                # Interpolate projector coordinates
                u = ut[0] * (1 - weight[0]) + ut[1] * weight[0]
                v = vt[0] * (1 - weight[1]) + vt[1] * weight[1]

                uvProjector.append([u, v])

            except (IndexError, KeyError):
                # Handle out-of-bounds or missing data
                uvProjector.append([np.nan, np.nan])
                print('oei, problem in imge to projector coordinates')



        return (uvProjector)
    def __projector_calibration(self):

        #objps = np.zeros((self.boardsize[0] * self.boardsize[1], 3), np.float32)
        #objps[:, :2] = self.checkerboard_size * \
         #              np.mgrid[0:self.boardsize[0], 0:self.boardsize[1]].T.reshape(-1, 2)

        logging.info('start projector calibration')
        cam_shape = self.camera_parameters[0].sensor_dimensions
        patch_size_half = int(np.ceil(cam_shape[1] / 180/2))
        print('  patch size :', patch_size_half * 2 + 1)



        logging.info('decoding calibration data')
        decoded_data_hor, decoded_data_vert = self.__calibration_mono_loading(self.EXRINSIC_CALIBRATION_DATA_DIRECTORY+"/*scan_*.h5",'L')
        calibrated_data = glob.glob(self.EXRINSIC_CALIBRATION_DATA_DIRECTORY + "/*_scan_*.h5")
        images_list = []
        #for data in calibrated_data:
        ref_images_l = self._file_loading("L_pattern_white", r'\*scan_*.h5')
        ref_images_r = self._file_loading("R_pattern_white", r'\*scan_*.h5') #todo, make this camera name independent
        images_list.append(ref_images_l)
        images_list.append(ref_images_r)
        # Perform stereo calibration
        filename = self.EXRINSIC_CALIBRATION_DATA_DIRECTORY #todo, make this naming independant of number of cams

        calibrator = CameraCalibrator()
        calibrator.camera_parameters = self.camera_parameters[0]
        image_array = ref_images_l

        calibrator.sensor_dimensions = np.array([image_array.shape[1], image_array.shape[0]])
        calibrator.construct_feature_list(image_array, self.checkerboard_size, self.boardsize)
        calibrator.construct_points_lists([], absolute = False)

        image_checker_corner_list = deepcopy(calibrator.image_points_list)
        object_point_list = deepcopy(calibrator.object_points_list)

        # correspondence_map is a list, in the first element is the decoded data of the vertical
        # and in the second element is the decoded data of the horizontal
        projector_checker_corner_list = []
        logging.info('convert decoded data to corners')

        for idx,corners in enumerate(image_checker_corner_list):
            correspondence_map = []
            print('converting checkekerboard idx',idx)
            correspondence_map.append(decoded_data_vert[idx])
            correspondence_map.append(decoded_data_hor[idx])
            corners_uv = self.image_to_projector_coordinates(corners,correspondence_map)
            projector_checker_corner_list.append( np.array(corners_uv).astype(np.float32))
        # for every ndarray in the list



            #proj_corners_list.append(np.float32(proj_corners))
        logging.info('calibrate projector')
        projector_calibrator = CameraCalibrator()

        projector_calibrator.image_points_list = projector_checker_corner_list
        projector_calibrator.object_points_list= deepcopy(calibrator.object_points_list)
        projector_calibrator.sensor_dimensions= np.array(self.projector_dimensions)
        projector_calibrator.indices = deepcopy(calibrator.indices)
        # make a list from 1 to len(calibrator.image_points_list)
        projector_calibrator.indices = list(range(len(projector_calibrator.image_points_list)))
        projector_calibrator.camera_parameters = None
        projector_params = projector_calibrator.opencv_calibration(self.projector_dimensions)

        projector_params.sensor_dimensions = np.array(self.projector_dimensions)
        old_indices = projector_calibrator.indices
        #proj_images_points = projector_calibrator.image_points_list
        for i in range(10):  # do this max 10 times
            print("select outliers, exit to continue")
            new_indices = projector_calibrator.plot_and_filter_reproj_error()
            if new_indices == old_indices:
                print("New indices are the same as old indices. Continuing to the next calibration.")
                projector_params.sensor_dimensions = np.array(self.projector_dimensions)

                projector_params.save_parameters_to_json(filename + '/projector_intrinsic_parameters.json')
                break  # Skip the rest of the loop and move to the next iteration
            else:
                projector_calibrator.image_points_list = []
                projector_calibrator.object_points_list = []
                projector_calibrator.camera_parameters = None

                for idx in new_indices:
                    print(idx)
                    projector_calibrator.image_points_list.append(projector_checker_corner_list[idx])
                    try:
                        projector_calibrator.object_points_list.append( calibrator.object_points_list[idx])
                    except:
                        print('error')
                projector_calibrator.indices = new_indices
                projector_params = projector_calibrator.opencv_calibration(self.projector_dimensions)
                old_indices = new_indices
        projector_params.sensor_dimensions = np.array(self.projector_dimensions)
        projector_params.save_distortion_image(filename + '/projector_distortion.png')
        projector_params.save_parameters_to_json(filename + '/projector_intrinsic_parameters.json')
        self.projector_parameters = projector_params

        ## stereo calib
        #super().load_intrinsic_calibration(self.INTRINSIC_CALIBRATION_DATA_DIRECTORY)
        stereocalibrator = StereoCalibrator()
        stereo_parameters = stereocalibrator.calibrate_from_features(image_checker_corner_list,
                                                 projector_checker_corner_list,self.camera_parameters[0],self.projector_parameters,self.checkerboard_size,self.boardsize,object_point_list)


        # %% Retry calibration after removing potential outliers
        stereocalibrator.indices = list(range(len(calibrator.image_points_list)))
        old_indices = stereocalibrator.indices

        for i in range(10):  # do this max 10 times
            print("select outliers, exit to continue")
            new_indices = stereocalibrator.plot_and_filter_reproj_error()
            if new_indices == old_indices:
                print("New indices are the same as old indices. Continuing to the next calibration.")
                break  # Skip the rest of the loop and move to the next iteration
            else:
                stereo_parameters = stereocalibrator.calibrate_indices(new_indices)
                stereo_parameters.save_parameters(filename + '/_extrinsic_proj_parameters.h5')
                print(stereo_parameters.T)
                old_indices = new_indices
        stereo_parameters.info = ['L,proj']  # todo make this camera name independent

        stereo_parameters.save_parameters(filename + '/_extrinsic_proj_parameters.h5')
        print(filename + '_extrinsic_proj_parameters.h5')
        pass

    def load_intrinsic_calibration(self,EXRINSIC_CALIBRATION_DATA_DIRECTORY,INTRINSIC_CALIBRATION_DATA_DIRECTORY = None):
        ## load intrinsic parameters from json files
        super().load_intrinsic_calibration(INTRINSIC_CALIBRATION_DATA_DIRECTORY)

        calibrated_data = glob.glob(EXRINSIC_CALIBRATION_DATA_DIRECTORY+"/*projector_intrinsic_parameters.json")
        projector_parameters = []
        for data in calibrated_data:
            projector_parameters.append(CameraParameters())
            projector_parameters[-1].load_parameters_from_json(data)
        self.projector_parameters = projector_parameters


    def load_extrinsic_calibration(self,EXRINSIC_CALIBRATION_DATA_DIRECTORY = None):
        ## load intrinsic parameters from json files
        if EXRINSIC_CALIBRATION_DATA_DIRECTORY is not None:
            self.EXRINSIC_CALIBRATION_DATA_DIRECTORY = EXRINSIC_CALIBRATION_DATA_DIRECTORY
        calibrated_data = glob.glob(self.EXRINSIC_CALIBRATION_DATA_DIRECTORY+"/*_extrinsic_proj_parameters_TransformationMatrix.json")
        self.H = []
        for data in calibrated_data:
            self.H.append(TransformationMatrix())
            self.H[-1].load_from_json(data)


    def triangulate(self):
        """
        \ingroup structured_light
        \brief Triangulates the matched 3D points to obtain the 3D point cloud.

        \return A 3D point cloud.
        """
        # Implement triangulation logic here
        logging.info('finding correspondences...')

        #nn,dn = self.ray_matching()
        rays_cam1 = self.camera_parameters[0].generate_rays()
        rays_proj = self.projector_parameters[0].generate_rays()
        H = deepcopy(self.H[0])
        H.invert()
        rays_proj.TransformLines(H)
        # Flatten the indices
        w,h = self.projector_dimensions
        # clip self.left_horizontal_decoded_image

        #self.left_horizontal_decoded_image = np.clip(self.left_horizontal_decoded_image, 0, w-1)
        #self.left_vertical_decoded_image = np.clip(self.left_vertical_decoded_image, 0, h-1)
        index = self.left_horizontal_decoded_image.astype(np.int64)*w+self.left_vertical_decoded_image.astype(np.int64)
        #index = np.ravel_multi_index((self.left_horizontal_decoded_image, self.left_vertical_decoded_image), [w,h])
        #plt.imshow(index)
        #plt.show()
        index = index.ravel()

        # Create a boolean mask for values that will be clipped
        # Store the indices where the values are clipped
        clipped_indices = np.where(np.asarray(index)<self.projector_dimensions[0]* self.projector_dimensions[1])
        clipped_indices2 = np.where(np.asarray(index)>0.5)

        # Clip the index values
        index2 = np.clip(index, 0, self.projector_dimensions[0]* self.projector_dimensions[1] - 1)
        index2 = index2.astype(np.int64)
        flat_indices = index2# Assign values in a vectorized manner
        rays_proj_match = Line()
        rays_proj_match.Ps = rays_proj.Ps[flat_indices]
        rays_proj_match.V= rays_proj.V[flat_indices]



        rays_proj_match1 = Line()
        rays_proj_match1.Ps = rays_proj_match.Ps[clipped_indices2[0]]
        rays_proj_match1.V = rays_proj_match.V[clipped_indices2[0]]
        rays_cam11 = Line()
        rays_cam11.Ps = rays_cam1.Ps[clipped_indices2[0]]
        rays_cam11.V = rays_cam1.V[clipped_indices2[0]]
        rays_cam1 = rays_cam11
        rays_proj_match = rays_proj_match1
        #rays_proj_match.PlotLine()
        #rays_cam1.PlotLine()


        logging.info('interesect...')
        ptcloud,d = intersection_between_2_lines(rays_cam1
                                                 , rays_proj_match)


        logging.info('Triangulating done...')
        # whre clipped_indices2 make d 999
        #ptcloud = ptcloud[ clipped_indices[0]]
        #d = d[ clipped_indices[0]]


        return ptcloud,d




        pass


if __name__ == "__main__":
    import logging
    import matplotlib

    #matplotlib.use("qt5agg")
    matplotlib.use("tkagg")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    scanner = GrayCode_mono()
    scanner.projector_dimensions = [1280,720]
    scanner.INTRINSIC_CALIBRATION_DATA_DIRECTORY = r"../Examples/static/1_calibration_data/intrinsic"
    scanner.load_intrinsic_calibration(r"../Examples/static/1_calibration_data/extrinsic",r"../Examples/static/1_calibration_data/intrinsic")
    #scanner.calibrate_projector_extrinsics(r"../Examples/static/1_calibration_data/extrinsic",r"../Examples/static/1_calibration_data/intrinsic")
    scanner.load_intrinsic_calibration(r"../Examples/static/1_calibration_data/extrinsic",r"../Examples/static/1_calibration_data/intrinsic")
    scanner.load_extrinsic_calibration(r"../Examples/static/1_calibration_data/extrinsic")
    scanner.decode_patterns_from_file(captured_images_path=r"..\Examples\static\2_object_data\Seppe/")
    xyz,d = scanner.triangulate()

    scanner.filter_and_plot_pointcloud(xyz,d, treshold=10)



    pass
