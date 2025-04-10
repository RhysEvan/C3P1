"""
cam_proj.py:
    
    class document pretaining to the calibration sequence
    required to calibrate a camproj configuration.

    MonoCalibration:
        class object containing all executables for
        one camera projector calibration.
"""
import os
import cv2
import glob
import h5py
import fnmatch

import numpy as np
from scipy.ndimage import median_filter

from PyCamCalib.core.calibration import CameraCalibrator

class MonoCalibration():
    """
    class object used to calibrate stereo configurations with graycode
    configured to run stereo_calib in accordance with correct folder
    hierarchy, initalization is enough to retrieve stereo_parameters.h5

    Attributes
    ----------
    None

    Methods
    -------
    file_loading: loads image folder into array per identifier

    calibrate_cams: creates param_l and param_r containing all
    necessary info for stereo calibration and intrinsic calibration

    calibrate_stereo: store stereo parameters in .h5 file format after
    determining the transformation and rotation matrix
    """
    def __init__(self, accepted_index, decoding,
                 CALIBRATION_DATA_DIRECTORY,
                 CHESS_SHAPE,
                 CHESS_BLOCK_SIZE,
                 WIDTH,
                 HEIGHT,
                 IMAGE_RESOLUTION,
                 cam_int, cam_dist,
                 identifier=None):

        self.calibration_data_directory = CALIBRATION_DATA_DIRECTORY
        self.chess_shape = CHESS_SHAPE
        self.chess_block_size = CHESS_BLOCK_SIZE
        self.width = WIDTH
        self.height = HEIGHT
        self.image_resolution = IMAGE_RESOLUTION
        self.indexes = accepted_index

        self.cam_int = cam_int
        self.cam_dist = cam_dist

        self.decoder = decoding
        self.camera_calibrator = CameraCalibrator()

        if identifier is None:
            key = "pattern_white"
        else:
            key = f"{identifier}_pattern_white"
        self.white_images = self.file_loading(key, r'\scan_*.h5')

        self.calibrate_cam()

        self.calibrate_projector(self.camera_calibrator, identifier)

    def calibrate_cam(self):
        """
        uses PyCamCalib to calibrate the configuration such that intrinsic values
        of both cameras are known. Is more robust than opencv checkerboard, which
        is required for specular reflection on checkerboard paper.

        Variables:
        ----------
        None

        Returns:
        --------
        None
        """
        self.param = self.camera_calibrator.calibrate(self.white_images, self.chess_block_size)

    def file_loading(self, identifier, path):
        """
        file_loading: loads image folder into array per identifier
        Variables:
        ----------
        identifier: (str) indexing either right or left camera with L or R
        
        Return:
        -------
        image_array = array containing all images compliant to the identifier
        """
        images = []

        files = glob.glob(self.calibration_data_directory+ path)
        for file in files:
            with h5py.File(file, 'r') as f:
                key_name = f"{identifier}"
                if key_name in f:
                    image = f[key_name][:]
                    images.append(image)
                else:
                    raise KeyError(f"Key {key_name} not found in {file}")

        if images:
            # Stack images into a numpy array
            image_array = np.stack(images, axis=-1)
            return np.squeeze(image_array)
        else:
            raise ValueError("No images found in the provided files with the key format.")


    def calibration_mono_loading(self, path, identifier = None):
        files = glob.glob(self.calibration_data_directory + path)
        # Define the substrings to be excluded
        excluded_substrings = ["pattern_white", "pattern_black"]

        # Initialize a list to hold the arrays from each HDF5 file
        decoded_calibration_hor_scene = []
        decoded_calibration_vert_scene = []
        for file in files:
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
                    temp_images = [[],[],[],[]]
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

                    temp_hor = self.decoder.decode_process_calibration(scene_data_hor, white_array, black_array)
                    temp_vert = self.decoder.decode_process_calibration(scene_data_vert, white_array, black_array)
                    # plt.imshow(temp_hor)
                    # plt.show()
                    # plt.imshow(temp_vert)
                    # plt.show()
                    decoded_calibration_hor_scene.append(temp_hor)
                    decoded_calibration_vert_scene.append(temp_vert)
                else:
                    print(f"No keys matched in file: {file}")

        return decoded_calibration_hor_scene, decoded_calibration_vert_scene

    def calibrate_projector(self, calibrated, identifier = None):

        objps = np.zeros((self.chess_shape[0]*self.chess_shape[1], 3), np.float32)
        objps[:, :2] = self.chess_block_size * \
                       np.mgrid[0:self.chess_shape[0], 0:self.chess_shape[1]].T.reshape(-1, 2)

        print('Calibrating ...')
        cam_shape = self.image_resolution
        patch_size_half = int(np.ceil(cam_shape[1] / 180))
        print('  patch size :', patch_size_half * 2 + 1)

        cam_corners_list = []
        cam_objps_list = []
        cam_corners_list2 = []
        proj_objps_list = []
        proj_corners_list = []

        decoded_data_hor, decoded_data_vert = self.calibration_mono_loading(r'\scan_*.h5', identifier)
        for idx, _ in enumerate(decoded_data_hor):
            if idx not in self.indexes:
                pass
            else:
                idx_ver_filt = median_filter(decoded_data_vert[idx], 5)
                idx_hor_filt = median_filter(decoded_data_hor[idx], 5)
                outliers_ver = (np.abs(decoded_data_vert[idx] - idx_ver_filt)) > 10
                outliers_hor = (np.abs(decoded_data_hor[idx] - idx_hor_filt)) > 10
                idx_ver_outlier_removed = np.where(outliers_ver, idx_ver_filt, decoded_data_vert[idx])
                idx_hor_outlier_removed = np.where(outliers_hor, idx_hor_filt, decoded_data_hor[idx])

                feature_index = calibrated.indices.index(idx)
                cam_corners = calibrated.image_points_list[feature_index]

                cam_objps_list.append(objps)
                cam_corners_list.append(cam_corners)
                proj_objps = []
                proj_corners = []
                cam_corners2 = []# viz_proj_points = np.zeros(proj_shape, np.uint8)

                for corner, objp in zip(cam_corners, objps):
                    c_x = int(round(corner[0]))
                    c_y = int(round(corner[1]))
                    src_points = []
                    dst_points = []
                    for dx in range(-patch_size_half, patch_size_half + 1):
                        for dy in range(-patch_size_half, patch_size_half + 1):
                            x = c_x + dx
                            y = c_y + dy
                            src_points.append((x, y))
                            dst_points.append([idx_ver_outlier_removed[y, x], idx_hor_outlier_removed[y, x]])

                    if len(src_points) < patch_size_half**2:
                        print(
                            '    Warning : corner', c_x, c_y,
                            'was skiped because decoded pixels were too few (check your images and thresholds)')
                        continue
                    h_mat, inliers = cv2.findHomography(
                        np.array(src_points), np.array(dst_points))

                    point = h_mat@np.array([corner[0], corner[1], 1]).transpose()
                    point_pix = point[0:2]/point[2]
                    proj_objps.append(objp)
                    proj_corners.append([point_pix])
                    cam_corners2.append(corner)

                if len(proj_corners) < 3:
                    print('Error : too few corners were found in \'' +
                          str(idx) + '\' (less than 3)')
                    return None
                proj_objps_list.append(np.float32(proj_objps))
                proj_corners_list.append(np.float32(proj_corners))
                cam_corners_list2.append(np.float32(cam_corners2))

                # camera calibration for compatibility with next procedure

                print('Initial solution of projector\'s parameters')
                # projector calibration
                print(' Current Index Image Index   :', str(idx))
                ret, proj_int, proj_dist, proj_rvecs, proj_tvecs = cv2.calibrateCamera(
                    [np.float32(proj_objps)], [np.float32(proj_corners)], [self.height, self.width], None, None, None, None)
                print('  RMS :', ret)
                print('  Intrinsic parameters :')
                printNumpyWithIndent(proj_int, '    ')
                print('  Distortion parameters :')
                printNumpyWithIndent(proj_dist, '    ')
                print()

        # camera calibration for compatibility with next procedure

        print('Initial solution of projector\'s parameters')
        # projector calibration        
        ret, proj_int, proj_dist, proj_rvecs, proj_tvecs = cv2.calibrateCamera(
            proj_objps_list, proj_corners_list, [self.height, self.width], None, None, None, None)
        print('  RMS :', ret)
        print('  Intrinsic parameters :')
        printNumpyWithIndent(proj_int, '    ')
        print('  Distortion parameters :')
        printNumpyWithIndent(proj_dist, '    ')
        print()

        print('=== Result ===')
        # emulated stereo from proj and cam
        ret, cam_int, cam_dist, proj_int, proj_dist, cam_proj_rmat, cam_proj_tvec, E, F = cv2.stereoCalibrate(
            proj_objps_list, cam_corners_list2, proj_corners_list, self.cam_int, self.cam_dist, proj_int, proj_dist, None)
        print('  RMS :', ret)
        print('  Camera intrinsic parameters :')
        printNumpyWithIndent(cam_int, '    ')
        print('  Camera distortion parameters :')
        printNumpyWithIndent(cam_dist, '    ')
        print('  Projector intrinsic parameters :')
        printNumpyWithIndent(proj_int, '    ')
        print('  Projector distortion parameters :')
        printNumpyWithIndent(proj_dist, '    ')
        print('  Rotation matrix / translation vector from camera to projector')
        print('  (they translate points from camera coord to projector coord) :')
        printNumpyWithIndent(cam_proj_rmat, '    ')
        printNumpyWithIndent(cam_proj_tvec, '    ')
        print()

        # rectifying stereo parameters for better performance
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(self.cam_int, self.cam_dist, proj_int, proj_dist,
                                                    cam_shape, cam_proj_rmat, cam_proj_tvec)

        # Check if the directory already exists
        if not os.path.exists(self.calibration_data_directory):
            # Create the new directory
            os.makedirs(self.calibration_data_directory)
            print(f"Created directory: {self.calibration_data_directory}")

        if identifier is None:
            h5_file_path = os.path.join(self.calibration_data_directory, "mono_parameters.h5")
        # Path to the H5 file
        else:
            h5_file_path = os.path.join(self.calibration_data_directory, f"{identifier}_mono_parameters.h5")

        # Save calibration results in H5 file
        with h5py.File(h5_file_path, 'w') as h5_file:
            h5_file.create_dataset('cam_int', data=self.cam_int)
            h5_file.create_dataset('cam_dist', data=self.cam_dist)
            h5_file.create_dataset('proj_int', data=proj_int)
            h5_file.create_dataset('proj_dist', data=proj_dist)
            h5_file.create_dataset('R', data=cam_proj_rmat)
            h5_file.create_dataset('T', data=cam_proj_tvec)
            h5_file.create_dataset('R1', data=R1)
            h5_file.create_dataset('R2', data=R2)
            h5_file.create_dataset('P1', data=P1)
            h5_file.create_dataset('P2', data=P2)

        print(f"Saved calibration data to {h5_file_path}")

def printNumpyWithIndent(tar, indentchar):
    print(indentchar + str(tar).replace('\n', '\n' + indentchar))
