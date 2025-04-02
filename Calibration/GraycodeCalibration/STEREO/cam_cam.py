"""
cam_cam.py:
    
    class document pretaining to the calibration sequence
    required to calibrate a cam_cam configuration.

    StereoCalibration: 
        class object containing all executables for 
        two camera calibration.
"""

# Imports
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
from PyCamCalib.core.calibration import CameraCalibrator, StereoCalibrator

class StereoCalibration():
    def __init__(self, decoding,
                 CALIBRATION_DATA_DIRECTORY,
                 CHESS_SHAPE,
                 CHESS_BLOCK_SIZE):

        self.calibration_data_directory = CALIBRATION_DATA_DIRECTORY
        self.chess_shape = CHESS_SHAPE
        self.chess_block_size = CHESS_BLOCK_SIZE

        self.decoder = decoding
        # Calibrate each camera.
        self.left_camera_calibrator = CameraCalibrator()
        self.right_camera_calibrator = CameraCalibrator()
        # Calibrate stereo configuration.
        self.stereo_calibrator = StereoCalibrator()

        self.white_images_l = self.file_loading("L_pattern_white", r'\scan_*.h5')
        self.white_images_r = self.file_loading("R_pattern_white", r'\scan_*.h5')

        self.black_images_l = self.file_loading("L_pattern_black", r'\scan_*.h5')
        self.black_images_r = self.file_loading("R_pattern_black", r'\scan_*.h5')

        self.calibrate_cams()
        self.calibrate_stereo()

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

    def calibrate_cams(self):
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
        self.param_l = self.left_camera_calibrator.calibrate(self.white_images_l, self.chess_block_size)
        self.param_r = self.right_camera_calibrator.calibrate(self.white_images_r, self.chess_block_size)

    def calibrate_stereo(self):
        """
        uses PyCamCalib to calibrate the configuration such that the transformation
        and rotation matrix for both cameras are known. Meaning the stereo configuration
        is calibrated. This gets stored in an h5 file format.

        Variables:
        ----------
        None

        Returns:
        --------
        None
        """
        stereo_parameters = self.stereo_calibrator.calibrate(self.white_images_l,
                                                             self.white_images_r,
                                                             self.param_l,
                                                             self.param_r, self.chess_block_size, self.chess_shape)
        stereo_parameters.save_parameters("./static/1_calibration_data/stereo_parameters.h5")
        self.visualize_detection()

    def visualize_detection(self):
        """
        Final check up function to visualize all checkerboard captures to see whether
        all corners are detected.

        Variables:
        ----------
        None

        Returns:
        --------
        None
        """
        for image_idx in range(self.white_images_l.shape[-1]):
            image_1 = self.white_images_l[..., image_idx]
            image_2 = self.white_images_r[..., image_idx]
            fig, axarr = plt.subplots(1, 2, constrained_layout=True)
            axarr[0].imshow(image_1)
            axarr[1].imshow(image_2)
            if image_idx in self.stereo_calibrator.indices:
                feature_index = self.stereo_calibrator.indices.index(image_idx)
                axarr[0].plot(self.stereo_calibrator.image_points_list_1[feature_index][:, 0],
                              self.stereo_calibrator.image_points_list_1[feature_index][:, 1],
                              '-o', color='lime')
                axarr[1].plot(self.stereo_calibrator.image_points_list_2[feature_index][:, 0],
                              self.stereo_calibrator.image_points_list_2[feature_index][:, 1],
                              '-o', color='lime')
                color = 'g'
            else:
                axarr[0].plot(self.stereo_calibrator.feature_list_1[image_idx].image_points[:, 0],
                              self.stereo_calibrator.feature_list_1[image_idx].image_points[:, 1],
                              '-o', color='red')
                axarr[1].plot(self.stereo_calibrator.feature_list_2[image_idx].image_points[:, 0],
                              self.stereo_calibrator.feature_list_2[image_idx].image_points[:, 1],
                              '-o', color='red')
                color = 'r'
            fig.suptitle('Image nr ' + str(image_idx+1), color=color)
            axarr[0].get_xaxis().set_visible(False)
            axarr[0].get_yaxis().set_visible(False)
            axarr[1].get_xaxis().set_visible(False)
            axarr[1].get_yaxis().set_visible(False)
            plt.show()

            self.stereo_calibrator.plot_reproj_error()

def printNumpyWithIndent(tar, indentchar):
    print(indentchar + str(tar).replace('\n', '\n' + indentchar))
