"""
cam_calib.py:
    
    class document pretaining to the calibration sequence
    required to calibrate a cam_cam configuration.

    IntrinsicCalibration: 
        class object containing all executables for 
        intrinsic camera calibration.
"""
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
from PyCamCalib.core.calibration import CameraCalibrator

class IntrisicCalibration():
    def __init__(self, accepted_index,
                 INTRINSIC_CALIBRATION_DATA_DIRECTORY,
                 CHESS_SHAPE,
                 CHESS_BLOCK_SIZE,
                 WIDTH,
                 HEIGHT,
                 IMAGE_RESOLUTION,
                 identifier=None,
                 visualize = False):

        self.intrinsic_calibration_data_directory = INTRINSIC_CALIBRATION_DATA_DIRECTORY
        self.chess_shape = CHESS_SHAPE
        self.chess_block_size = CHESS_BLOCK_SIZE
        self.width = WIDTH
        self.height = HEIGHT
        self.image_resolution = IMAGE_RESOLUTION
        self.indexes = accepted_index

        self.camera_calibrator = CameraCalibrator()
        if identifier != None:
            self.images = self.file_loading(fr'\{identifier}_scan.h5')
        else:
            self.images = self.file_loading(r'\scan.h5')

        self.calibrate_cam(identifier, visualize)

    def calibrate_cam(self, identifier, visualize = False):
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
        self.param = self.camera_calibrator.calibrate(self.images, self.chess_block_size)
        if visualize is True:
            self.visualize_detection()
        if identifier is None:
            self.param.save_parameters(self.intrinsic_calibration_data_directory+'/camera_intrinsic_parameters.h5')
        else:
            self.param.save_parameters(self.intrinsic_calibration_data_directory+f'/{identifier}_camera_intrinsic_parameters.h5')

    def file_loading(self, path):
        """
        file_loading: loads image folder into array per identifier
        Variables:
        ----------
        TODO INSERT proper info
        Return:
        -------
        image_array = array containing all images compliant to the identifier
        """
        images = []

        files = glob.glob(self.intrinsic_calibration_data_directory+ path)
        for file in files:
            with h5py.File(file, 'r') as f:
                for key in f.keys():
                    image = f[key][:]
                    images.append(image)

        if images:

            filtered_images = [images[i] for i in self.indexes]
            # Stack the filtered images along the last axis
            image_array = np.stack(filtered_images, axis=-1)

            return np.squeeze(image_array)
        else:
            raise ValueError("No images found in the provided files with the key format.")

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
        for image_idx in range(self.images.shape[-1]):
            image = self.images[..., image_idx]
            plt.imshow(image)
            if image_idx in self.camera_calibrator.indices:
                feature_index = self.camera_calibrator.indices.index(image_idx)
                plt.plot(self.camera_calibrator.image_points_list[feature_index][:, 0],
                         self.camera_calibrator.image_points_list[feature_index][:, 1],
                         '-o', color='lime')
                color = 'g'
            elif self.camera_calibrator.feature_list[image_idx].score != 0:
                plt.plot(self.camera_calibrator.feature_list[image_idx].image_points[:, 0],
                         self.camera_calibrator.feature_list[image_idx].image_points[:, 1],
                         '-o', color='r')
                color = 'r'
            else:
                color = 'r'
            plt.title('Image nr ' + str(image_idx+1), color=color)
            plt.gca().get_xaxis().set_visible(False)
            plt.gca().get_yaxis().set_visible(False)
            plt.show()

            #%% Plot reprojection errors
            self.camera_calibrator.plot_reproj_error()