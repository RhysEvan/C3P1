import glob
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
from PyCamCalib.core.calibration import *
from Classes.Multicam import *
from Classes.IntrinsicCapture.intrinsic_capture import *

class Stereo:
    """
    \brief A class to represent a stereo camera system.

    This class holds lists of camera intrinsics, extrinsics, cameras, and rays. It also includes a method to transform rays to world coordinates.
    """

    def __init__(self):
        """
        \brief Initializes the Stereo class with empty lists for camera intrinsics, extrinsics, cameras, and rays.
        """
        self.camera_intrinsics = [] # list of cameraparameters (pycamcalib)
        self.camera_extrinsics = []
        self.rays = []

        self.INTRINSIC_CALIBRATION_DATA_DIRECTORY = None
        self.checkerboard_size = 30
        self.boardsize = (8,11)
        self.H = None

        self.cameras = MultiCam()

    def load_cams_from_file(self, filename):
        self.cameras.load_cams_from_file(filename)
    def transform_rays_to_world_coordinates(self):
        """
        \brief Transforms rays to world coordinates using the corresponding extrinsic matrices.

        \return A list of rays transformed to world coordinates.
        """
        pass
    def capture_extrinsic_calibration_images(self,EXRINSIC_CALIBRATION_DATA_DIRECTORY = None,numberofimages = 40):
        # if extrinsic calibration dir does not exist
        if EXRINSIC_CALIBRATION_DATA_DIRECTORY is None:
            EXRINSIC_CALIBRATION_DATA_DIRECTORY = self.EXRINSIC_CALIBRATION_DATA_DIRECTORY
        else:
            self.EXRINSIC_CALIBRATION_DATA_DIRECTORY = EXRINSIC_CALIBRATION_DATA_DIRECTORY
        if not os.path.exists(EXRINSIC_CALIBRATION_DATA_DIRECTORY):
            os.makedirs(EXRINSIC_CALIBRATION_DATA_DIRECTORY)
        for counter in range(numberofimages):
            print('capturing image ' + str(counter) +' of '  + str(numberofimages))
            extrinsic_calibration_capture(EXRINSIC_CALIBRATION_DATA_DIRECTORY, counter, self.cameras, identifier='black')

    def capture_intrinsic_calibration_images(self,INTRINSIC_CALIBRATION_DATA_DIRECTORY = None,numberofimages = 40):
        """
        \brief Captures images for intrinsic calibration of the stereo camera system.
        """
        #cameras = self.cameras
        #self.set_exposure_time(cameras,self.exposure_intrinsic)
        if INTRINSIC_CALIBRATION_DATA_DIRECTORY is None:
            INTRINSIC_CALIBRATION_DATA_DIRECTORY = self.INTRINSIC_CALIBRATION_DATA_DIRECTORY
        else:
            self.INTRINSIC_CALIBRATION_DATA_DIRECTORY = INTRINSIC_CALIBRATION_DATA_DIRECTORY

        try:
            # Your code that might raise an exception
            intrinsic_calibration_capture(INTRINSIC_CALIBRATION_DATA_DIRECTORY, self.cameras, identifier=['L', 'R'],
                                          numberofimages=numberofimages)
        except Exception as e:
            # Print the last error message
            import warnings

            print(f"An error occurred: {e}")
            warnings.warn(f"An error occurred: {e}", RuntimeWarning)
            print("Closing cameras to avoid memory leaks after error.")
            self.cameras.Close()
        pass
    def calibrate_camera_intrinsics(self,INTRINSIC_CALIBRATION_DATA_DIRECTORY = None):
        """
        Calibrates the camera intrinsics.
        Camera images (h5) and intrinics (h5) are saved in the INTRINSIC_CALIBRATION_DATA_DIRECTORY (class variable)
        :return: none
        """
        if INTRINSIC_CALIBRATION_DATA_DIRECTORY is None:
            INTRINSIC_CALIBRATION_DATA_DIRECTORY = self.INTRINSIC_CALIBRATION_DATA_DIRECTORY
        else:
            self.INTRINSIC_CALIBRATION_DATA_DIRECTORY = INTRINSIC_CALIBRATION_DATA_DIRECTORY

        calibrated_data = []
        calibrated_data = glob.glob(INTRINSIC_CALIBRATION_DATA_DIRECTORY+"/*_scan.h5")

        if len(calibrated_data) == 0:
            # trow error, no calibration images found
            raise FileNotFoundError("No calibration images found in the directory.")


        else:
            print("Intrinsic calibration images loading")
        calibrated_data = []
        #calibrated_data = glob.glob(INTRINSIC_CALIBRATION_DATA_DIRECTORY+"/*parameters.h5")
        if len(calibrated_data) == 0:
            print("no calibration ")
            self.__intrinsic_calibration()

        self.calibration_data = calibrated_data
    def calibrate_camera_extrinsics(self,EXRINSIC_CALIBRATION_DATA_DIRECTORY = None, INTRINSIC_CALIBRATION_DATA_DIRECTORY = None):
        """
        Calibrates the camera extrinsics.

        This method captures graycode data and performs stereo calibration to determine the rotations and translations
        between the cameras.

        Steps:
        1. Capture graycode data if not already available.
        2. Perform stereo calibration if not already done.
        3. Set the correct exposure time for the cameras.

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
        calibration_data = glob.glob(EXRINSIC_CALIBRATION_DATA_DIRECTORY+"/*.h5")
        if len(calibration_data) == 0:
            raise FileNotFoundError("No extrinsic calibration images found in the directory.")
        calibrated_data = glob.glob(INTRINSIC_CALIBRATION_DATA_DIRECTORY+"/*_intrinsic_parameters.json")
        if len(calibrated_data) == 0:
            # trow error, no calibration images found
            raise FileNotFoundError("No calibration images found in the directory.")

        self.__stereo_calibration()
        pass
    def __intrinsic_calibration(self):
        calibrated_data = glob.glob(self.INTRINSIC_CALIBRATION_DATA_DIRECTORY+"/*_scan.h5")
        for data in calibrated_data:
            images = self._file_loading_single(data)
            calibrator = CameraCalibrator()
            #calibrator.initilize_camera_parameters(1700,1700,images.shape[1]/2,images.shape[0]/2)
            filename, file_extension = os.path.splitext(data)
            print('Calibrating camera intrinsics ' + data)
            print("this might take a while")
            camera_parameters = calibrator.calibrate(images,self.checkerboard_size )
            camera_parameters.info = os.path.basename(filename)
            print("Calibration done")
            # %% save all detection to a file
            calibrator.save_checkerboard_detection_to_images(images, filename)
            # %% Retry calibration after removing potential outliers
            old_indices = calibrator.indices
            for i in range(10): # do this max 10 times
                print("select outliers, exit to continue")
                new_indices = calibrator.plot_and_filter_reproj_error()
                if new_indices == old_indices:
                    print("New indices are the same as old indices. Continuing to the next calibration.")
                    camera_parameters.save_parameters_to_json(filename + '_intrinsic_parameters.json')
                    break  # Skip the rest of the loop and move to the next iteration
                else:
                    camera_parameters = calibrator.calibrate_indices(new_indices)
                    old_indices = new_indices
            camera_parameters.save_parameters_to_json(filename + '_intrinsic_parameters.json')
            camera_parameters.save_distortion_image(filename + '_distortion.png')
        pass
    def load_intrinsic_calibration(self,INTRINSIC_CALIBRATION_DATA_DIRECTORY = None):
        ## load intrinsic parameters from json files
        if INTRINSIC_CALIBRATION_DATA_DIRECTORY is not None:
            self.INTRINSIC_CALIBRATION_DATA_DIRECTORY = INTRINSIC_CALIBRATION_DATA_DIRECTORY
        calibrated_data = glob.glob(self.INTRINSIC_CALIBRATION_DATA_DIRECTORY+"/*_intrinsic_parameters.json")
        camera_parameters = []
        for data in calibrated_data:
            camera_parameters.append(CameraParameters())
            camera_parameters[-1].load_parameters_from_json(data)
        self.camera_parameters = camera_parameters
    def load_extrinsic_calibration(self,EXRINSIC_CALIBRATION_DATA_DIRECTORY = None):
        ## load intrinsic parameters from json files
        if EXRINSIC_CALIBRATION_DATA_DIRECTORY is not None:
            self.EXRINSIC_CALIBRATION_DATA_DIRECTORY = EXRINSIC_CALIBRATION_DATA_DIRECTORY
        calibrated_data = glob.glob(self.EXRINSIC_CALIBRATION_DATA_DIRECTORY+"/*_extrinsic_parameters_TransformationMatrix.json")
        self.H = []
        for data in calibrated_data:
            self.H.append(TransformationMatrix())
            self.H[-1].load_from_json(data)
    def __stereo_calibration(self):

        self.load_intrinsic_calibration()
        calibrated_data = glob.glob(self.EXRINSIC_CALIBRATION_DATA_DIRECTORY + "/*_scan_*.h5")
        images_list = []
        #for data in calibrated_data:
        black_images_l = self._file_loading("L_pattern_black", r'\*scan_*.h5')
        black_images_r = self._file_loading("R_pattern_black", r'\scan_*.h5') #todo, make this camera name independent
        images_list.append(black_images_l)
        images_list.append(black_images_r)
        # Perform stereo calibration
        filename = self.EXRINSIC_CALIBRATION_DATA_DIRECTORY #todo, make this naming independant of number of cams

        calibrator = StereoCalibrator()
        stereo_parameters = calibrator.calibrate(images_list[0], images_list[1],self.camera_parameters[0] , self.camera_parameters[1], self.checkerboard_size, self.boardsize)
        # %% save all detection to a file
        calibrator.save_checkerboard_detection_to_images(images_list[0], images_list[1], self.EXRINSIC_CALIBRATION_DATA_DIRECTORY + "/checkerboard_detection")
        # %% Retry calibration after removing potential outliers
        old_indices = calibrator.indices
        for i in range(10):  # do this max 10 times
            print("select outliers, exit to continue")
            new_indices = calibrator.plot_and_filter_reproj_error()
            if new_indices == old_indices:
                print("New indices are the same as old indices. Continuing to the next calibration.")
                break  # Skip the rest of the loop and move to the next iteration
            else:
                stereo_parameters = calibrator.calibrate_indices(new_indices)
                old_indices = new_indices
        stereo_parameters.info = ['L,R']  # todo make this camera name independent

        stereo_parameters.save_parameters(filename + '/_extrinsic_parameters.h5')
        print(filename + '_extrinsic_parameters.h5')
    def _file_loading_single(self, file):
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


        with h5py.File(file, 'r') as f:
            for key in f.keys():
                image = f[key][:]

                images.append(np.transpose(image, (1, 2, 0)))

        return np.stack(images, axis=-1)
    def _file_loading(self, identifier, path):
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

        files = glob.glob(self.EXRINSIC_CALIBRATION_DATA_DIRECTORY  + path)

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
    def Close(self):
        pass

        if hasattr(self.cameras, 'Close'):
            self.cameras.Close()
            print('Projector: Closed all cameras.')
            self.cameras = []
    def __del__(self):
        # check if cameras have Close function
        print('Destructor called')
        if self.cameras is None:
            return
        if hasattr(self.cameras, 'Close'):
            print('Destructor: Closed all cameras.')
            self.cameras.Close()
            self.cameras = []


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    import matplotlib
    matplotlib.use('tkagg')

    stereo = Stereo()
    stereo.calibrate_camera_intrinsics(r"../Examples/static/1_calibration_data/intrinsic")
    stereo.load_intrinsic_calibration(r"../Examples/static/1_calibration_data/intrinsic")
    stereo.calibrate_camera_extrinsics(r"../Examples/static/1_calibration_data/extrinsic",r"../Examples/static/1_calibration_data/intrinsic")
    #stereo.load_extrinsic_calibration()
    pass