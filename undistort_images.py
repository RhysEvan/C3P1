import os
import glob
from PIL import Image
import numpy as np
import cv2 as cv
from PyCamCalib.core.calibration import StereoParameters
from ..examples.inputparameters import CALIBRATION_DATA_DIRECTORY, OBJECT_DATA_DIRECTORY, IMAGE_RESOLUTION
store_name = "undistorted_pattern"
WIDTH = IMAGE_RESOLUTION[0]
HEIGHT = IMAGE_RESOLUTION[1]

def undistort():
    folders = glob.glob(CALIBRATION_DATA_DIRECTORY + r'\scan_*')
    stereo_parameters = StereoParameters()
    stereo_parameters.load_parameters(CALIBRATION_DATA_DIRECTORY + r'\stereo_parameters.h5')
    c_l = stereo_parameters.camera_parameters_1.c
    f_l = stereo_parameters.camera_parameters_1.f
    dist_rad_l = stereo_parameters.camera_parameters_1.radial_dist_coeffs
    dist_tan_l = stereo_parameters.camera_parameters_1.tangential_dist_coeffs

    mtx_left = np.array([[f_l[0], 0, c_l[0]],
                         [0, f_l[1], c_l[1]],
                         [0, 0, 1]])

    dist_l = np.array([dist_rad_l[0], dist_rad_l[1], dist_tan_l[0], dist_tan_l[1], dist_rad_l[2]])

    c_r = stereo_parameters.camera_parameters_2.c
    f_r = stereo_parameters.camera_parameters_2.f
    dist_rad_r = stereo_parameters.camera_parameters_2.radial_dist_coeffs
    dist_tan_r = stereo_parameters.camera_parameters_2.tangential_dist_coeffs

    mtx_right = np.array([[f_r[0], 0, c_r[0]],
                          [0, f_r[1], c_r[1]],
                          [0, 0, 1]])

    dist_r = np.array([dist_rad_r[0], dist_rad_r[1], dist_tan_r[0], dist_tan_r[1], dist_rad_r[2]])

    for i, folder in enumerate(folders):

        files_l = glob.glob(folder + r'\L*.tiff')
        files_r = glob.glob(folder + r'\R*.tiff')

        for j, (file_l, file_r) in enumerate(zip(files_l, files_r)):

            image_l = np.array(Image.open(file_l))
            image_r = np.array(Image.open(file_r))

            new_camera_matrix_l, roi_l = cv.getOptimalNewCameraMatrix(mtx_left, dist_l, (WIDTH, HEIGHT), 1, (WIDTH, HEIGHT))
            undistorted_l = cv.undistort(image_l, mtx_left, dist_l, None, new_camera_matrix_l)
            # Optional: Uncomment the lines below if you need to crop the image to the ROI
            #x, y, w, h = roi_l
            #undistorted_l = undistorted_l[y:y + h, x:x + w]
            cv.imwrite(os.path.join(OBJECT_DATA_DIRECTORY + fr"\scan_{i+1}", f"L_{store_name}{str(j).zfill(2)}.tiff"), undistorted_l)

            new_camera_matrix_r, roi_r = cv.getOptimalNewCameraMatrix(mtx_right, dist_r, (WIDTH, HEIGHT), 1, (WIDTH, HEIGHT))
            undistorted_r = cv.undistort(image_r, mtx_right, dist_r, None, new_camera_matrix_r)
            # Optional: Uncomment the lines below if you need to crop the image to the ROI
            #x, y, w, h = roi_r
            #undistorted_r = undistorted_r[y:y + h, x:x + w]
            cv.imwrite(os.path.join(OBJECT_DATA_DIRECTORY + fr"\scan_{i+1}", f"R_{store_name}{str(j).zfill(2)}.tiff"), undistorted_r)
