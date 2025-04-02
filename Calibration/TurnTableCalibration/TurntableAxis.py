import cv2
import os
import numpy as np
import h5py
import re
from scipy.linalg import svd


class TurnTableCalibration():
    def __init__(self, path_to_charuco, cam_mtx, cam_dst,
                 squares_x, squares_y, square_length, marker_length, identifier= None):

        self.camera_matrix =cam_mtx
        self.dist_coeffs = cam_dst
        self.square_length = square_length
        self.identifier = identifier

        self.charuco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.charuco_board = cv2.aruco.CharucoBoard((squares_x, squares_y), self.square_length, marker_length, self.charuco_dict)
        self.charuco_board.setLegacyPattern(True)
        self.params = cv2.aruco.DetectorParameters()
        size_ratio = squares_y / squares_x
        img = cv2.aruco.CharucoBoard.generateImage(self.charuco_board, (640, int(640*size_ratio)), marginSize=20)
        cv2.imshow("img", img)
        cv2.waitKey(2000)

        self.poses = []  # To store the rotation and translation vectors for each image
        self.charuco_points = []
        if identifier is not None:
            self.load_file(path_to_charuco, identifier)
        else:
            print("can't not have an id")
            return

    def load_file(self, path, identifier):
        self.images = []
        file_dir = os.path.join(path, "scan.h5")

        with h5py.File(file_dir, 'r') as f:
            # Define a function to extract the numerical part for sorting
            def extract_number(key):
                # Use regex to find the numeric part, ignoring the prefix (L_ or R_)
                match = re.search(r'_(\d+(\.\d+)?)$', key)  # Match the number after the last underscore
                return float(match.group(1)) if match else float('inf')  # Return numeric value for sorting

            keys = sorted(list(f.keys()), key=extract_number)
            for key in keys:
                if identifier in key:
                    image = f[key][:]
                    self.images.append(np.squeeze(image))

    def calibrate(self):
        for image in self.images:
            image_copy = image.copy()
            corners, ids, _ = cv2.aruco.detectMarkers(image_copy, self.charuco_dict, parameters=self.params)
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(image_copy, corners, ids)
                cv2.imshow("Detected Markers", cv2.resize(image_copy, (1200, 800)))
                cv2.waitKey(50)

                ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, image,
                                                                                        self.charuco_board, cameraMatrix=self.camera_matrix,
                                                                                        distCoeffs=self.dist_coeffs)

                if ret:
                    # Optionally apply cornerSubPix only if needed
                    if charuco_corners is not None:
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                        refined_corners = cv2.cornerSubPix(image, charuco_corners, (11, 11), (-1, -1), criteria)
                        self.charuco_points.append(refined_corners)

                    retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(refined_corners, charuco_ids,
                                                                            self.charuco_board,
                                                                            self.camera_matrix, self.dist_coeffs,
                                                                            None, None)
                    if retval:
                        rvec, _ = cv2.Rodrigues(rvec)

                        self.poses.append((rvec, tvec))

                        cv2.drawFrameAxes(image, self.camera_matrix,
                                          self.dist_coeffs, rvec, tvec,
                                          length=10, thickness=15)
                        cv2.imshow("Draw axes", cv2.resize(image, (1200,800)))
                        cv2.waitKey(50)

        # Stack the rotation matrices into an array
        # Extract rotation matrices from poses
        rotation_matrices = [pose[0] for pose in self.poses]  # This extracts rvecs (3x3 matrices)
        translation_matrices = [pose[1] for pose in self.poses]

        # Stack the rotation matrices along a new axis
        stacked_rotations = np.array(rotation_matrices).transpose(1,2,0)  # This will give you a 3x3xN matrix
        stacked_translations = np.array(translation_matrices).transpose(1,2,0)  # This will give you a 3x3xN matrix
        rotation_axis = np.mean(stacked_rotations, axis=2)
        translation_axis = np.mean(stacked_translations, axis=2)
        filename = f"{self.identifier}_turntable_matrix_data.h5"

        # Open the HDF5 file in write mode
        with h5py.File(filename, 'w') as h5file:
            # Store each set of charuco points in separate datasets
            for idx, charuco_points in enumerate(self.charuco_points):
                h5file.create_dataset(f'charuco_points_{idx}', data=charuco_points)
            h5file.create_dataset('stacked_rotations', data=stacked_rotations)
            h5file.create_dataset('stacked_translations', data=stacked_translations)
            h5file.create_dataset('rotation_matrix', data=rotation_axis)
            h5file.create_dataset('translation_matrix', data=translation_axis)

        print(f"Data saved to {filename}")
        cv2.destroyAllWindows()
    