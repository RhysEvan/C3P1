import os
import h5py
import cv2 as cv
import numpy as np
import open3d as o3d

from calibration_toolbox_python.src.PyCamCalib.core.calibration import StereoParameters


class StereoTriangulator:
    """
    class instance for triangulating two images to determine the depth.
    """
    def __init__(self, CALIBRATION_DATA_DIRECTORY):

        self.calibration_data_directory = CALIBRATION_DATA_DIRECTORY

        self.array_vert_l_masked = []
        self.array_hor_l_masked = []
        self.array_vert_r_masked = []
        self.array_hor_r_masked = []

        self.testarray = []
        self.colors = []

        camera_matrx_l, camera_matrix_r, rot, trans = self.setup_params()

        self.inverse_matrix_left_cam = []
        self.inverse_matrix_right_cam = []

        self.p1 = camera_matrx_l @ np.concatenate([np.eye(3), [[0], [0], [0]]], axis=-1)
        self.p2 = camera_matrix_r @ np.concatenate([rot, trans], axis=-1)

    def setup_params(self):
        stereo_parameters = StereoParameters()
        stereo_parameters.load_parameters(self.calibration_data_directory + r'stereo_parameters.h5')
        c_l = stereo_parameters.camera_parameters_1.c
        f_l = stereo_parameters.camera_parameters_1.f

        mtx_left = np.array([[f_l[0], 0, c_l[0]],
                             [0, f_l[1], c_l[1]],
                             [0, 0, 1]])

        c_r = stereo_parameters.camera_parameters_2.c
        f_r = stereo_parameters.camera_parameters_2.f

        mtx_right = np.array([[f_r[0], 0, c_r[0]],
                              [0, f_r[1], c_r[1]],
                              [0, 0, 1]])

        r = stereo_parameters.R
        t = stereo_parameters.T

        return mtx_left, mtx_right, r, t

    import numpy as np
    def process_frames_fast(self, inverse_left_cam, inverse_right_cam, shape_invers_left, shape_invers_right, testarray):
        min_rows = min(shape_invers_left[0], shape_invers_right[0]) - 1
        min_cols = min(shape_invers_left[1], shape_invers_right[1]) - 1
        index = 0

        for ii in range(min_rows):
            for jj in range(min_cols):
                index += 1
                leftframe = inverse_left_cam[ii, jj]
                rightframe = inverse_right_cam[ii, jj]

                if all(leftframe) and all(rightframe):
                    self.point_localizer(leftframe, rightframe, index)
                elif all(leftframe):
                    neighbors = [
                        inverse_right_cam[ii, jj + 1],
                        inverse_right_cam[ii, jj - 1],
                        inverse_right_cam[ii + 1, jj],
                        inverse_right_cam[ii - 1, jj]
                    ]
                    for rightframe in neighbors:
                        if all(rightframe):
                            self.point_localizer(leftframe, rightframe, index)
                            break
                elif all(rightframe):
                    neighbors = [
                        inverse_left_cam[ii, jj + 1],
                        inverse_left_cam[ii, jj - 1],
                        inverse_left_cam[ii + 1, jj],
                        inverse_left_cam[ii - 1, jj]
                    ]
                    for leftframe in neighbors:
                        if all(leftframe):
                            self.point_localizer(leftframe, rightframe, index)
                            break
    def process_frames(self,inverse_left_cam, inverse_right_cam, shape_invers_left, shape_invers_right, testarray):
        index = 0
        for ii in range(0, min(shape_invers_left[0], shape_invers_right[0]) - 1):
            for jj in range(0, min(shape_invers_left[1], shape_invers_right[1]) - 1):
                flag = False
                index = index + 1
                leftframe = inverse_left_cam[ii][jj]
                rightframe = inverse_right_cam[ii][jj]

                if leftframe[0] != 0 and leftframe[1] != 0 and rightframe[0] != 0 and rightframe[1] != 0:
                    self.point_localizer(leftframe, rightframe, index)
                    # self.array_colorizer(img, leftframe)

                elif leftframe[0] != 0 and leftframe[1] != 0 and rightframe[0] == 0 and rightframe[1] == 0:

                    rightframe = inverse_right_cam[ii][jj + 1]

                    if rightframe[0] != 0 and rightframe[1] != 0:
                        self.point_localizer(leftframe, rightframe, index)
                        flag = True
                        # self.array_colorizer(img, leftframe)

                    rightframe = inverse_right_cam[ii][jj - 1]

                    if rightframe[0] != 0 and rightframe[1] != 0 and flag == False:
                        self.point_localizer(leftframe, rightframe, index)
                        flag = True
                        # self.array_colorizer(img, leftframe)

                    rightframe = inverse_right_cam[ii + 1][jj]

                    if rightframe[0] != 0 and rightframe[1] != 0 and flag == False:
                        self.point_localizer(leftframe, rightframe, index)
                        flag = True
                        # self.array_colorizer(img, leftframe)

                    rightframe = inverse_right_cam[ii - 1][jj]

                    if rightframe[0] != 0 and rightframe[1] != 0 and flag == False:
                        self.point_localizer(leftframe, rightframe, index)
                        flag = True
                        # self.array_colorizer(img, leftframe)

                elif leftframe[0] == 0 and leftframe[1] == 0 and rightframe[0] != 0 and rightframe[1] != 0:

                    leftframe = inverse_left_cam[ii][jj + 1]

                    if leftframe[0] != 0 and leftframe[1] != 0:
                        self.point_localizer(leftframe, rightframe, index)
                        flag = True
                        # self.array_colorizer(img, leftframe)

                    leftframe = inverse_left_cam[ii][jj - 1]

                    if leftframe[0] != 0 and leftframe[1] != 0 and flag == False:
                        self.point_localizer(leftframe, rightframe, index)
                        flag = True
                        # self.array_colorizer(img, leftframe)

                    leftframe = inverse_left_cam[ii + 1][jj]

                    if leftframe[0] != 0 and leftframe[1] != 0 and flag == False:
                        self.point_localizer(leftframe, rightframe, index)
                        flag = True
                        # self.array_colorizer(img, leftframe)

                    leftframe = inverse_left_cam[ii - 1][jj]

                    if leftframe[0] != 0 and leftframe[1] != 0 and flag == False:
                        self.point_localizer(leftframe, rightframe, index)
                        flag = True
                        # self.array_colorizer(img, leftframe)

    def triangulate(self,
                    array_left_hor_mask, array_left_vert_mask,
                    array_right_hor_mask, array_right_vert_mask
                    ):
        print("TODO insert new array implementation for correspondence.")
        print("TODO create array that does everything in for loop with appends")
        print("TODO once completed arrays are made (left and right) insert into point localizer function for cv2 parallel efficiency!")
        self.array_vert_l_masked = array_left_vert_mask
        self.array_hor_l_masked = array_left_hor_mask
        self.array_vert_r_masked = array_right_vert_mask
        self.array_hor_r_masked = array_right_hor_mask

        print("Calculating correspondence")
        inverse_left_cam, inverse_right_cam = self.find_correspondence()
        self.testarray = np.zeros((min(np.shape(inverse_left_cam)[0], np.shape(inverse_right_cam)[0]) * min(np.shape(inverse_left_cam)[1], np.shape(inverse_right_cam)[1]), 3))
        self.colors = []

        shape_invers_left = np.shape(inverse_left_cam)
        shape_invers_right = np.shape(inverse_right_cam)
        print("Triangulating points")
        import time
        start = time.time()
        self.process_frames_fast(inverse_left_cam, inverse_right_cam, shape_invers_left, shape_invers_right, self.testarray)
        print("Time triangulation: ", time.time()-start)

        colors = np.array(self.colors)               ## Matching RGB values

        #xyz = np.array(testarray)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(self.testarray))
        ## visualise pcd
        #o3d.visualization.draw_geometries([pcd])
        pc = np.asarray(pcd.points, dtype=object)

        return pc

    def point_localizer(self, leftframe, rightframe, index = None):
        """
        Creating all the 3d points for frame couples
        """
        projpoints1 = np.array([[leftframe[0],leftframe[1]],
                                [rightframe[0], rightframe[1]]], dtype=np.float32)

        points4d = cv.triangulatePoints(self.p1, self.p2, projpoints1[0], projpoints1[1])
        points3d = points4d[:3] / points4d[3]
        if index == None:
            self.testarray.append(points3d[:3])
        else:
            self.testarray[index] = points3d.flatten()


    def array_colorizer(self, img, frame):
        """
        Connecting a colour value to the points
        """
        a = frame[0]
        b = frame[1]
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        self.colors.append(img[int(b)][int(a)] / 255)

    def find_correspondence(self):
        ## Find max values to create empty (0) matrices to use
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

class MonoTriangulator():
    def __init__(self, IMAGE_RESOLUTION, CALIBRATION_DATA_DIRECTORY, WIDTH, HEIGHT, identifier = None):

        self.calibration_data_directory = CALIBRATION_DATA_DIRECTORY
        self.width = WIDTH
        self.height = HEIGHT
        self.image_resolution = IMAGE_RESOLUTION

        calib_data = self.setup_params(identifier)
        self.cam_mtx = calib_data['cam_int']
        self.cam_dist = calib_data['cam_dist']
        self.proj_mtx = calib_data['proj_int']
        self.proj_dist = calib_data['proj_dist']
        self.proj_R = calib_data['R']
        self.proj_R = self.proj_R.T
        self.proj_T = calib_data['T']
        self.proj_T = self.proj_T.flatten()


    def setup_params(self, identifier):
        if identifier is None:
            path = self.calibration_data_directory + r'mono_parameters.h5'
        else:
            path = self.calibration_data_directory + fr'{identifier}_mono_parameters.h5'

        if not os.path.exists(path):
            raise FileNotFoundError(f"The file {path} does not exist.")

        data = {}
        with h5py.File(path, 'r') as h5f:
            for key in h5f.keys():
                data[key] = np.array(h5f[key])
        return data

    def get_cam_proj_pts(self, image_white,
                         horizontal_pixels, vertical_pixels):
        """
        Get 2D points from camera and projector with their color

        Returns:
            cam_pts: 2D points from camera
            proj_pts: 2D points from projector
            colors: colors of the points
        """
        camera_width = self.image_resolution[1]
        camera_height = self.image_resolution[0]

        projector_resolution_width = self.width
        projector_resolution_height = self.height

        projector_3d_points = {}
        camera_3d_points = {}
        colors_per_point = []

        outer_collumn = projector_resolution_width
        outer_row = projector_resolution_height

        for w in range(camera_width):

            for h in range(camera_height):

                h_value = horizontal_pixels[h, w]
                v_value = vertical_pixels[h, w]

                if h_value == 0 or v_value == 0:

                    pass

                else:

                    proj_point = (v_value / 1, h_value / 1)
                    index = int(proj_point[1]) * outer_collumn + int(proj_point[0])

                    if index not in projector_3d_points:

                        projector_3d_points[index] = proj_point
                        camera_3d_points[index] = []
                        camera_3d_points[index].append((w, h))


        if image_white is not None:

            colors_per_point.append(image_white[h, w, :])

        return camera_3d_points, projector_3d_points, np.array(colors_per_point)


    def triangulate_mono(self, camera_points, projector_points):

        result = []
        dist = []

        for n, (index, projector_point) in enumerate(projector_points.items()):

            cam_point_list = camera_points[index]
            camera_set = np.mean([np.array(p) for p in cam_point_list],axis=0)
            projector_set = (projector_point[0], projector_point[1])

            # To image camera coordinates
            camera_pixel = np.array([[camera_set[0], camera_set[1]]], dtype=np.float64)
            projector_pixel = np.array([[projector_set[0], projector_set[1]]], dtype=np.float64)

            undistorted_camera_pixel = cv.undistortPoints(camera_pixel, self.cam_mtx, self.cam_dist)
            undistorted_projector_pixel = cv.undistortPoints(projector_pixel, self.proj_mtx, self.proj_dist)

            camera_vector = undistorted_camera_pixel[0, 0]
            projector_vector = undistorted_projector_pixel[0, 0]

            u1 = np.array([camera_vector[0], camera_vector[1], 1.0])
            u2 = np.array([projector_vector[0], projector_vector[1], 1.0])

            # To world coordinates
            w1 = u1
            w2 = np.dot(self.proj_R, (u2 - self.proj_T))

            # World rays
            v1 = w1
            v2 = np.dot(self.proj_R, u2)

            # Compute ray-ray approximate intersection
            p, distance = self.approximate_ray_intersection(v1, w1, v2, w2)

            if distance >= 5:

                continue

            else:

                result.append(p)
                dist.append(distance)

        xyz = result

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pc = np.asarray(pcd.points, dtype=object)
        return pc

    def approximate_ray_intersection(self, v1, q1, v2, q2):

        V = np.array([
            [np.dot(v1, v1), -np.dot(v1, v2)],
            [-np.dot(v2, v1), np.dot(v2, v2)]
        ])

        Vinv = np.linalg.inv(V)

        q2_q1 = q2 - q1
        Q1 = np.dot(v1, q2_q1)
        Q2 = -np.dot(v2, q2_q1)

        lambda1, lambda2 = np.dot(Vinv, [Q1, Q2])

        p1 = lambda1 * v1 + q1
        p2 = lambda2 * v2 + q2

        p = 0.5 * (p1 + p2)
        distance = np.linalg.norm(p2 - p1)

        return p, distance
