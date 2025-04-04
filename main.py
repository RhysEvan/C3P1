"""
main running sequence for projcet
"""
import os
import cv2
import h5py
import glob
import inspect
import numpy as np
import StructuredLight.Graycode as slg
import Calibration.GraycodeCalibration as cg
import Calibration.IntrinsicCalirbration as ci
import Calibration.TurnTableCalibration as ct
import DataCapture as dc
import Triangulation as tl
import config.config as cfg

import inputparameters as ip

try:
    from CameraModel.GenICam.GenICamCamera import GenICamCamera
except:
    print("no luck with module")


class Mapping:
    """
    class initiation to execute a 3d scan
    """

    def __init__(self, folder_name_pattern, cameras, texture_camera, dimensions, folder_name_capture,
                 range_steps_calib, range_steps_scan, com_port=cfg.com_port_default, obj_path=None,
                 exposure=cfg.exposure_default,
                 exposure_intrinsic=cfg.exposure_intrinsic_default):
        self.decoder = slg.Decode_Gray()

        ##########################################################################################
        ######################### generate projection pattern ####################################
        ##########################################################################################

        graycode_folder = glob.glob(ip.PROJECTOR_DIRECTORY + "/*")
        # check if folder exists
        if len(graycode_folder) == 0:
            self.graycode = slg.ProjectionPattern(ip.WIDTH, ip.HEIGHT, ip.PROJECTOR_DIRECTORY)
            self.graycode.generate_images()

        ##########################################################################################
        ######################### center camera's to projector ###################################
        ##########################################################################################
        # TODO see if calib is done recently or have a request notice to determine if the following steps should be done
        input("""Before the next step please verify that the projector is connected and turned on.
Use the camera windows to align the camera's to the best of you capabilities with the black cross projected. Press Enter to continue""")
        dc.innitiate_support(cameras, dimensions)

        ##########################################################################################
        ######################### calibrate camera intrinsics ####################################
        ##########################################################################################
        try:
            cameras[0].SetParameterDouble("ExposureTime", exposure_intrinsic)
            cameras[1].SetParameterDouble("ExposureTime", exposure_intrinsic)
        except:
            print("")

        calibrated_data = glob.glob(ip.INTRINSIC_CALIBRATION_DATA_DIRECTORY + "/*_scan.h5")

        if len(calibrated_data) == 0:
            dc.intrinsic_calibration_capture(ip.INTRINSIC_CALIBRATION_DATA_DIRECTORY,
                                             cameras, texture_camera,
                                             identifier=["L", "R"])

        calibrated_data = glob.glob(ip.INTRINSIC_CALIBRATION_DATA_DIRECTORY + "/*parameters.h5")
        if len(calibrated_data) == 0:
            self.intrinsic_calibration()

        try:
            cameras[0].SetParameterDouble("ExposureTime", exposure)
            cameras[1].SetParameterDouble("ExposureTime", exposure)
        except:
            print("")

        ##########################################################################################
        ################# rotations and translations calibration procedure #######################
        ##########################################################################################

        calibration_data = glob.glob(ip.CALIBRATION_DATA_DIRECTORY + "/*.h5")
        if len(calibration_data) == 0:
            time_hold = input("turn on projector then press enter")
            for _ in range(20):
                dc.graycode_data_capture(folder_name_pattern, cameras, texture_camera,
                                         dimensions, folder_name_capture, 0)

        calibrated_data = glob.glob(ip.CALIBRATION_DATA_DIRECTORY + "/*parameters.h5")
        if len(calibrated_data) == 0:
            self.stereo_calibration()

        ##########################################################################################
        ######################## turntable calibration procedure #################################
        ##########################################################################################

        calibrated_data = glob.glob(ip.TURNTABLE_CALIBRATION_DATA_DIRECTORY + "/scan.h5")
        input("please turn off projector")
        if len(calibrated_data) == 0:
            cameras[0].SetParameterDouble("ExposureTime", exposure_intrinsic)
            cameras[1].SetParameterDouble("ExposureTime", exposure_intrinsic)
            dc.turntable_calibration_capture(cameras, ip.TURNTABLE_CALIBRATION_DATA_DIRECTORY, range_steps_calib,
                                             com_port)

        calibrated_data = glob.glob(ip.TURNTABLE_CALIBRATION_DATA_DIRECTORY + "/*matrix_data.h5")
        if len(calibrated_data) == 0:
            self.turntable_calibration()
        try:
            cameras[0].SetParameterDouble("ExposureTime", exposure)
            cameras[1].SetParameterDouble("ExposureTime", exposure)
        except:
            print("")

        calibrated_data = glob.glob(obj_path + "*.h5")
        input("please turn on projector")
        if len(calibrated_data) == 0:
            dc.graycode_data_capture(folder_name_pattern, cameras, texture_camera, dimensions, obj_path,
                                     range_steps_scan)

        self.stereo_triangulate = tl.StereoTriangulator(ip.CALIBRATION_DATA_DIRECTORY)

        self.mono_triangulate_left = tl.MonoTriangulator(ip.IMAGE_RESOLUTION, ip.CALIBRATION_DATA_DIRECTORY, ip.WIDTH,
                                                         ip.HEIGHT,
                                                         'L')
        self.mono_triangulate_right = tl.MonoTriangulator(ip.IMAGE_RESOLUTION, ip.CALIBRATION_DATA_DIRECTORY, ip.WIDTH,
                                                          ip.HEIGHT,
                                                          'R')

        scenes = self.load_h5(obj_path)

        # Lists to collect point clouds for all scenes
        stereo_pointclouds = []
        mono_left_pointclouds = []
        mono_right_pointclouds = []

        for i, item in enumerate(scenes):
            print(i)
            left_horizontal_decoded_image, left_vertical_decoded_image = self.decoder.scene_decoder(item, "L")
            right_horizontal_decoded_image, right_vertical_decoded_image = self.decoder.scene_decoder(item, "R")

            self.stereo_pointcloud = self.stereo_triangulate.triangulate(
                left_horizontal_decoded_image, left_vertical_decoded_image,
                right_horizontal_decoded_image, right_vertical_decoded_image
            )
            camera_points, projector_points, _ = self.mono_triangulate_left.get_cam_proj_pts(None,
                                                                                             left_horizontal_decoded_image,
                                                                                             left_vertical_decoded_image)

            self.mono_left_pointcloud = self.mono_triangulate_left.triangulate_mono(camera_points,
                                                                                    projector_points)

            camera_points, projector_points, _ = self.mono_triangulate_right.get_cam_proj_pts(None,
                                                                                              right_horizontal_decoded_image,
                                                                                              right_vertical_decoded_image)

            self.mono_right_pointcloud = self.mono_triangulate_right.triangulate_mono(camera_points,
                                                                                      projector_points)

            # Append the point clouds to the respective lists
            stereo_pointclouds.append(self.stereo_pointcloud)
            mono_left_pointclouds.append(self.mono_left_pointcloud)
            mono_right_pointclouds.append(self.mono_right_pointcloud)

            ###############VISUALIZATION#############
            # visualisepointcloud(self.stereo_pointcloud)
            # visualisepointcloud(self.mono_left_pointcloud)
            # visualisepointcloud(self.mono_right_pointcloud)

        # Save all point clouds to separate npy files
        print("saving pointclouds")
        np.savez('stereo_pointclouds.npz', *stereo_pointclouds)
        np.savez('mono_left_pointclouds.npz', *mono_left_pointclouds)
        np.savez('mono_right_pointclouds.npz', *mono_right_pointclouds)

    def intrinsic_calibration(self):
        """
        Execute calibration
        """
        frame_list = list(range(cfg.cal_image_count_intrinsic))
        self.intrinsic_calibration_L = self.quality_optimisation(
            ci.IntrisicCalibration,
            frame_list,
            identifier="L",
            format_type="int_",
            visualize=True
        )

        self.intrinsic_calibration_R = self.quality_optimisation(
            ci.IntrisicCalibration,
            frame_list,
            identifier="R",
            format_type="int_",
            visualize=True
        )

        self.intrinsic_calibration_RGB = self.quality_optimisation(
            ci.IntrisicCalibration,
            frame_list,
            identifier="RGB",
            format_type="int_",
            visualize=True
        )

    def stereo_calibration(self):
        h5_path_L = ip.INTRINSIC_CALIBRATION_DATA_DIRECTORY + "/L_camera_intrinsic_parameters.h5"
        h5_path_R = ip.INTRINSIC_CALIBRATION_DATA_DIRECTORY + "/R_camera_intrinsic_parameters.h5"
        cam_int_L, cam_dist_L = self.load_parameters(h5_path_L)
        cam_int_R, cam_dist_R = self.load_parameters(h5_path_R)

        frame_list = list(range(cfg.cal_image_count_stereo_cal))

        self.calib_mono_left = self.quality_optimisation(
            cg.MonoCalibration,
            frame_list,
            self.decoder,
            cam_int_L, cam_dist_L,
            identifier="L",
            format_type="mono_"
        )

        self.calib_mono_right = self.quality_optimisation(
            cg.MonoCalibration,
            frame_list,
            self.decoder,
            cam_int_R, cam_dist_R,
            identifier="R",
            format_type="mono_"
        )

        self.calib_stereo = cg.StereoCalibration(self.decoder,
                                                 ip.CALIBRATION_DATA_DIRECTORY,
                                                 ip.CHESS_SHAPE,
                                                 ip.CHESS_BLOCK_SIZE)

    def turntable_calibration(self):
        h5_path_L = ip.INTRINSIC_CALIBRATION_DATA_DIRECTORY + "/L_camera_intrinsic_parameters.h5"
        h5_path_R = ip.INTRINSIC_CALIBRATION_DATA_DIRECTORY + "/R_camera_intrinsic_parameters.h5"

        cam_int_L, cam_dist_L = self.load_parameters(h5_path_L)
        cam_int_R, cam_dist_R = self.load_parameters(h5_path_R)

        turntable_L = ct.TurnTableCalibration(ip.TURNTABLE_CALIBRATION_DATA_DIRECTORY, cam_int_L,
                                              cam_dist_L, ip.SQUARES_X, ip.SQUARES_Y, ip.SQUARES_LENGTH,
                                              ip.MARKER_LENGTH, "L")
        turntable_L.calibrate()

        turntable_R = ct.TurnTableCalibration(ip.TURNTABLE_CALIBRATION_DATA_DIRECTORY, cam_int_R,
                                              cam_dist_R, ip.SQUARES_X, ip.SQUARES_Y, ip.SQUARES_LENGTH,
                                              ip.MARKER_LENGTH, "R")
        turntable_R.calibrate()

    def load_parameters(self, path):
        calibration_paths = glob.glob(path)
        h5_file_path = calibration_paths[0]
        with h5py.File(h5_file_path, 'r') as h5_file:
            # Iterate over each key in the file
            calibration = h5_file["camera_calibration/camera_parameters"]
            c = calibration["c"]
            print(c[:])
            f = calibration["f"]
            print(f[:])
            rad_dst = calibration["radial_dist_coeffs"]
            print(rad_dst[:])
            tan_dst = calibration["tangential_dist_coeffs"]
            print(tan_dst[:])
            cam_in = [
                [f[0], 0, c[0]],
                [0, f[1], c[1]],
                [0, 0, 1]
            ]
            cam_dist = [rad_dst[0], rad_dst[1], tan_dst[0], tan_dst[1], rad_dst[2]]
        return np.array(cam_in), np.array(cam_dist)

    def load_h5(self, path):
        scenes = []
        h5_file_paths = glob.glob(path + '*')
        for _, path in enumerate(h5_file_paths):
            if ".h5" in path:
                scenes.append(self.load_object(path))
            else:
                continue
        return scenes

    def load_object(self, scene_path):
        if not os.path.exists(scene_path):
            raise FileNotFoundError(f"The file {scene_path} does not exist.")

        scene_data = {}
        with h5py.File(scene_path, 'r') as h5f:
            for key in h5f.keys():
                scene_data[key] = np.array(h5f[key])
        return scene_data

    def quality_optimisation(self, class_object, frame_list, *args, **kwargs):
        """
        Allows iterative optimization of calibration quality by dynamically updating `frame_list`.

        Parameters:
            class_object: The calibration class or function to be called (e.g., IntrisicCalibration).
            frame_list: The initial list of frames to use for calibration (will be updated dynamically).
            *args: Additional positional arguments required by the calibration class.
            **kwargs: Additional keyword arguments required by the calibration class.
        
        Returns:
            The final updated calibration object.
        """
        # Get the constructor signature
        signature = inspect.signature(class_object)
        valid_kwargs = {}

        # Loop through the signature parameters and only keep kwargs that match
        for param in signature.parameters.values():
            if param.kind in [inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY]:
                if param.name in kwargs:
                    valid_kwargs[param.name] = kwargs[param.name]

        # Call the provided class or function with the current frame list and other parameters
        calibration = class_object(
            frame_list,
            ip.INTRINSIC_CALIBRATION_DATA_DIRECTORY,
            ip.CHESS_SHAPE,
            ip.CHESS_BLOCK_SIZE,
            ip.WIDTH,
            ip.HEIGHT,
            ip.IMAGE_RESOLUTION,
            **valid_kwargs
        )

        # Save all frames initially
        identifier = kwargs.get("identifier", "default")
        format_type = kwargs.get("format_type", "default")
        save_selected_frames(format_type, identifier, frame_list)

        while True:

            # Perform the task (e.g., visualization or other operations in `class_object`)
            print(f"Calibration performed with frames: {frame_list}")

            # Ask the user if they are satisfied with the results
            satisfied = input("Are you satisfied with the results? (y/n): ").strip().lower()
            if satisfied == "y":
                # Save the frame list and return the final calibration object
                print("Calibration finalized.")
                return calibration

            # Allow the user to modify the list
            frame_list = modify_saved_frames(format_type, identifier, frame_list)


def modify_saved_frames(type_format, identifier, frame_list):
    """
    Removes unwanted frames from the saved file instead of reprocessing.
    """
    while True:
        print(f"Current frame list: {frame_list}")
        modify = input("Enter frames to remove (comma-separated) or 'done' to keep current list: ").strip()
        if modify.lower() == "done":
            break
        try:
            to_remove = list(map(int, modify.split(",")))
            frame_list = [frame for frame in frame_list if frame not in to_remove]
            save_selected_frames(type_format, identifier, frame_list)  # Overwrite with the new list
            print(f"Updated frame list saved: {frame_list}")
        except ValueError:
            print("Invalid input. Please enter comma-separated integers.")
    return frame_list


def save_selected_frames(type_format, identifier, frame_list):
    """
    Saves the selected frames list to a file.
    """
    filename = f"{type_format}selected_frames_{identifier}.txt"
    with open(filename, "w") as file:
        file.write(",".join(map(str, frame_list)))
    print(f"Frame list saved as {filename}")


if __name__ == "__main__":
    folder_name_pattern = cfg.patterns_folder
    folder_name_capture = cfg.capture_folder
    OBJECT_PATH = cfg.object_path

    range_steps_calib = cfg.steps_cal  # 1 if no turn table
    range_steps_scan = cfg.steps_scan  # 1 if no turn table
    com_port = cfg.com_port

    gain = cfg.gain
    exposure_intrinsic = cfg.exposure_intrinsic
    exposure_procam = cfg.exposure_procam
    try:
        cameras = []
        camera_ids = [cfg.cam1_id, cfg.cam2_id]

        for camera_id in camera_ids:
            cam = GenICamCamera(cfg.cam_interface)
            cam.Open(camera_id)
            cam.SetParameterDouble("ExposureTime", exposure_procam)
            cam.SetParameterDouble("Gain", gain)
            cam.Start()
            cameras.append(cam)

        # Get camera dimensions for fixed width calculation
        fixed_width = cfg.cam_fixed_width
        dimensions = []

        for cam in cameras:
            frame = cam.GetFrame()
            h, w = frame.shape[:2]
            ratio = fixed_width / float(w)
            dim = (fixed_width, int(h * ratio))
            dimensions.append(dim)

        fixed_width = cfg.cam_fixed_width
        frame_l = cameras[0].GetFrame()
        frame_r = cameras[1].GetFrame()
        (h_l, w_l) = frame_l.shape[:2]
        (h_r, w_r) = frame_r.shape[:2]

        ratio_l = fixed_width / float(w_l)
        dim_l = (fixed_width, int(h_l * ratio_l))
        ratio_r = fixed_width / float(w_r)
        dim_r = (fixed_width, int(h_r * ratio_r))

        dimensions = [dim_l, dim_r]
        print(dimensions)
    except:
        cameras = []
        dimensions = [(cfg.cam_default_width, cfg.cam_default_height), (cfg.cam_default_width, cfg.cam_default_height)]

    texture_camera = cv2.VideoCapture(0)  # 0 for the default camera

    if not texture_camera.isOpened():
        print("Error: Could not open camera.")

    MAP = Mapping(folder_name_pattern, cameras, texture_camera, dimensions,
                  folder_name_capture, range_steps_calib, range_steps_scan, com_port,
                  OBJECT_PATH, exposure_procam, exposure_intrinsic)

    # Stop cameras
    cameras = []

    if cameras:
        for cam in cameras:
            cam.Stop()
            cam.Close()
    print("done :)")
