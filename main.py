"""
main running sequence for projcet
"""
import os
import cv2
# import time
import h5py
import glob
import inspect
import numpy as np

try:
    from CameraModel.GenICam.GenICamCamera import GenICamCamera
except:
    print("no luck with module")
import StructuredLight.Graycode as slg
import Calibration.GraycodeCalibration as cg
import Calibration.IntrinsicCalirbration as ci
import Calibration.TurnTableCalibration as ct  
import DataCapture as dc
import Triangulation as tl

from inputparameters import PROJECTOR_DIRECTORY
from inputparameters import CALIBRATION_DATA_DIRECTORY, INTRINSIC_CALIBRATION_DATA_DIRECTORY, \
    TURNTABLE_CALIBRATION_DATA_DIRECTORY
from inputparameters import WIDTH, HEIGHT
from inputparameters import IMAGE_RESOLUTION
from inputparameters import CHESS_SHAPE, CHESS_BLOCK_SIZE
from inputparameters import SQUARES_X, SQUARES_Y
from inputparameters import SQUARES_LENGTH, MARKER_LENGTH


class Mapping():
    """
    class initiation to execute a 3d scan
    """
    #TODO: yaml magic
    def __init__(self, folder_name_pattern, cameras, dimensions, folder_name_capture,
                 range_steps_calib, range_steps_scan, com_port='COM4', obj_path=None, exposure=2500,
                 exposure_intrinsic=100000):
        self.cameras = cameras
        self.decoder = slg.Decode_Gray()

        ##########################################################################################
        ######################### generate projection pattern ####################################
        ##########################################################################################

        graycode_folder = glob.glob(PROJECTOR_DIRECTORY + "/*")
        # check fi folder exists
        if len(graycode_folder) == 0:
            self.graycode = slg.ProjectionPattern(WIDTH, HEIGHT, PROJECTOR_DIRECTORY)
            self.graycode.generate_images()

        ##########################################################################################
        ######################### center camera's to projector ###################################
        ##########################################################################################
        # TODO see if calib is done recently or have a request notice to determine if the following steps should be done
#         input("""Before the next step please verify that the projector is connected and turned on.
# Use the camera windows to align the camera's to the best of you capabilities with the black cross projected. Press Enter to continue""")
#         dc.innitiate_support(cameras, dimensions)

        ##########################################################################################
        ######################### calibrate camera intrinsics ####################################
        ##########################################################################################
        try:
            cameras[0].SetParameterDouble("ExposureTime", exposure_intrinsic)
            cameras[1].SetParameterDouble("ExposureTime", exposure_intrinsic)
        except:
            print("")

        calibrated_data = []
        calibrated_data = glob.glob(INTRINSIC_CALIBRATION_DATA_DIRECTORY + "/*_scan.h5")

        if len(calibrated_data) == 0:
            dc.intrinsic_calibration_capture(INTRINSIC_CALIBRATION_DATA_DIRECTORY,
                                             cameras, identifier=["L", "R", "Texture"])

        calibrated_data = []
        calibrated_data = glob.glob(INTRINSIC_CALIBRATION_DATA_DIRECTORY + "/*parameters.h5")
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

        calibrated_data = []
        calibration_data = glob.glob(CALIBRATION_DATA_DIRECTORY + "/*.h5")
        if len(calibration_data) == 0:
            time_hold = input("turn on projector then press enter")
            for _ in range(20):
                dc.graycode_data_capture(folder_name_pattern, cameras,
                                         dimensions, folder_name_capture, 0)

        calibrated_data = []
        calibrated_data = glob.glob(CALIBRATION_DATA_DIRECTORY + "/*parameters.h5")
        if len(calibrated_data) == 0:
            self.stereo_calibration()

        ##########################################################################################
        ######################## turntable calibration procedure #################################
        ##########################################################################################

        calibrated_data = []
        calibrated_data = glob.glob(TURNTABLE_CALIBRATION_DATA_DIRECTORY + "/scan.h5")
        input("please turn off projector")
        if len(calibrated_data) == 0:
            cameras[0].SetParameterDouble("ExposureTime", exposure_intrinsic)
            cameras[1].SetParameterDouble("ExposureTime", exposure_intrinsic)
            dc.turntable_calibration_capture(cameras, TURNTABLE_CALIBRATION_DATA_DIRECTORY, range_steps_calib, com_port)

        calibrated_data = []
        calibrated_data = glob.glob(TURNTABLE_CALIBRATION_DATA_DIRECTORY + "/*matrix_data.h5")
        if len(calibrated_data) == 0:
            self.turntable_calibration()
        try:
            cameras[0].SetParameterDouble("ExposureTime", exposure)
            cameras[1].SetParameterDouble("ExposureTime", exposure)
        except:
            print("")

        calibrated_data = []
        calibrated_data = glob.glob(obj_path + "*.h5")
        input("please turn on projector")
        if len(calibrated_data) == 0:
            dc.graycode_data_capture(folder_name_pattern, cameras, dimensions, obj_path,
                                     range_steps_scan)

        self.stereo_triangulate = tl.StereoTriangulator(CALIBRATION_DATA_DIRECTORY)

        self.mono_triangulate_left = tl.MonoTriangulator(IMAGE_RESOLUTION, CALIBRATION_DATA_DIRECTORY, WIDTH, HEIGHT,
                                                         'L')
        self.mono_triangulate_right = tl.MonoTriangulator(IMAGE_RESOLUTION, CALIBRATION_DATA_DIRECTORY, WIDTH, HEIGHT,
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
        #TODO: yaml magic
        image_count = 15
        frame_list = list(range(image_count))
        self.intrinsic_calibration_L = self.quality_optimisation(
            ci.IntrisicCalibration,
            frame_list,
            INTRINSIC_CALIBRATION_DATA_DIRECTORY,
            CHESS_SHAPE,
            CHESS_BLOCK_SIZE,
            WIDTH,
            HEIGHT,
            IMAGE_RESOLUTION,
            identifier="L",
            format_type="int_",
            visualize=True
        )

        self.intrinsic_calibration_R = self.quality_optimisation(
            ci.IntrisicCalibration,
            frame_list,
            INTRINSIC_CALIBRATION_DATA_DIRECTORY,
            CHESS_SHAPE,
            CHESS_BLOCK_SIZE,
            WIDTH,
            HEIGHT,
            IMAGE_RESOLUTION,
            identifier="R",
            format_type="int_",
            visualize=True
        )

        self.intrinsic_calibration_RGB = self.quality_optimisation(
            ci.IntrisicCalibration,
            frame_list,
            INTRINSIC_CALIBRATION_DATA_DIRECTORY,
            CHESS_SHAPE,
            CHESS_BLOCK_SIZE,
            WIDTH,
            HEIGHT,
            IMAGE_RESOLUTION,
            identifier="Texture",
            format_type="int_",
            visualize=True
        )

    def stereo_calibration(self):
        h5_path_L = INTRINSIC_CALIBRATION_DATA_DIRECTORY+"/L_camera_intrinsic_parameters.h5"
        h5_path_R = INTRINSIC_CALIBRATION_DATA_DIRECTORY+"/R_camera_intrinsic_parameters.h5"
        cam_int_L, cam_dist_L = self.load_parameters(h5_path_L)
        cam_int_R, cam_dist_R = self.load_parameters(h5_path_R)

        image_count = 20
        frame_list = list(range(image_count))

        self.calib_mono_left = self.quality_optimisation(
            cg.MonoCalibration,
            frame_list,
            self.decoder,
            CALIBRATION_DATA_DIRECTORY,
            CHESS_SHAPE,
            CHESS_BLOCK_SIZE,
            WIDTH,HEIGHT,
            IMAGE_RESOLUTION,
            cam_int_L, cam_dist_L,
            identifier="L",
            format_type = "mono_"
        )

        self.calib_mono_right = self.quality_optimisation(
            cg.MonoCalibration,
            frame_list,
            self.decoder,
            CALIBRATION_DATA_DIRECTORY,
            CHESS_SHAPE,
            CHESS_BLOCK_SIZE,
            WIDTH,HEIGHT,
            IMAGE_RESOLUTION,
            cam_int_R, cam_dist_R,
            identifier="R",
            format_type = "mono_"
        )

        self.calib_stereo = cg.StereoCalibration(self.decoder,
                                              CALIBRATION_DATA_DIRECTORY,
                                              CHESS_SHAPE,
                                              CHESS_BLOCK_SIZE)

    def turntable_calibration(self):
        h5_path_L = INTRINSIC_CALIBRATION_DATA_DIRECTORY + "/L_camera_intrinsic_parameters.h5"
        h5_path_R = INTRINSIC_CALIBRATION_DATA_DIRECTORY + "/R_camera_intrinsic_parameters.h5"

        cam_int_L, cam_dist_L = self.load_parameters(h5_path_L)
        cam_int_R, cam_dist_R = self.load_parameters(h5_path_R)

        turntable_L = ct.TurnTableCalibration(TURNTABLE_CALIBRATION_DATA_DIRECTORY, cam_int_L,
                                              cam_dist_L, SQUARES_X, SQUARES_Y, SQUARES_LENGTH,
                                              MARKER_LENGTH, "L")
        turntable_L.calibrate()

        turntable_R = ct.TurnTableCalibration(TURNTABLE_CALIBRATION_DATA_DIRECTORY, cam_int_R,
                                              cam_dist_R, SQUARES_X, SQUARES_Y, SQUARES_LENGTH,
                                              MARKER_LENGTH, "R")
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

        while True:
            # Call the provided class or function with the current frame list and other parameters
            calibration = class_object(frame_list, *args, **valid_kwargs)

            # Perform the task (e.g., visualization or other operations in `class_object`)
            print(f"Calibration performed with frames: {frame_list}")

            # Ask the user if they are satisfied with the results
            satisfied = input("Are you satisfied with the results? (y/n): ").strip().lower()
            if satisfied == "y":
                # Save the frame list and return the final calibration object
                identifier = kwargs.get("identifier", "default")
                format_type = kwargs.get("format_type", "default")
                save_selected_frames(format_type, identifier, frame_list)
                print("Calibration finalized.")
                return calibration

            # Allow the user to modify the list
            print(f"Current frame list: {frame_list}")
            modify = input("Enter frames to remove (comma-separated) or 'done' to keep current list: ").strip()
            if modify.lower() != "done":
                try:
                    to_remove = list(map(int, modify.split(",")))
                    frame_list = [frame for frame in frame_list if frame not in to_remove]
                    print(f"Updated frame list: {frame_list}")
                except ValueError:
                    print("Invalid input. Please enter comma-separated integers.")
    def Close(self):
        for cam in self.cameras:
            cam.Close()
        self.cameras = []
        print('MultiCam: Closed all cameras.')
    def __del__(self):
        print("MultiCam: Destructor called.")
        for cam in self.cameras:
            cam.Close()


def save_selected_frames(type_format, identifier, frame_list):
    filename = f"{type_format}selected_frames_{identifier}.txt"
    with open(filename, "w") as file:
        file.write(",".join(map(str, frame_list)))
    print(f"Frame list saved as {filename}")


if __name__ == "__main__":
    #TODO: yaml magic
    folder_name_pattern = r"./static/0_projection_pattern/"
    folder_name_capture = r"./static/1_calibration_data/"
    OBJECT_PATH = r'.\static\2_object_data\Bart/'

    #TODO: yaml magic
    range_steps_calib = 4  # 1 if no turn table
    range_steps_scan = 4  # 1 if no turn table
    com_port = 'COM3'

    #TODO: yaml magic
    gain = 3
    exposure_intrinsic = 75000
    exposure_procam = 20000
    #TODO: yaml magic
    camera_id = "2BA200003744"
    cam_l = GenICamCamera('pleora')
    cam_l.Open(camera_id)

    #TODO: yaml magic
    camera_id = "2BA200003745"
    cam_r = GenICamCamera('pleora')
    cam_r.Open(camera_id)

    #TODO: yaml magic
    camera_id = "2BA200004266"
    cam_texture = GenICamCamera('pleora')
    cam_texture.Open(camera_id)

    cameras = [cam_l, cam_r, cam_texture]
    try:
        for cam in cameras:
            cam.SetParameterDouble("ExposureTime", exposure_procam)
            cam.SetParameterDouble("Gain", gain)
            cam.Start()

        #TODO: yaml magic
        fixed_width = 800
        frame_l = cam_l.GetFrame()
        frame_r = cam_r.GetFrame()
        frame_tex = cam_texture.GetFrame()
        (h_l, w_l) = frame_l.shape[:2]
        (h_r, w_r) = frame_r.shape[:2]
        (h_tex, w_tex) = frame_tex.shape[:2]

        ratio_l = fixed_width / float(w_l)
        dim_l = (fixed_width, int(h_l * ratio_l))
        
        ratio_r = fixed_width / float(w_r)
        dim_r = (fixed_width, int(h_r * ratio_r))

        ratio_tex = fixed_width / float(w_tex)
        dim_tex = (fixed_width, int(h_tex * ratio_tex))

        dimensions = [dim_l, dim_r, dim_tex]
        print(dimensions)
    
    except:
        dimensions = [(640,480), (640,480), (640,480)]
        print(dimensions)
    
    MAP = Mapping(folder_name_pattern, cameras, dimensions,
                  folder_name_capture, range_steps_calib, range_steps_scan, com_port,
                  OBJECT_PATH, exposure_procam, exposure_intrinsic)

    # Stop cameras
    cameras = []

    if cameras:
        for cam in cameras:
            cam.Stop()
            cam.Close()
    print("done :)")
    