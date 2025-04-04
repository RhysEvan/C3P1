## \file Example_GrayCodeScanner.py
#  \brief GrayCodeScanner class.
#  \author Seppe / Rhys
#  \copyright InviLab
#  \ingroup GrayCodeScannerGroup
#
#  This file provides an example of how to use the GrayCodeScanner class to perform various operations such as loading cameras, generating projection patterns, and scanning objects.

## \defgroup GrayCodeScannerGroup Gray Code Scanner
#  \brief This group contains all the classes and functions related to the Gray Code Scanner.
import os
# import time
import h5py
import glob
import inspect
import numpy as np

try:
    from CameraModel.GenICam.GenICamCamera import GenICamCamera
except:
    print("no luck with module")
from StructuredLight.Graycode import ProjectionPattern, Decode_Gray
import Calibration.GraycodeCalibration as cg
import Calibration.IntrinsicCalirbration as ci
import Calibration.TurnTableCalibration as ct
from DataCapture import innitiate_support
from DataCapture import graycode_data_capture
from DataCapture import intrinsic_calibration_capture
from DataCapture import turntable_calibration_capture
from Triangulation import MonoTriangulator, StereoTriangulator
import json

from inputparameters import PROJECTOR_DIRECTORY
from inputparameters import CALIBRATION_DATA_DIRECTORY, INTRINSIC_CALIBRATION_DATA_DIRECTORY, TURNTABLE_CALIBRATION_DATA_DIRECTORY
from inputparameters import WIDTH, HEIGHT
from inputparameters import IMAGE_RESOLUTION
from inputparameters import CHESS_SHAPE, CHESS_BLOCK_SIZE
from inputparameters import SQUARES_X, SQUARES_Y
from inputparameters import SQUARES_LENGTH, MARKER_LENGTH

def save_selected_frames(type_format, identifier, frame_list):
    filename = f"{type_format}selected_frames_{identifier}.txt"
    with open(filename, "w") as file:
        file.write(",".join(map(str, frame_list)))
    print(f"Frame list saved as {filename}")

class GrayCodeScanner:
    def __init__(self, folder_name_pattern, texture_camera, folder_name_capture,
                 range_steps_calib, range_steps_scan, com_port='COM4', obj_path=None, exposure=2500, exposure_intrinsic=100000):

        self.stereo_triangulate = None
        self.mono_triangulate_left = None
        self.mono_triangulate_right = None # 0705 todo: init with the proper class. This is not the right way to do it. However, the classes need to be changed to be able to do this. It needs to allow an empty init.


        self.folder_name_pattern = folder_name_pattern
        self.cameras = []
        self.texture_camera = texture_camera
        self.dimensions = []
        self.folder_name_capture = folder_name_capture
        self.range_steps_calib = range_steps_calib
        self.range_steps_scan = range_steps_scan
        self.com_port = com_port
        self.obj_path = obj_path # path to objects
        if not os.path.exists(obj_path):
            os.makedirs(obj_path)
        self.exposure = exposure
        self.exposure_intrinsic = exposure_intrinsic
        self.image_count_calibration = 40
        self.calibration_data = None

        self.decoder = []
        self.decoder = Decode_Gray()

        # note: global variables: 0705 todo: change this, just a param xml-file, json or txt file will work better.
        if not os.path.exists(PROJECTOR_DIRECTORY):
            os.makedirs(PROJECTOR_DIRECTORY)
        self.PROJECTOR_DIRECTORY = glob.glob(PROJECTOR_DIRECTORY)
        if not os.path.exists(INTRINSIC_CALIBRATION_DATA_DIRECTORY):
            os.makedirs(INTRINSIC_CALIBRATION_DATA_DIRECTORY)
        self.INTRINSIC_CALIBRATION_DATA_DIRECTORY = (INTRINSIC_CALIBRATION_DATA_DIRECTORY)
        if not os.path.exists(CALIBRATION_DATA_DIRECTORY):
            os.makedirs(CALIBRATION_DATA_DIRECTORY)
        self.CALIBRATION_DATA_DIRECTORY = CALIBRATION_DATA_DIRECTORY
        if not os.path.exists(TURNTABLE_CALIBRATION_DATA_DIRECTORY):
            os.makedirs(TURNTABLE_CALIBRATION_DATA_DIRECTORY)
        self.TURNTABLE_CALIBRATION_DATA_DIRECTORY = TURNTABLE_CALIBRATION_DATA_DIRECTORY


    def load_cams(self,cameras):
        """
        Load the cameras
        :param cameras:     cameras = [cam_l, cam_r]
        :type  cameras:     list of Genpycam objects

        :return: none
        """
        #0705 todo: fix that this works for n-cams
        fixed_width = 800

        frame_l = cameras[0].GetFrame()
        frame_r = cameras[1].GetFrame()
        (h_l, w_l) = frame_l.shape[:2]
        (h_r, w_r) = frame_r.shape[:2] #0705 todo this parameter should be saved somewhere, now it is loaded from a file.

        ratio_l = fixed_width / float(w_l)
        dim_l = (fixed_width, int(h_l * ratio_l))
        ratio_r = fixed_width / float(w_r)
        dim_r = (fixed_width, int(h_r * ratio_r))

        dimensions = [dim_l, dim_r]
        self.cameras = cameras
        self.dimensions = dimensions
        print(dimensions)
    def load_cams_from_ids(self, device_ids, exposures):
        """
        Load the cameras from a list of device IDs.
        :param device_ids: List of device IDs for the cameras.
        :type device_ids: list of str
        :param exposures: List of exposure times for the cameras.
        :type exposures: list of int
        :return: none
        """

        #0705 todo make/set  a class variable for the exposures.

        cameras = []
        for dev_id, exposure in zip(device_ids, exposures):
            try:
                camera = GenICamCamera()
                error = camera.Open(dev_id)
                if error:
                    print(f"Error opening camera {dev_id}")
                    continue
                camera.SetParameterDouble("ExposureTime", exposure)
                camera.Start()
                cameras.append(camera)
            except Exception as e:
                print(f"Failed to load camera {dev_id}: {e}")
        self.load_cams(cameras)
    def load_cams_from_file(self, file = 'camera_config.json'):
        """
        Load the cameras from a json file.
        :param file: string to file (json) with camera info
        :return: none
        """

        # Load the JSON file
        with open(file, 'r') as json_file:
            camera_data = json.load(json_file)

        cameras = []
        for cam_info in camera_data["cameras"]:
            try:
                camera = GenICamCamera()
                error = camera.Open(cam_info["dev_id"])
                if error:
                    print(f"Error opening camera {cam_info['dev_id']}")
                    continue
                camera.SetParameterDouble("ExposureTime", cam_info["exposure"]) #0705 todo: load multiple exposures and set the class variable.
                camera.Start()
                cameras.append(camera)
            except Exception as e:
                print(f"Failed to load camera {cam_info['dev_id']}: {e}")
        self.load_cams( cameras)
        pass
    def generate_projection_pattern(self):
        """
        Generate the projection pattern.
        the projector pattern is saved in the PROJECTOR_DIRECTORY (class variable)
        :return: none
        """
        self.decoder = Decode_Gray()

        ##########################################################################################
        ######################### generate projection pattern ####################################
        ##########################################################################################

        graycode_folder = glob.glob(PROJECTOR_DIRECTORY+"/*")
        # check fi folder exists
        if len(graycode_folder) == 0:
            self.graycode = ProjectionPattern(WIDTH, HEIGHT, PROJECTOR_DIRECTORY)
            self.graycode.generate_images()
        pass
    def center_cameras_to_projector(self):
        '''
        Center the cameras to the projector. Windows with a feed from all cameras is shown, press enter (with a selected window to continue)
        :return:
        '''

        # TODO see if calib is done recently or have a request notice to determine if the following steps should be done
        input("""Before the next step please verify that the projector is connected and turned on.
        Use the camera windows to align the camera's to the best of you capabilities with the black cross projected. Press Enter to continue""")
        innitiate_support(self.cameras, self.dimensions)
        pass
    def calibrate_camera_intrinsics(self):
        """
        Calibrates the camera intrinsics.
        Camera images (h5) and intrinics (h5) are saved in the INTRINSIC_CALIBRATION_DATA_DIRECTORY (class variable)
        :return: none
        """

        cameras = self.cameras
        self.set_exposure_time(cameras,self.exposure_intrinsic)


        calibrated_data = []
        calibrated_data = glob.glob(self.INTRINSIC_CALIBRATION_DATA_DIRECTORY+"/*_scan.h5")

        if len(calibrated_data) == 0:
            intrinsic_calibration_capture(self.INTRINSIC_CALIBRATION_DATA_DIRECTORY,
                                            cameras, self.texture_camera,
                                            identifier=["L","R"])
        else:
            print("Intrinsic calibration images already exist, skipping capture")
        calibrated_data = []
        calibrated_data = glob.glob(INTRINSIC_CALIBRATION_DATA_DIRECTORY+"/*parameters.h5")
        if len(calibrated_data) == 0:
            self.intrinsic_calibration()
        self.set_exposure_time(self.cameras)
        self.calibration_data = calibrated_data
    def intrinsic_calibration(self):
        """
        Execute calibration
        """
        image_count = self.image_count_calibration
        frame_list = list(range(image_count))
        self.intrinsic_calibration_L = self.quality_optimisation(ci.IntrisicCalibration,frame_list,
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

        if self.texture_camera is not None:
            self.intrinsic_calibration_RGB = self.quality_optimisation(
                ci.IntrisicCalibration,
                frame_list,
                INTRINSIC_CALIBRATION_DATA_DIRECTORY,
                CHESS_SHAPE,
                CHESS_BLOCK_SIZE,
                WIDTH,
                HEIGHT,
                IMAGE_RESOLUTION,
                identifier="RGB",
                format_type="int_",
                visualize=True
                )
    def set_exposure_time(self,cameras,exposure = None):
        """
        Set the exposure time for the cameras in the cameralist.
        :param cameras:
        :param exposure: exposure time in microseconds
        :return:
        """

        if exposure is None:
            exposure = self.exposure

        try:
            if isinstance(self.exposure, list): #0705 changed this, now setting a different exposure for each camera is possible.
                for i, camera in enumerate(cameras):
                    camera.SetParameterDouble("ExposureTime", exposure[i])
            else:
                for camera in cameras:
                    camera.SetParameterDouble("ExposureTime", exposure)
        except:
            print("setting exposure failed, this might not be supported byt the camera")
    def calibrate_camera_extrinsics(self):
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
        calibrated_data = []
        calibration_data = glob.glob(self.CALIBRATION_DATA_DIRECTORY+"/*.h5")

        if len(calibration_data) == 0:
            time_hold = input("turn on projector then press enter for extrinsic calibration")
            counter = 0
            for _ in range(20):
                counter += 1

                print(f"capturing image {counter} of 20")
                graycode_data_capture(self.folder_name_pattern, self.cameras, self.texture_camera,
                                      self.dimensions, self.CALIBRATION_DATA_DIRECTORY, 0)

        calibrated_data = []
        calibrated_data = glob.glob(self.CALIBRATION_DATA_DIRECTORY + "/*parameters.h5")
        if len(calibrated_data) == 0:
            print("Calibrating stereo")
            self.stereo_calibration()
        pass
    def stereo_calibration(self):
        h5_path_L = self.INTRINSIC_CALIBRATION_DATA_DIRECTORY+"/L_camera_intrinsic_parameters.h5"
        h5_path_R = self.INTRINSIC_CALIBRATION_DATA_DIRECTORY+"/R_camera_intrinsic_parameters.h5"
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

    def calibrate_turntable_extrinsics(self,obj_path = None, range_steps_calib = None, range_steps_scan = None, com_port = None):
        """
        Captures turntable calibration data.

        :param obj_path: Path to the object data directory. Defaults to None.
        :type obj_path: str, optional
        :param range_steps_calib: Number of steps for calibration. Defaults to None.
        :type range_steps_calib: int, optional
        :param range_steps_scan: Number of steps for scanning. Defaults to None.
        :type range_steps_scan: int, optional
        :param com_port: COM port for the turntable. Defaults to None.
        :type com_port: str, optional
        :return: None
        """
         #0705 todo: make this function return 0: success, 1: failure
        #get the parameters
        cameras = self.cameras
        exposure_intrinsic = self.exposure_intrinsic
        exposure = self.exposure #0705 todo: does this also needs to be overloaded from the function call?
        folder_name_pattern = self.folder_name_pattern
        texture_camera = self.texture_camera
        if obj_path is None:
            obj_path = self.obj_path
        if range_steps_calib is None:
            range_steps_calib = self.range_steps_calib
        if range_steps_scan is None:
            range_steps_scan = self.range_steps_scan
        if com_port is None:
            com_port = self.com_port
        # the code
        calibrated_data = []
        calibrated_data = glob.glob(self.TURNTABLE_CALIBRATION_DATA_DIRECTORY + "/scan.h5")
        input("please turn off projector (press enter to continue)")
        if len(calibrated_data) == 0:
            self.set_exposure_time(cameras)
            turntable_calibration_capture(cameras, self.TURNTABLE_CALIBRATION_DATA_DIRECTORY, range_steps_calib, com_port)

        calibrated_data = []
        calibrated_data = glob.glob(self.TURNTABLE_CALIBRATION_DATA_DIRECTORY + "/*matrix_data.h5")
        if len(calibrated_data) == 0:
            self.turntable_calibration()

        self.set_exposure_time(cameras)
    def turntable_calibration(self): #0705 todo: add folder names as input parameters
        """
        Perform turntable calibration.
        :return: none
        """
        h5_path_L = self.INTRINSIC_CALIBRATION_DATA_DIRECTORY+"/L_camera_intrinsic_parameters.h5"
        h5_path_R = self.INTRINSIC_CALIBRATION_DATA_DIRECTORY+"/R_camera_intrinsic_parameters.h5"

        cam_int_L, cam_dist_L = self.load_parameters(h5_path_L)
        cam_int_R, cam_dist_R = self.load_parameters(h5_path_R)

        turntable_L = ct.TurnTableCalibration(self.TURNTABLE_CALIBRATION_DATA_DIRECTORY, cam_int_L,
                                           cam_dist_L, SQUARES_X, SQUARES_Y, SQUARES_LENGTH,
                                           MARKER_LENGTH, "L")
        turntable_L.calibrate()

        turntable_R = ct.TurnTableCalibration(self.TURNTABLE_CALIBRATION_DATA_DIRECTORY, cam_int_R,
                                           cam_dist_R, SQUARES_X, SQUARES_Y, SQUARES_LENGTH,
                                           MARKER_LENGTH, "R")
        turntable_R.calibrate()
    def scan_object(self,obj_path=None, range_steps_scan=1):
        """
        Capture object data or load data if already captured.
        :param obj_path: path where the capture data is stored. if it is empty, a new capture will be done, if it is contains data, the data will be loaded. if none is given, the default path will be used.
        :param range_steps_scan: if a turntable is pressent this is the number of steps
        :return: pointclouds (stereo, mono_left, mono_right)
        """
        self.capture_or_load_object_data(obj_path=obj_path,range_steps_scan=range_steps_scan)
        self.calculate_pointcloud(obj_path=obj_path)

        return self.stereo_pointclouds, self.mono_left_pointclouds, self.mono_right_pointclouds
    def capture_or_load_object_data(self, obj_path=None, range_steps_scan=1):
        """
        Capture object data or load data if already captured.
        :param obj_path: path where the capture data is stored. if it is empty, a new capture will be done, if it is contains data, the data will be loaded. if none is given, the default path will be used.
        :param range_steps_scan: if a turntable is pressent this is the number of steps
        :return: none
        """
        if obj_path is None:
            obj_path = self.obj_path
        if not os.path.exists(obj_path): #make the path if is does not exist
            os.makedirs(obj_path)
        self.obj_path = obj_path # update the path
        capture_data = []
        capture_data = glob.glob(obj_path + "*.h5") # check if the data is already captured
        input("please turn on projector")
        if len(capture_data) == 0: # if the data is not captured, capture it
            graycode_data_capture(self.folder_name_pattern, self.cameras, self.texture_camera, self.dimensions, obj_path, range_steps_scan)
        else:
            print("object data already exists, skipping capture, calculating pointclouds")
    def calculate_pointcloud(self,obj_path=None):
        """
        Calculate pointclouds from the captured object data.
        :param obj_path:
        :return: none
        """
        if obj_path is None:
            obj_path = self.obj_path
        scenes = self.load_h5(obj_path)
        if scenes is None:
            print("no data found in " ,obj_path)
            return

        self.stereo_triangulate = StereoTriangulator(
            self.CALIBRATION_DATA_DIRECTORY)  # loading this every time from files is not efficient, should be a class variable
        # 0705, this is the first time that we use image_resoltion and width and height, from global variables. This should be a class variable, and info needs to come from the camaera. Note
        # note, is is always possible that your are using different resulutions for the cameras, this is not supported by the current code.
        # image resolution should be a variable in the calib-data. Changing the resolution affects the calibration.
        self.mono_triangulate_left = MonoTriangulator(IMAGE_RESOLUTION, self.CALIBRATION_DATA_DIRECTORY, WIDTH, HEIGHT,
                                                      'L')
        self.mono_triangulate_right = MonoTriangulator(IMAGE_RESOLUTION, self.CALIBRATION_DATA_DIRECTORY, WIDTH, HEIGHT,
                                                       'R')


        # Lists to collect point clouds for all scenes
        self.stereo_pointclouds = []
        self.mono_left_pointclouds = []
        self.mono_right_pointclouds = []

        for i, item in enumerate(scenes):
            print(i)
            print("decoding scene")
            left_horizontal_decoded_image, left_vertical_decoded_image = self.decoder.scene_decoder(item, "L")
            right_horizontal_decoded_image, right_vertical_decoded_image = self.decoder.scene_decoder(item, "R")
            print("triangulating scene")
            stereo_pointcloud = self.stereo_triangulate.triangulate(
                left_horizontal_decoded_image, left_vertical_decoded_image,
                right_horizontal_decoded_image, right_vertical_decoded_image
            )
            camera_points, projector_points, _ = self.mono_triangulate_left.get_cam_proj_pts(None,
                                                                                             left_horizontal_decoded_image,
                                                                                             left_vertical_decoded_image)

            mono_left_pointcloud = self.mono_triangulate_left.triangulate_mono(camera_points,
                                                                                    projector_points)

            camera_points, projector_points, _ = self.mono_triangulate_right.get_cam_proj_pts(None,
                                                                                              right_horizontal_decoded_image,
                                                                                              right_vertical_decoded_image)

            mono_right_pointcloud = self.mono_triangulate_right.triangulate_mono(camera_points,
                                                                                      projector_points)

            # Append the point clouds to the respective lists
            self.stereo_pointclouds.append(stereo_pointcloud)
            self.mono_left_pointclouds.append(mono_left_pointcloud)
            self.mono_right_pointclouds.append(mono_right_pointcloud)
    def load_h5(self, path):
        """
        Load the h5 files from the given path.
        :param path: The path to the h5 files.
        :return: A list of scenes.
        """
        scenes = []
        h5_file_paths = glob.glob(path+'*')
        for _,path in enumerate(h5_file_paths):
            if ".h5" in path:
                scenes.append(self.load_object(path))
            else:
                continue
        return scenes
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

    def load_object(self, scene_path):
        if not os.path.exists(scene_path):
            raise FileNotFoundError(f"The file {scene_path} does not exist.")

        scene_data = {}
        with h5py.File(scene_path, 'r') as h5f:
            for key in h5f.keys():
                scene_data[key] = np.array(h5f[key])
        return scene_data
    def Visualize(self):
        pass
        #visualisepointcloud(self.stereo_pointcloud)
        #visualisepointcloud(self.mono_left_pointcloud)
        #visualisepointcloud(self.mono_right_pointcloud)
    def save_pointclouds(self, obj_path="output"):
        # Save all point clouds to separate npy files
        if not os.path.exists(obj_path):
            os.makedirs(obj_path)
        print("saving pointclouds")
        np.savez(os.path.join(obj_path, 'stereo_pointclouds.npz'), *self.stereo_pointclouds)
        np.savez(os.path.join(obj_path, 'mono_left_pointclouds.npz'), *self.mono_left_pointclouds)
        np.savez(os.path.join(obj_path, 'mono_right_pointclouds.npz'), *self.mono_right_pointclouds)
        pass
    def save_selected_frames(self, type_format, identifier, frame_list):
        pass

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
        valid_kwargs["visualize"] = False
        while True:
            # Call the provided class or function with the current frame list and other parameters
            calibration = class_object(frame_list, *args, **valid_kwargs)
            if calibration.projector_calibrator is not None:
                calibration.camera_calibrator.plot_reproj_error()
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

    def close_cams(self):
        """
        Close the cameras.
        :return: none
        """
        print("Closing cameras")
        for camera in self.cameras:
            try:
                camera.Stop()
                camera.Close()
            except Exception as e:
                print(f"Failed to close camera: {e}")
    def __del__(self):
        """
        Destructor to stop and close the cameras.
        """
        self.close_cams()
