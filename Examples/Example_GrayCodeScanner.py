## \file Example_GrayCodeScanner.py
#  \brief Example script for using the GrayCodeScanner class.
#  \author Seppe / Rhys
#  \copyright InviLab
#  \ingroup GrayCodeScannerGroup
#
#  This file provides an example of how to use the GrayCodeScanner class to perform various operations such as loading cameras, generating projection patterns, and scanning objects.

## \defgroup GrayCodeScannerGroup Gray Code Scanner
#  \brief This group contains all the classes and functions related to the Gray Code Scanner.

from CameraModel import GenICamCamera
import os
#import time
import glob
import h5py
import numpy as np
import inspect
import matplotlib.pyplot as plt

from main import *



from Classes.GrayCodeScanner import *

if __name__ == "__main__":


    #todo: check if this is needed. the paths should be made by the class.
    folder_name_pattern = r"./static/0_projection_pattern/"
    #check if the folder exists
    if not os.path.exists(folder_name_pattern):
        os.makedirs(folder_name_pattern)
    folder_name_capture = r"./static/1_calibration_data2/"
    #check if the folder exists
    if not os.path.exists(folder_name_capture):
        os.makedirs(folder_name_capture)
    OBJECT_PATH = r'.\static\2_object_data\Seppe/'
    #check if the folder exists
    if not os.path.exists(OBJECT_PATH):
        os.makedirs(OBJECT_PATH)




    range_steps_calib = 1  # 1 if no turn table
    range_steps_scan = 1  # 1 if no turn table
    com_port = 'COM3'

    #gain = 15000
    exposure_intrinsic = 15000/2
    exposure_procam = 15000/2
    gain = 3





    #create GrayCodeScanner Object
    GrayCodeScanner = GrayCodeScanner(folder_name_pattern, None ,
                                      folder_name_capture, range_steps_calib, range_steps_scan, com_port,
                                      OBJECT_PATH, exposure_procam, exposure_intrinsic)

    #GrayCodeScanner.load_cams([cam_l, cam_r])
    GrayCodeScanner.load_cams_from_file('camera_config.json')
    #GrayCodeScanner.load_cams_from_ids(['1AB22800147A', '1AB22800147B'],[exposure_intrinsic, exposure_intrinsic])
    GrayCodeScanner.generate_projection_pattern()
    #GrayCodeScanner.center_cameras_to_projector()
    #GrayCodeScanner.calibrate_camera_intrinsics()
    GrayCodeScanner.calibrate_camera_extrinsics()
    #GrayCodeScanner.calibrate_turntable_extrinsics()
    GrayCodeScanner.set_exposure_time(exposure_procam)
    GrayCodeScanner.scan_object(obj_path=r"C:\Users\Seppe\PycharmProjects\Projector\CPC_CamProCam_UAntwerp\Examples\static\2_object_data2\Seppe/") #this does: capture_or_load_object_data(), calculate_pointcloud()

    import open3d as o3d
    import numpy as np
    # Assuming mono_left_pointclouds is a list of 3D points
    import numpy as np

    # Convert the list to a numpy array
    points = np.array(GrayCodeScanner.mono_left_pointclouds)
    points = np.array(GrayCodeScanner.stereo_pointclouds)
    points = points.reshape(-1, 3)

    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])
    # Stop cameras
    try:
        GrayCodeScanner.close_cams()
        print("done :)")
    except:
        print("done :)")