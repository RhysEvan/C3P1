# import wx
# import os
# import glob
# from PIL import Image
import numpy as np
# import time
# import cv2
import json
from typing import Dict, Any, Optional, Union, List, Tuple

from CameraModel import GenICamCamera
from CameraModel.OpenCV.OpenCVCamera import OpenCVCamera

class MultiCam:

    def __init__(self, width=640, height=480,  camera_dictionary: Optional[Dict[str, Dict[str, Any]]] = None):

        self.width = width
        self.height = height
        self.frame_count = 0
        self.captured_frames = []

        if camera_dictionary is not None:
            self.cameras = camera_dictionary
        else:
            self.cameras = {}
        
        print("Initialized MultiCam")

    def load_cameras_from_file(self, filepath):
        """Loads camera configuration from a JSON file and stores it in self.cameras."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        processed_data = {}
        for key, camera in data.items():
            camera_class = camera.get('camera_class')
            if camera_class not in ['pleora', 'opencv']:
                print(f"Warning: Unknown camera_class '{camera_class}' for camera {key}")
                continue
            elif camera_class in "pleora":
                camera['camera_class'] = GenICamCamera("pleora")
            elif camera_class in "opencv":
                camera['camera_class'] = OpenCVCamera()

            processed_data[key] = camera

        self.cameras = processed_data

    def OpenCameras(self) -> None:
        for val in self.cameras.values():
            val['camera_class'].Open(val['camera_adress'])

    def GrabAllFrames(self, identities: List[str] = None) -> List[Tuple[str, np.array]]:
        assert identities, (
            f"Execution of code ceased due to identities being: {identities}\n"
            "In the case of it being None, make sure to insert a list of the desired camera IDs.\n"
            "In the case of it being [], make sure to write the necessary IDs in the list."
        )
        self.captured_frames = []
        for camera in self.cameras.values():
            if camera['identifier'] in identities:
                frame = camera['camera_class'].GetFrame()
                self.captured_frames.append((camera['identifier'], frame))

        return self.captured_frames

    def EditIntrinsicExposure(self, parameters: Union[int, List[int]] = 75000):
        if len(parameters) == 1:
            for val in self.cameras.values():
                val['intrinsic_exposure'] = parameters
        
        elif len(parameters) == len(self.cameras.values()):
            for i, val in enumerate(self.cameras.values()):
                val['intrinsic_exposure'] = parameters[i]

    def SetIntrinsicExposure(self) -> None:
        for val in self.cameras.values():
            self.setExposure(val['camera_class'], val['intrinsic_exposure'])

    def EditExtrinsicExposure(self, parameters: Union[int, List[int]] = 75000):
        if len(parameters) == 1:
            for val in self.cameras.values():
                val['extrinsic_exposure'] = parameters
        
        elif len(parameters) == len(self.cameras.values()):
            for i, val in enumerate(self.cameras.values()):
                val['extrinsic_exposure'] = parameters[i]

    def SetExtrinsicExposure(self) -> None:
        for i, val in enumerate(self.cameras.values()):
            self.setExposure(val['camera_class'], val['extrinsic_exposure'])

    def SetIntrinsicGain(self, parameters: Union[int, List[int]] = 75000,
                         apply: bool = False) -> None:
        if len(parameters) == 1:
            for val in self.cameras.values():
                val['intrinsic_gain'] = parameters
                if apply:
                    self.setExposure(val['camera_class'], parameters)
        elif len(parameters) == len(self.cameras.values()):
            for i, val in enumerate(self.cameras.values()):
                val['intrinsic_gain'] = parameters[i]
                if apply:
                    self.setGain(val['camera_class'], parameters[i])

    def SetExtrinsicGain(self, parameters: Union[int, List[int]] = 75000,
                         apply: bool = False) -> None:
        if len(parameters) == 1:
            for val in self.cameras.values():
                val['extrinsic_gain'] = parameters
                if apply:
                    self.setExposure(val['camera_class'], parameters)
        elif len(parameters) == len(self.cameras.values()):
            for i, val in enumerate(self.cameras.values()):
                val['extrinsic_gain'] = parameters[i]
                if apply:
                    self.setGain(val['camera_class'], parameters[i])

    def setExposure(self, camera, value) -> None:
        camera.SetParameterDouble("ExposureTime", value)

    def setGain(self, camera, value) -> None:
        camera.SetParameterDouble("Gain", value)

    def Release(self):
        print("MultiCam: Releasing resources.")
        pass

    #write a destructor
    def Close(self):
        for cam in self.cameras:
            cam.Close()
        self.cameras = []
        print('MultiCam: Closed all cameras.')
    def __del__(self):
        print("MultiCam: Destructor called.")
        for cam in self.cameras:
            cam.Close()


if __name__ == "__main__":
    # Replace this with the actual path to your JSON file
    json_file_path = r"C:\Users\mheva\OneDrive\Bureaublad\GitHub\C3P1\Examples\example_config.json"

    # Initialize MultiCam
    multicam = MultiCam()

    # Load the camera configurations
    multicam.load_cameras_from_file(json_file_path)

    # Print the resulting camera dictionary
    print("Loaded Camera Configuration:")
    for cam_id, cam_info in multicam.cameras.items():
        print(f"{cam_id}: {cam_info}")