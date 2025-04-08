import wx
import os
import glob
from PIL import Image
import numpy as np
import time
import cv2

class MultiCam:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.frame_count = 0
        self.cameras = []
        print("Initialized MultiCam")
    def load_cams_from_file(self, file = 'camera_config.json'):
        """
        Load the cameras from a json file.
        :param file: string to file (json) with camera info
        :return: none
        """
        import json
        from CameraModel import GenICamCamera

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
        self.cameras = cameras
        self.load_cams( cameras)
        pass
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


    def GetFrame(self):
        print("MultiCam: Capturing frame...")
        idx = 0
        pil_images = []

        for cam in self.cameras:
            frame = cam.GetFrame()
            # save the frame with indx_frame_count as grayscale image
            # convert to RGB
            # convert to PIL image
            pil_image = Image.fromarray(frame)
            pil_image = pil_image.convert("RGB")
            # save the image
            #pil_image.save(f"frame_{idx}_{self.frame_count}.png")
            pil_images.append(pil_image)

            idx += 1


        self.frame_count += 1
        return pil_images

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
