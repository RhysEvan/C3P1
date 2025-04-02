import os
import glob
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import h5py

class ImageLoader:
    def __init__(self, object_path):
        self.object_path = object_path

    def load_images(self, image_paths):
        with ThreadPoolExecutor() as executor:
            images = list(executor.map(self.load_image, image_paths))
        return images

    @staticmethod
    def load_image(image_path):
        return np.array(Image.open(image_path))

    def load_object(self):
        scene_folders = sorted(glob.glob(os.path.join(self.object_path, 'scan_*')))
        for scene_folder in scene_folders:
            pattern_h_l = []
            pattern_h_l_inv = []
            pattern_h_r = []
            pattern_h_r_inv = []
            pattern_v_l = []
            pattern_v_l_inv = []
            pattern_v_r = []
            pattern_v_r_inv = []

            scene_name = os.path.basename(scene_folder)
            h5_file_path = os.path.join(self.object_path, f"{scene_name}.h5")

            with h5py.File(h5_file_path, 'w') as h5f:
                left_images_horizontal = glob.glob(scene_folder + '/L_pattern_H_*.tiff')
                left_images_vertical = glob.glob(scene_folder + '/L_pattern_V_*.tiff')
                right_images_horizontal = glob.glob(scene_folder + '/R_pattern_H_*.tiff')
                right_images_vertical = glob.glob(scene_folder + '/R_pattern_V_*.tiff')

                pattern_white_l = glob.glob(scene_folder + '/L_pattern_white.tiff')
                pattern_black_l = glob.glob(scene_folder + '/L_pattern_black.tiff')
                pattern_white_r = glob.glob(scene_folder + '/R_pattern_white.tiff')
                pattern_black_r = glob.glob(scene_folder + '/R_pattern_black.tiff')

                for p in left_images_horizontal:
                    if 'H_I_' in os.path.basename(p):
                        pattern_h_l_inv.append(p)
                    else:
                        pattern_h_l.append(p)
                for p in left_images_vertical:
                    if 'V_I_' in os.path.basename(p):
                        pattern_v_l_inv.append(p)
                    else:
                        pattern_v_l.append(p)
                for p in right_images_horizontal:
                    if 'H_I_' in os.path.basename(p):
                        pattern_h_r_inv.append(p)
                    else:
                        pattern_h_r.append(p)
                for p in right_images_vertical:
                    if 'V_I_' in os.path.basename(p):
                        pattern_v_r_inv.append(p)
                    else:
                        pattern_v_r.append(p)

                h5f.create_dataset("L_pattern_H", data=self.load_images(pattern_h_l))
                h5f.create_dataset("R_pattern_H", data=self.load_images(pattern_h_r))
                h5f.create_dataset("L_pattern_H_I", data=self.load_images(pattern_h_l_inv))
                h5f.create_dataset("R_pattern_H_I", data=self.load_images(pattern_h_r_inv))

                h5f.create_dataset("L_pattern_V", data=self.load_images(pattern_v_l))
                h5f.create_dataset("R_pattern_V", data=self.load_images(pattern_v_r))
                h5f.create_dataset("L_pattern_V_I", data=self.load_images(pattern_v_l_inv))
                h5f.create_dataset("R_pattern_V_I", data=self.load_images(pattern_v_r_inv))

                h5f.create_dataset("L_pattern_white", data=self.load_images(pattern_white_l))
                h5f.create_dataset("L_pattern_black", data=self.load_images(pattern_black_l))

                h5f.create_dataset("R_pattern_white", data=self.load_images(pattern_white_r))
                h5f.create_dataset("R_pattern_black", data=self.load_images(pattern_black_r))

# Example usage
object_path = r'C:\Users\Rhys\Documents\GitHub\CPC_CamProCam_UAntwerp\temp\data'
loader = ImageLoader(object_path)
loader.load_object()

# The images are now saved in HDF5 files named scene_name.h5 in the object_path directory.
