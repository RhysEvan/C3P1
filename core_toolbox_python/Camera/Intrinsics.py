import numpy as np
import copy
import os

from core_toolbox_python.Plucker.Line import *

class RadialDistortion:
    def __init__(self, k1=0, k2=0, k3=0):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3

    def set_from_list(self, coeffs):
        if len(coeffs) != 3:
            raise ValueError("List must contain exactly three elements.")
        self.k1, self.k2, self.k3 = coeffs

class IntrinsicMatrix:
    def __init__(self):
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.s = 0
        self.width = None
        self.height = None
        self.RadialDistortion = RadialDistortion()
        self._MatlabIntrinsics = np.zeros((3, 3))
        self._OpenCVIntrinsics = np.zeros((3, 3))
        self.pixel_size = None  # Pixel size in mm e.g. 0,0034mm (3.4 um)
        self.info = None # Additional information (camera id, lens id, ...)


    @property
    def MatlabIntrinsics(self):
        I = np.zeros((3, 3))
        I[0, 0] = self.fx
        I[1, 1] = self.fy
        I[2, 0] = self.cx
        I[2, 1] = self.cy
        I[2, 2] = 1
        return I

    @MatlabIntrinsics.setter
    def MatlabIntrinsics(self, I):
        self.fx = I[0, 0]
        self.fy = I[1, 1]
        self.cx = I[2, 0]
        self.cy = I[2, 1]
        self.s = I[1, 0]

    @property
    def OpenCVIntrinsics(self):
        I = np.zeros((3, 3))
        I[0, 0] = self.fx
        I[1, 1] = self.fy
        I[0, 2] = self.cx
        I[1, 2] = self.cy
        I[2, 2] = 1
        return I

    @OpenCVIntrinsics.setter
    def OpenCVIntrinsics(self, I):
        self.fx = I[0, 0]
        self.fy = I[1, 1]
        self.cx = I[0, 2]
        self.cy = I[1, 2]
        self.s = 0

    @property
    def focal_length_mm(self):
        """Get the focal length in millimeters."""
        if self.pixel_size is None:
            raise ValueError('Pixel size not set!')
        return self.fx * self.pixel_size, self.fy * self.pixel_size

    @property
    def PerspectiveAngle(self):
        if self.width is None:
            raise ValueError('set width first!')

        aspectRatio = self.width / self.height
        if aspectRatio > 1:
            return 2 * np.arctan(self.width / (2 * self.fx))*180/np.pi
        else:
            return 2 * np.arctan(self.height / (2 * self.fy))*180/np.pi

    @PerspectiveAngle.setter
    def PerspectiveAngle(self, p):
        if self.width is None:
            raise ValueError('set width first!')

        aspectRatio = self.width / self.height
        if aspectRatio > 1:
            self.fx = (self.width / 2) / np.tan(p / 2)
            self.fy = self.fx
            self.cx = self.width / 2
            self.cy = self.height / 2
        else:
            self.fy = (self.height / 2) / np.tan(p / 2)
            self.fx = self.fy
            self.cx = self.width / 2
            self.cy = self.height / 2

    def CameraParams2Intrinsics(self, CameraParams):
        try:
            self.width = CameraParams.ImageSize[1]
            self.height = CameraParams.ImageSize[0]
        except AttributeError:
            print('cam not open, set resolution manually!!')

        self.MatlabIntrinsics = CameraParams.IntrinsicMatrix

    def Intrinsics2CameraParams(self):
        P = {
            'IntrinsicMatrix': self.MatlabIntrinsics,
            'ImageSize': [self.height, self.width]
        }
        if self.RadialDistortion is not None:
            P['RadialDistortion'] = self.RadialDistortion
        return P

    def ScaleIntrinsics(self, Scale):
        self.fx *= Scale
        self.fy *= Scale
        self.cx *= Scale
        self.cy *= Scale
        self.width *= Scale
        self.height *= Scale

    def generate_rays(self):

        """
        Generate rays for every pixel in the image based on the intrinsic matrix.

        Parameters:
        I : IntrinsicMatrix
            Camera intrinsic parameters.

        Returns:
        rays : list of Line objects
            A list of rays originating from sensor points.
        """

        schaal = 2.0  # Just for visualization purposes 0705 todo: change name, make it a parameter
        x_vals, y_vals = np.meshgrid(np.arange(self.width), np.arange(self.height))

        if self.RadialDistortion.k1 == 0:
            x_vals = (x_vals - self.cx) / self.fx * schaal
            y_vals = (y_vals - self.cy) / self.fy * schaal
            z_vals = np.ones_like(x_vals) * schaal
        else:
            x_norm = (x_vals - self.cx) / self.fx
            y_norm = (y_vals - self.cy) / self.fy
            r2 = x_norm ** 2 + y_norm ** 2
            radial_factor = 1 + I.RadialDistortion.k1 * r2 + I.RadialDistortion.k2 * r2 ** 2 + I.RadialDistortion.k3 * r2 ** 3
            x_vals = x_norm * radial_factor * schaal
            y_vals = y_norm * radial_factor * schaal
            z_vals = np.ones_like(x_vals) * schaal


        Ps = np.stack([x_vals, y_vals, z_vals], axis=-1)
        Pf = np.zeros_like(Ps)
        Pf[..., 2] = 0  # Set Z component to zero

        directions = Ps - Pf
        rays = Line()
        rays.Ps = Pf.reshape(Pf.shape[0] * Pf.shape[1], 3)
        rays.V = directions.reshape(directions.shape[0] * directions.shape[1], 3)


        return rays

    def save_intrinsics_to_json(self, filename):
        import json
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)

        intrinsic_matrix = self
        data = {
            'OpenCVIntrinsics': intrinsic_matrix.OpenCVIntrinsics.tolist(),
            'RadialDistortion': {
                'k1': intrinsic_matrix.RadialDistortion.k1,
                'k2': intrinsic_matrix.RadialDistortion.k2,
                'k3': intrinsic_matrix.RadialDistortion.k3
            },
            'width': intrinsic_matrix.width,
            'height': intrinsic_matrix.height,
            'pixel_size': intrinsic_matrix.pixel_size,
            'info': intrinsic_matrix.info

        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)

    def load_intrinsics_from_json(self,filename):
        import json
        with open(filename, 'r') as f:
            data = json.load(f)

        intrinsic_matrix = self
        self.OpenCVIntrinsics = np.array(data['OpenCVIntrinsics'])
        intrinsic_matrix.RadialDistortion.set_from_list([
            data['RadialDistortion']['k1'],
            data['RadialDistortion']['k2'],
            data['RadialDistortion']['k3']
        ])
        intrinsic_matrix.width = data['width']
        intrinsic_matrix.height = data['height']
        intrinsic_matrix.pixel_size = data['pixel_size']
        intrinsic_matrix.info = data['info']


        return intrinsic_matrix


if __name__ == "__main__":
    I = IntrinsicMatrix()
    I.info = "testCamera"
    I.fx = 1770
    I.fy = 1770

    I.width = 1440
    I.height = 1080
    I.cx = 685
    I.cy = 492
    I.RadialDistortion.set_from_list([-0.5,0.18,0])

    I.save_intrinsics_to_json('test.json')
    rays = I.generate_rays()
    I2 = IntrinsicMatrix()
    I2.load_intrinsics_from_json('test.json')
    #rays.PlotLine()

    from core_toolbox_python.Transformation.TransformationMatrix import TransformationMatrix
    H = TransformationMatrix()
    H.T = [0,0,0]
    H.angles_degree = [45,0,0]
    rayst= copy.deepcopy(rays)
    rayst.TransformLines(H)
    #rayst.PlotLine()
    import time
    start = time.time()
    p,d = intersection_between_2_lines(rays,rayst)
    print("elapsed time: ", time.time()-start)

    ## standard deviation and mean of d
    print("mean: ", np.mean(d))
    print("std: ", np.std(d))



    print(I.PerspectiveAngle)
    I.PerspectiveAngle = np.deg2rad(60)
    print(I.MatlabIntrinsics)
    print(I.OpenCVIntrinsics)
    print(I.PerspectiveAngle)
