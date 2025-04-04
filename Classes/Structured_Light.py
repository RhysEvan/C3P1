from Stereo import Stereo
import os
import numpy as np
import h5py
import glob

class Structured_Light(Stereo):
    """
    \ingroup structured_light
    \brief A class to represent a structured light system.

    This class is a child of the Stereo class and includes an additional attribute for the projector. It also includes methods for triangulation and capturing patterns.
    """

    def __init__(self):
        """
        \ingroup structured_light
        \brief Initializes the StructuredLight class with an additional projector attribute.
        """
        super().__init__()
        self.projector = None
        self.turntable = None

    def filter_and_plot_pointcloud(self,ptcloud,distances,treshold = 0.3):
        # set the values to nan where d is larger than 10
        ptcloud[distances > treshold] = [0,0,0]
        #ptcloud[indexl == 0] = [0,0,0]
        #ptcloud[indexr== 0] = [0,0,0]


        # remove 0 values from pointcloud
        ptcloud = ptcloud[~np.all(ptcloud == 0, axis=1)]

        #ptcloud[dn > 4] = [0,0,0]
        #plot the pointcloud with open3d
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(ptcloud)
        o3d.visualization.draw_geometries([pcd])
    def save_as_ply_file(self,path, ptcloud,distances,treshold = 0.3):

        import open3d as o3d
        ptcloud[distances > treshold] = [0,0,0]
        #ptcloud[indexl == 0] = [0,0,0]
        #ptcloud[indexr== 0] = [0,0,0]


        # remove 0 values from pointcloud
        ptcloud = ptcloud[~np.all(ptcloud == 0, axis=1)]

        # Assuming ptcloud is your numpy array of points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(ptcloud)

        # Save the point cloud as a PLY file
        o3d.io.write_point_cloud(path, pcd)

        # Optionally, visualize the point cloud
        o3d.visualization.draw_geometries([pcd])

    def triangulate(self):
        """
        \ingroup structured_light
        \brief Triangulates points using the structured light system.

        \return A list of 3D points obtained from triangulation.
        """
        # Implement triangulation logic here
        pass

    def plot_and_capture_pattern(self):
        """
        \ingroup structured_light
        \brief Plots a pattern using the projector and captures the resulting images.

        \return Captured images after projecting the pattern.
        """
        # Implement pattern plotting and capturing logic here
        pass

    def generate_projection_pattern(self):
        pass

    def _load_object(self, scene_path):
        if not os.path.exists(scene_path):
            raise FileNotFoundError(f"The file {scene_path} does not exist.")

        scene_data = {}
        with h5py.File(scene_path, 'r') as h5f:
            for key in h5f.keys():
                scene_data[key] = np.array(h5f[key])
        return scene_data
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
                scenes.append(self._load_object(path))
            else:
                continue
        return scenes



