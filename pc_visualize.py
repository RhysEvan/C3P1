import numpy as np
import open3d as o3d

file_path = r"C:\Users\InViLab\Desktop\3-View_Application\CPC_CamProCam_UAntwerp\mono_right_pointclouds.npz"
npz_data = np.load(file_path, allow_pickle=True)
point_clouds = [o3d.geometry.PointCloud(o3d.utility.Vector3dVector(npz_data[key])) for key in npz_data]
point_clouds = [point_clouds[0], point_clouds[1]]
o3d.visualization.draw_geometries(point_clouds)