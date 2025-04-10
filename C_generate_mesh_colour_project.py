import numpy as np
import open3d as o3d
import cv2

# === Parameters ===
image_path = "calibrated_image.png"  # RGB image path
merged_pcd_path = "merged_final.ply"  # Final point cloud from previous script

# Replace these with your actual calibration matrices
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])  # Intrinsic matrix
R = np.eye(3)  # Rotation matrix
t = np.zeros((3, 1))  # Translation vector

# === Load data ===
print("Loading merged point cloud...")
pcd = o3d.io.read_point_cloud(merged_pcd_path)
o3d.visualization.draw_geometries([pcd], window_name="Original Point Cloud")

# === Estimate normals (required for Poisson) ===
print("Estimating normals...")
pcd.estimate_normals()
pcd.orient_normals_consistent_tangent_plane(30)
o3d.visualization.draw_geometries([pcd], window_name="Point Cloud with Normals")

# === Create mesh from point cloud ===
print("Performing Poisson surface reconstruction...")
mesh_raw, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

# Visualize raw mesh
print("Visualizing raw mesh (uncropped)...")
mesh_raw.paint_uniform_color([0.7, 0.7, 0.7])
o3d.visualization.draw_geometries([mesh_raw], window_name="Raw Mesh")

# === Crop mesh to bounding box of original cloud ===
print("Cropping mesh to bounding box...")
bbox = pcd.get_axis_aligned_bounding_box()
mesh_cropped = mesh_raw.crop(bbox)

# Visualize cropped mesh
print("Visualizing cropped mesh...")
mesh_cropped.paint_uniform_color([0.8, 0.6, 0.6])
o3d.visualization.draw_geometries([mesh_cropped], window_name="Cropped Mesh")

# === Project mesh vertices to image plane and apply color ===
print("Applying color to mesh vertices...")
P = K @ np.hstack((R, t))  # 3x4 projection matrix
image = cv2.imread(image_path)
image_height, image_width = image.shape[:2]

colors = []
vertices = np.asarray(mesh_cropped.vertices)

for v in vertices:
    v_cam = R @ v.reshape(3, 1) + t  # Transform to camera frame
    v_proj = K @ v_cam
    x, y = (v_proj[0] / v_proj[2])[0], (v_proj[1] / v_proj[2])[0]
    x_pix, y_pix = int(round(x)), int(round(y))

    if 0 <= x_pix < image_width and 0 <= y_pix < image_height:
        color = image[y_pix, x_pix] / 255.0  # Normalize to [0,1]
    else:
        color = np.array([0.0, 0.0, 0.0])  # default color
    colors.append(color)

mesh_cropped.vertex_colors = o3d.utility.Vector3dVector(np.array(colors))

# === Save and visualize ===
print("Saving final colored mesh...")
o3d.io.write_triangle_mesh("colored_mesh.ply", mesh_cropped)

print("Visualizing final colored mesh...")
o3d.visualization.draw_geometries([mesh_cropped], window_name="Final Colored Mesh")
