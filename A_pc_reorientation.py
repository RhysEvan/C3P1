import numpy as np
import open3d as o3d
import h5py
from scipy.linalg import svd

def load_calibration_data(filename):
    """Load and orthogonalize rotation and translation calibration data from an HDF5 file."""
    with h5py.File(filename, 'r') as h5file:
        avg_rotation_matrix = h5file['rotation_matrix'][:]
        avg_translation_matrix = h5file['translation_matrix'][:]

    # Orthogonalize the average rotation matrix to remove any scaling artifacts
    U, _, Vt = svd(avg_rotation_matrix)
    avg_rotation_orthogonal = U @ Vt

    return avg_rotation_orthogonal, avg_translation_matrix

def visualize_center_and_normal(center, normal_vector):
    """Visualize the center point and normal vector as a line in the point cloud."""

    def compute_perpendicular_vectors(normal_vector):
        """Compute two perpendicular vectors in the plane orthogonal to the normal vector."""
        # Choose an arbitrary vector to cross with (if the normal vector is not parallel to this, it's a valid choice)
        if np.abs(normal_vector[0]) > np.abs(normal_vector[1]):
            arbitrary_vector = np.array([0, 1, 0])  # Choose Y-axis as an arbitrary vector
        else:
            arbitrary_vector = np.array([1, 0, 0])  # Choose X-axis as an arbitrary vector

        # First perpendicular vector (v1)
        v1 = np.cross(normal_vector, arbitrary_vector)
        v1 = v1 / np.linalg.norm(v1)  # Normalize it

        # Second perpendicular vector (v2)
        v2 = np.cross(normal_vector, v1)
        v2 = v2 / np.linalg.norm(v2)  # Normalize it

        return v1, v2
    center = np.asarray(center).flatten()
    normal_vector = np.asarray(normal_vector).flatten()

    # Create a point cloud for the center point
    center_point_cloud = o3d.geometry.PointCloud()
    center_point_cloud.points = o3d.utility.Vector3dVector([center])

    # Define the points for the normal, X, and Y axes (extended along each axis)
    lines = [
        [0, 1],  # Normal vector (Red)
        [0, 2],  # X-axis (Black)
        [0, 3]   # Y-axis (Blue)
    ]

    v1, v2 = compute_perpendicular_vectors(normal_vector)

    # Create the points for each line
    points = np.array([
        center,                          # center point
        center + normal_vector * -100,    # normal vector (red)
        center + v1 * 100,  # X-axis (black)
        center + v2 * 100   # Y-axis (blue)
    ])

    # Create the LineSet object
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    # Color the lines: red for normal vector, black for X-axis, blue for Y-axis
    line_set.paint_uniform_color([1, 0, 0])  # Default color for all lines (red)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    # Paint specific lines for each axis
    # Normal vector (Red)
    line_set.paint_uniform_color([1, 0, 0])  # Normal vector (Red)

    # Separate colors for the X and Y axis
    line_set.colors = o3d.utility.Vector3dVector([
        [1, 0, 0],  # Red for normal vector
        [0, 0, 0],  # Black for X-axis
        [0, 0, 1]   # Blue for Y-axis
    ])

    return center_point_cloud, line_set

def transform_pointcloud(pcd, rotation_matrix, translation_matrix):
    """Transform the point cloud to the new coordinate system.
    
        basic principle of matrices:
    
            A = Y^(-1)B '^(-1) is the same as inverse of a matix'
            <=>
            AB = Y
    """
    # Step 1: Apply translation (you already did this)
    pcd.translate(-translation_matrix.flatten())

    open3d_world = np.eye(3)
    open3d_world[-1,-1] = -1
    resulting_matrix = np.dot(rotation_matrix, open3d_world)
    transformation_matrix = np.linalg.inv(resulting_matrix)

    pcd.rotate(transformation_matrix, (0,0,0))
    return pcd

def filter_negative_z(pcd):
    """Remove all points with a negative Z-value from the point cloud."""
    points = np.asarray(pcd.points)
    filtered_points = points[points[:, 2] >= 0]  # Keep only points where y >= 0
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    return filtered_pcd

def filter_outliers_from_pointcloud(pcd):
    """Remove outliers using Statistical Outlier Removal and Radius Outlier Removal."""
    # Step 1: Statistical Outlier Removal
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=5, std_ratio=1.0)
    pcd_cleaned = pcd.select_by_index(ind)

    # Step 2: Radius Outlier Removal
    cl, ind = pcd_cleaned.remove_radius_outlier(nb_points=1, radius=0.5)
    pcd_cleaned = pcd_cleaned.select_by_index(ind)

    return pcd_cleaned

def rotation_matrix(avg_rot):
    normal_vector = avg_rot @ [[0],[0],[1]]
    return normal_vector

def visualise_pointclouds(file_path, avg_rotation, avg_translation, save_name):
    """Visualize transformed point clouds and rotate them around their own Z-axis incrementally."""
    npz_data = np.load(file_path, allow_pickle=True)
    point_clouds = [o3d.geometry.PointCloud(o3d.utility.Vector3dVector(npz_data[key])) for key in npz_data]

    transformed_pointclouds = []
    n = len(point_clouds)  # Number of point clouds for incremental rotation

    for i, pcd in enumerate(point_clouds):
        # Step 1: Transform to new coordinate system
        transformed_pcd = transform_pointcloud(pcd, avg_rotation, avg_translation)

        # Step 2: Compute the incremental rotation angle
        rotation_angle = (2 * np.pi * i) / n  # Incremental rotation per point cloud

        # Step 3: Create a Z-axis rotation matrix
        R_z = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, -rotation_angle])

        # Step 4: Rotate the point cloud around its own Z-axis
        transformed_pcd.rotate(R_z, center=(0, 0, 0))  # Rotate around the new origin

        transformed_pointclouds.append(transformed_pcd)

    # Visualize the transformed and rotated point clouds
    center_point_cloud, line_set = visualize_center_and_normal([0, 0, 0], [0,0,-1])
    print("Visualizing original transformed point clouds...")
    o3d.visualization.draw_geometries([center_point_cloud, line_set] + transformed_pointclouds)

    # Step 4: Filter out points with z < 0
    filtered_pointclouds = [filter_negative_z(pcd) for pcd in transformed_pointclouds]

    # Filter outliers from the transformed point clouds
    cleaned_pointclouds = [filter_outliers_from_pointcloud(pcd) for pcd in filtered_pointclouds]

    # Visualize the filtered point clouds
    print("Visualizing filtered point clouds (z >= 0 only)...")
    o3d.visualization.draw_geometries([center_point_cloud, line_set] + cleaned_pointclouds)

    # Save the final rotated point clouds
    points = [np.asarray(pcd.points) for pcd in cleaned_pointclouds]
    np.savez(save_name, *points)

print("x = right, y = height, z= distance from camera; source: https://github.com/isl-org/Open3D/issues/1347")
# Load and orthogonalize calibration data for each turntable
left_avg_rotation, left_avg_translation = load_calibration_data(r'./static/1_calibration_data/turntable/L_turntable_matrix_data.h5')
right_avg_rotation, right_avg_translation = load_calibration_data(r'./static/1_calibration_data/turntable/R_turntable_matrix_data.h5')

# Use the average rotation axis when calling visualise_pointclouds
visualise_pointclouds('mono_left_pointclouds.npz', left_avg_rotation, left_avg_translation, "merged_left.npz")
visualise_pointclouds('mono_right_pointclouds.npz', right_avg_rotation, right_avg_translation, "merged_right.npz")
visualise_pointclouds('stereo_pointclouds.npz', left_avg_rotation, left_avg_translation, "merged_stereo.npz")

print("Doet transformatie naar matrix en translatie vector zodat pointcloud zijn 'z-as'\
      effectief loodrecht op de plaat is. alles onder 0 kan ook verwijdert worden in theorie")
