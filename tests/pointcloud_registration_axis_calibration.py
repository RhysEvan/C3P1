import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

def load_point_clouds(filepaths):
    """
    Load point clouds from given file paths.
    Returns a list of Open3D point cloud objects.
    """
    point_clouds = []
    for filepath in filepaths:
        pcd = o3d.io.read_point_cloud(filepath)
        point_clouds.append(pcd)
    return point_clouds

def preprocess_point_cloud(pcd, voxel_size=0.02):
    """
    Downsample the point cloud and estimate normals.
    """
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    return pcd_down

def compute_fpfh_feature(pcd_down, voxel_size):
    """
    Compute FPFH feature for the downsampled point cloud.
    """
    radius_normal = voxel_size * 2
    radius_feature = voxel_size * 5
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return fpfh

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    """
    Perform global registration to roughly align source and target point clouds.
    """
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    return result

def refine_registration(source, target, initial_transformation, voxel_size):
    """
    Refine the alignment of source to target using ICP.
    """
    distance_threshold = voxel_size * 0.4
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, initial_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

def align_point_clouds(pcds, voxel_size=0.02):
    """
    Align the point clouds using global registration followed by ICP.
    """
    # Preprocess the point clouds
    pcds_down = [preprocess_point_cloud(pcd, voxel_size) for pcd in pcds]

    # Compute FPFH features
    fpfhs = [compute_fpfh_feature(pcd_down, voxel_size) for pcd_down in pcds_down]

    # Choose one point cloud as the base, and align the others to it
    transformations = []
    current_transform = np.identity(4)

    for i in range(1, len(pcds)):
        source = pcds[i]
        target = pcds[0] if i == 1 else pcds[i-1]

        source_down = pcds_down[i]
        target_down = pcds_down[0] if i == 1 else pcds_down[i-1]

        source_fpfh = fpfhs[i]
        target_fpfh = fpfhs[0] if i == 1 else fpfhs[i-1]

        # Global registration
        result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)

        # ICP refinement
        result_icp = refine_registration(source, target, result_ransac.transformation, voxel_size)

        # Update the transformation
        current_transform = result_icp.transformation @ current_transform
        transformations.append(current_transform)

    return transformations

def compute_rotation_axis(transformations, angle_increment):
    """
    Compute the rotation axis from the transformations and known angle increment.
    """
    rotation_axes = []
    for i, transformation in enumerate(transformations):
        rotation = R.from_matrix(transformation[:3, :3])
        # Assume rotations are around a common axis
        axis = rotation.as_rotvec() / (np.linalg.norm(rotation.as_rotvec()) + 1e-8)
        rotation_axes.append(axis)

    # Average the rotation axes
    rotation_axes = np.array(rotation_axes)
    avg_rotation_axis = np.mean(rotation_axes, axis=0)
    avg_rotation_axis /= np.linalg.norm(avg_rotation_axis)  # Normalize the axis

    return avg_rotation_axis

def compute_object_center(transformations):
    """
    Compute the object's center position from the translations in the transformations.
    """
    translations = [transformation[:3, 3] for transformation in transformations]
    center = np.mean(translations, axis=0)
    return center

def find_turntable_axis(filepaths, angle_increment=30.0, voxel_size=0.02):
    """
    Find the center axis and object position relative to the turntable given point cloud scans.
    """
    point_clouds = load_point_clouds(filepaths)
    transformations = align_point_clouds(point_clouds, voxel_size)

    rotation_axis = compute_rotation_axis(transformations, angle_increment)
    object_center = compute_object_center(transformations)

    return rotation_axis, object_center

# Example usage
filepaths = ["scan1.pcd", "scan2.pcd", "scan3.pcd", "scan4.pcd", "scan5.pcd", "scan6.pcd", "scan7.pcd", "scan8.pcd",
             "scan9.pcd", "scan10.pcd", "scan11.pcd", "scan12.pcd"]
rotation_axis, object_center = find_turntable_axis(filepaths, angle_increment=30.0)
print(f"The estimated rotation axis is: {rotation_axis}")
print(f"The estimated center position of the object is: {object_center}")
