import numpy as np
import open3d as o3d
import copy

def load_pointclouds_from_npz(npz_file):
    """Load the point clouds from a .npz file."""
    npz_data = np.load(npz_file, allow_pickle=True)
    point_clouds = []
    for key in npz_data:
        points = npz_data[key]
        if points.shape[0] > 0:
            point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            point_clouds.append(point_cloud)
    return point_clouds

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556])

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(left, right, stereo, voxel_size):
    print(":: Load two point clouds and disturb initial pose.")
    left_source, left_source_fpfh = preprocess_point_cloud(left, voxel_size)
    right_source, right_source_fpfh = preprocess_point_cloud(right, voxel_size)
    stereo_source, stereo_target_fpfh = preprocess_point_cloud(stereo, voxel_size)
    return left_source, right_source, stereo_source, left_source_fpfh, right_source_fpfh, stereo_target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

# Load point clouds from the npz files
left_pcds = load_pointclouds_from_npz('merged_left.npz')
right_pcds = load_pointclouds_from_npz('merged_right.npz')
stereo_pcds = load_pointclouds_from_npz('merged_stereo.npz')

# Check if any point clouds were loaded
if not left_pcds or not right_pcds or not stereo_pcds:
    print("One or more point cloud lists are empty!")

# Merge all the cleaned point clouds
left_pc = o3d.geometry.PointCloud()
for cleaned_pcd in left_pcds:
    left_pc += cleaned_pcd

right_pc = o3d.geometry.PointCloud()
for cleaned_pcd in right_pcds:
    right_pc += cleaned_pcd

stereo_pc = o3d.geometry.PointCloud()
for cleaned_pcd in stereo_pcds:
    stereo_pc += cleaned_pcd

voxel_size = 3  # means 3cm for this dataset
left_source, right_source, stereo_source, left_source_fpfh, right_source_fpfh, stereo_source_fpfh = \
    prepare_dataset(left_pc, right_pc, stereo_pc, voxel_size)

# Track transformations
transformations = []

# Register left to right
result_ransac = execute_global_registration(left_source, right_source,
                                            left_source_fpfh, right_source_fpfh,
                                            voxel_size)
transformations.append(result_ransac.transformation)
print(result_ransac)
draw_registration_result(left_source, right_source, result_ransac.transformation)

result_icp = refine_registration(left_source, right_source, left_source_fpfh, right_source_fpfh,
                                 voxel_size)
transformations.append(result_icp.transformation)
print(result_icp)
draw_registration_result(left_pc, right_pc, result_icp.transformation)

# Register left to stereo
result_ransac = execute_global_registration(left_source, stereo_source,
                                            left_source_fpfh, stereo_source_fpfh,
                                            voxel_size)
transformations.append(result_ransac.transformation)
print(result_ransac)
draw_registration_result(left_source, stereo_source, result_ransac.transformation)

result_icp = refine_registration(left_source, stereo_source, left_source_fpfh, stereo_source_fpfh,
                                 voxel_size)
transformations.append(result_icp.transformation)
print(result_icp)
draw_registration_result(left_pc, stereo_pc, result_icp.transformation)

# Register right to stereo
result_ransac = execute_global_registration(right_source, stereo_source,
                                            right_source_fpfh, stereo_source_fpfh,
                                            voxel_size)
transformations.append(result_ransac.transformation)
print(result_ransac)
draw_registration_result(right_source, stereo_source, result_ransac.transformation)

result_icp = refine_registration(right_source, stereo_source, right_source_fpfh, stereo_source_fpfh,
                                 voxel_size)
transformations.append(result_icp.transformation)
print(result_icp)
draw_registration_result(right_pc, stereo_pc, result_icp.transformation)

# Merge all point clouds into one
final_pc = copy.deepcopy(left_pc)
final_pc += right_pc.transform(transformations[1])
final_pc += stereo_pc.transform(transformations[3])

# Save the final merged point cloud to a file
o3d.io.write_point_cloud("merged_final.ply", final_pc)

# Save the transformations to a file
np.savez("transformations.npz", transformations=transformations)
