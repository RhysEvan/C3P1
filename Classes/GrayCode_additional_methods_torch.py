import torch
import torch.multiprocessing as mp

def process_neighbors(r_c_pairs, left_map_crop, inverse_right_cam, width_left, width_right, sentinel_value):
    indexl_neighbor = []
    indexr_neighbor = []
    for r, c in r_c_pairs:
        leftframe = left_map_crop[r, c]
        rightframe_neighbor = find_valid_neighbor(inverse_right_cam, r, c, sentinel_value)
        if rightframe_neighbor is not None:
            indexl_neighbor.append(leftframe[1] * width_left + leftframe[0])
            indexr_neighbor.append(rightframe_neighbor[1] * width_right + rightframe_neighbor[0])
    return indexl_neighbor, indexr_neighbor

def find_valid_neighbor(inverse_map, r, c, sentinel_value):
    # Dummy implementation of find_valid_neighbor
    # Replace with actual logic
    return inverse_map[r, c] if inverse_map[r, c][0] != sentinel_value else None

def process_frames_optimized_pytorch(self,inverse_left_cam, inverse_right_cam, shape_left, shape_right, sentinel_value=0):
    device = inverse_left_cam.device

    min_rows = min(inverse_left_cam.shape[0], inverse_right_cam.shape[0])
    min_cols = min(inverse_left_cam.shape[1], inverse_right_cam.shape[1])

    left_map_crop = inverse_left_cam[:min_rows, :min_cols]
    right_map_crop = inverse_right_cam[:min_rows, :min_cols]

    left_orig_col = left_map_crop[..., 0]
    left_orig_row = left_map_crop[..., 1]
    right_orig_col = right_map_crop[..., 0]
    right_orig_row = right_map_crop[..., 1]

    valid_left_mask = left_orig_row != sentinel_value
    valid_right_mask = right_orig_row != sentinel_value

    both_valid_mask = valid_left_mask & valid_right_mask

    left_orig_row_bv = left_orig_row[both_valid_mask]
    left_orig_col_bv = left_orig_col[both_valid_mask]
    right_orig_row_bv = right_orig_row[both_valid_mask]
    right_orig_col_bv = right_orig_col[both_valid_mask]

    width_left = shape_left[1]
    width_right = shape_right[1]

    flat_indices_l_bv = left_orig_row_bv * width_left + left_orig_col_bv
    flat_indices_r_bv = right_orig_row_bv * width_right + right_orig_col_bv

    left_only_mask = valid_left_mask & ~valid_right_mask
    rows_lo, cols_lo = torch.where(left_only_mask)
    r_c_pairs_lo = list(zip(rows_lo.tolist(), cols_lo.tolist()))

    right_only_mask = ~valid_left_mask & valid_right_mask
    rows_ro, cols_ro = torch.where(right_only_mask)
    r_c_pairs_ro = list(zip(rows_ro.tolist(), cols_ro.tolist()))

    # Parallel processing
    with mp.Pool(mp.cpu_count()) as pool:
        results_lo = pool.starmap(process_neighbors, [(r_c_pairs_lo[i::mp.cpu_count()], left_map_crop, inverse_right_cam, width_left, width_right, sentinel_value) for i in range(mp.cpu_count())])
        results_ro = pool.starmap(process_neighbors, [(r_c_pairs_ro[i::mp.cpu_count()], right_map_crop, inverse_left_cam, width_left, width_right, sentinel_value) for i in range(mp.cpu_count())])

    indexl_neighbor = [item for sublist in results_lo for item in sublist[0]]
    indexr_neighbor = [item for sublist in results_lo for item in sublist[1]]

    indexl_neighbor += [item for sublist in results_ro for item in sublist[0]]
    indexr_neighbor += [item for sublist in results_ro for item in sublist[1]]

    indexl_neighbor_tensor = torch.tensor(indexl_neighbor, dtype=flat_indices_l_bv.dtype, device=device)
    indexr_neighbor_tensor = torch.tensor(indexr_neighbor, dtype=flat_indices_r_bv.dtype, device=device)

    final_indexl = torch.cat((flat_indices_l_bv, indexl_neighbor_tensor))
    final_indexr = torch.cat((flat_indices_r_bv, indexr_neighbor_tensor))

    return final_indexl, final_indexr

# Example usage
# inverse_left_cam = torch.randint(0, 256, (100, 100, 2), dtype=torch.int32)
# inverse_right_cam = torch.randint(0, 256, (100, 100, 2), dtype=torch.int32)
# shape_left = (100, 100)
# shape_right = (100, 100)
# final_indexl, final_indexr = process_frames_optimized_pytorch(inverse_left_cam, inverse_right_cam, shape_left, shape_right)
