import numpy as np
from typing import Tuple, Optional, List
import numba
from numba import njit # Import the JIT decorator


# Apply the @njit decorator. Numba will compile this function.
# Signature hints can sometimes help Numba optimize.
# Note: Numba doesn't work directly with instance methods that access `self`
# if the class itself isn't JIT-compiled. We'll make helper functions static
# or pass necessary info explicitly.
@njit(cache=True)  # Cache compilation result for faster subsequent runs
def find_valid_neighbor_numba(inverse_map: np.ndarray, r: int, c: int, sentinel_value: int) -> Tuple[bool, int, int]:
    """
    Numba-optimized version to search 8 neighbors for a valid entry.
    Returns (found: bool, col: int, row: int).
    Returns (False, sentinel_value, sentinel_value) if not found.
    """
    map_rows, map_cols = inverse_map.shape[:2]
    # Numba works well with tuples
    neighbor_offsets = (
        (-1, 0), (1, 0), (0, -1), (0, 1),  # Cardinal
        (-1, -1), (-1, 1), (1, -1), (1, 1)  # Diagonal
    )

    for dr, dc in neighbor_offsets:
        nr, nc = r + dr, c + dc
        # Check bounds
        if 0 <= nr < map_rows and 0 <= nc < map_cols:
            neighbor_val_col = inverse_map[nr, nc, 0]
            neighbor_val_row = inverse_map[nr, nc, 1]
            # Check if valid
            if neighbor_val_row != sentinel_value:  # Assuming checking row is enough
                return True, neighbor_val_col, neighbor_val_row  # Return found flag and values
    return False, sentinel_value, sentinel_value  # Return not found flag and sentinel values


# Helper function to contain the loops that we want to JIT compile
@staticmethod
@njit(cache=True)
def _process_neighbor_search_numba(
        left_only_mask: np.ndarray,
        right_only_mask: np.ndarray,
        left_map_crop: np.ndarray,
        right_map_crop: np.ndarray,
        inverse_left_cam: np.ndarray,  # Full map for neighbor search
        inverse_right_cam: np.ndarray,  # Full map for neighbor search
        width_left: int,
        width_right: int,
        sentinel_value: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba-optimized function to perform neighbor searches and calculate indices.
    """
    # Numba works better with pre-allocated arrays or lists built inside
    # Let's build lists first, as size isn't known perfectly beforehand
    indexl_neighbor_list = []
    indexr_neighbor_list = []

    rows, cols = left_only_mask.shape

    # Loop for Case 2: Left valid, Right invalid -> Search neighbors in Right map
    for r in range(rows):
        for c in range(cols):
            if left_only_mask[r, c]:
                left_col = left_map_crop[r, c, 0]
                left_row = left_map_crop[r, c, 1]

                # Find a valid neighbor in the *right* map
                found, right_n_col, right_n_row = find_valid_neighbor_numba(
                    inverse_right_cam, r, c, sentinel_value
                )

                if found:
                    # Calculate flattened indices using original left and found right
                    indexl_neighbor_list.append(left_row * width_left + left_col)
                    indexr_neighbor_list.append(right_n_row * width_right + right_n_col)

    # Loop for Case 3: Right valid, Left invalid -> Search neighbors in Left map
    for r in range(rows):
        for c in range(cols):
            if right_only_mask[r, c]:
                right_col = right_map_crop[r, c, 0]
                right_row = right_map_crop[r, c, 1]

                # Find a valid neighbor in the *left* map
                found, left_n_col, left_n_row = find_valid_neighbor_numba(
                    inverse_left_cam, r, c, sentinel_value
                )

                if found:
                    # Calculate flattened indices using found left and original right
                    indexl_neighbor_list.append(left_n_row * width_left + left_n_col)
                    indexr_neighbor_list.append(right_row * width_right + right_col)

    # Convert lists to NumPy arrays - Numba handles this conversion efficiently
    # Specify dtype for empty list case
    dtype = np.int64  # Or appropriate integer type
    indexl_arr = np.array(indexl_neighbor_list, dtype=dtype)
    indexr_arr = np.array(indexr_neighbor_list, dtype=dtype)
    return indexl_arr, indexr_arr


def process_frames_numba(self,
                         inverse_left_cam: np.ndarray,
                         inverse_right_cam: np.ndarray,
                         shape_left: Tuple[int, int],
                         shape_right: Tuple[int, int],
                         sentinel_value: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Processes inverse maps using Numba JIT for neighbor search loops.
    """
    # --- (Same setup as process_frames_optimized until mask creation) ---
    shape_invers_left = inverse_left_cam.shape
    shape_invers_right = inverse_right_cam.shape
    min_rows = min(shape_invers_left[0], shape_invers_right[0])
    min_cols = min(shape_invers_left[1], shape_invers_right[1])

    left_map_crop = inverse_left_cam[:min_rows, :min_cols]
    right_map_crop = inverse_right_cam[:min_rows, :min_cols]

    left_orig_col = left_map_crop[..., 0]
    left_orig_row = left_map_crop[..., 1]
    right_orig_col = right_map_crop[..., 0]
    right_orig_row = right_map_crop[..., 1]

    valid_left_mask = left_orig_row != sentinel_value
    valid_right_mask = right_orig_row != sentinel_value

    # --- Case 1: Both valid (Pure NumPy, no change) ---
    both_valid_mask = valid_left_mask & valid_right_mask
    left_orig_row_bv = left_orig_row[both_valid_mask]
    left_orig_col_bv = left_orig_col[both_valid_mask]
    right_orig_row_bv = right_orig_row[both_valid_mask]
    right_orig_col_bv = right_orig_col[both_valid_mask]

    width_left = shape_left[1]
    width_right = shape_right[1]

    flat_indices_l_bv = left_orig_row_bv * width_left + left_orig_col_bv
    flat_indices_r_bv = right_orig_row_bv * width_right + right_orig_col_bv
    # Ensure consistent dtype
    flat_indices_l_bv = flat_indices_l_bv.astype(np.int64, copy=False)
    flat_indices_r_bv = flat_indices_r_bv.astype(np.int64, copy=False)

    # --- Cases 2 & 3: Neighbor search using Numba ---
    left_only_mask = valid_left_mask & ~valid_right_mask
    right_only_mask = ~valid_left_mask & valid_right_mask

    # Call the Numba-compiled helper function
    # Pass all necessary data
    indexl_neighbor_arr, indexr_neighbor_arr = self._process_neighbor_search_numba(
        left_only_mask, right_only_mask,
        left_map_crop, right_map_crop,
        inverse_left_cam, inverse_right_cam,  # Pass full maps for searching
        width_left, width_right, sentinel_value
    )

    # --- Combine results ---
    final_indexl = np.concatenate((flat_indices_l_bv, indexl_neighbor_arr))
    final_indexr = np.concatenate((flat_indices_r_bv, indexr_neighbor_arr))

    return final_indexl, final_indexr
