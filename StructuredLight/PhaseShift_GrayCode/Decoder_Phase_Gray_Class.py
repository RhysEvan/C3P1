import os
import re
import glob
import cv2
import numpy as np
import plotly.offline as po
import plotly.graph_objs as go
from CameraModel import GenICamCamera
import logging
import torch

class PhaseShifting:
    def __init__(self, width, height, step, gamma, output_dir, black_thr, white_thr, filter_size, input_prefix, config_file):
        self.WIDTH = width
        self.HEIGHT = height
        self.STEP = step
        self.GAMMA = gamma
        self.OUTPUTDIR = output_dir
        self.BLACKTHR = black_thr
        self.WHITETHR = white_thr
        self.FILTER = filter_size
        self.INPUTPRE = input_prefix
        self.CONFIG_FILE = config_file
        self.logger = logging.getLogger(__name__)

    def generate(self):
        GC_STEP = int(self.STEP / 2)

        if not os.path.exists(self.OUTPUTDIR):
            os.mkdir(self.OUTPUTDIR)

        imgs = []

        print('Generating sinusoidal patterns ...')
        angle_vel = np.array((6, 4)) * np.pi / self.STEP
        xs = np.array(range(self.WIDTH))
        for i in range(1, 3):
            for phs in range(1, 4):
                vec = 0.5 * (np.cos(xs * angle_vel[i - 1] + np.pi * (phs - 2) * 2 / 3) + 1)
                vec = 255 * (vec ** self.GAMMA)
                vec = np.round(vec)
                img = np.zeros((self.HEIGHT, self.WIDTH), np.uint8)
                for y in range(self.HEIGHT):
                    img[y, :] = vec
                imgs.append(img)

        ys = np.array(range(self.HEIGHT))
        for i in range(1, 3):
            for phs in range(1, 4):
                vec = 0.5 * (np.cos(ys * angle_vel[i - 1] + np.pi * (phs - 2) * 2 / 3) + 1)
                vec = 255 * (vec ** self.GAMMA)
                vec = np.round(vec)
                img = np.zeros((self.HEIGHT, self.WIDTH), np.uint8)
                for x in range(self.WIDTH):
                    img[:, x] = vec
                imgs.append(img)

        print('Generating graycode patterns ...')
        gc_height = int((self.HEIGHT - 1) / GC_STEP) + 1
        gc_width = int((self.WIDTH - 1) / GC_STEP) + 1

        graycode = cv2.structured_light_GrayCodePattern.create(gc_width, gc_height)
        patterns = graycode.generate()[1]
        for pat in patterns:
            img = np.zeros((self.HEIGHT, self.WIDTH), np.uint8)
            for y in range(self.HEIGHT):
                for x in range(self.WIDTH):
                    img[y, x] = pat[int(y / GC_STEP), int(x / GC_STEP)]
            imgs.append(img)
        imgs.append(255 * np.ones((self.HEIGHT, self.WIDTH), np.uint8))  # white
        imgs.append(np.zeros((self.HEIGHT, self.WIDTH), np.uint8))  # black

        for i, img in enumerate(imgs):
            #cv2.putText(img, str(i+1), (500, 500), cv2.FONT_HERSHEY_SIMPLEX, 20, (255, 155, 0), 20)
            cv2.imwrite(self.OUTPUTDIR + '/pat' + str(i).zfill(2) + '.png', img)

        print('Saving config file ...')
        fs = cv2.FileStorage(self.OUTPUTDIR + '/config.xml', cv2.FILE_STORAGE_WRITE)


        fs.write('disp_width', self.WIDTH)
        fs.write('disp_height', self.HEIGHT)
        fs.write('step', self.STEP)
        fs.release()

        print('Done')

    def load_images_from_folder(self):
        self.logger.info('Loading images ...')
        re_num = re.compile(r'(\d+)')

        def numerical_sort(text):
            return int(re_num.split(text)[-2])

        filenames = sorted(glob.glob(self.INPUTPRE + '\pat*.png'), key=numerical_sort)
        if len(filenames) != self.graycode.getNumberOfPatternImages() + 14:
            print('Number of images is not right (right number is ' + str(self.graycode.getNumberOfPatternImages() + 14) + ')')
            return

        imgs = []
        for f in filenames:
            imgs.append(cv2.imread(f, cv2.IMREAD_GRAYSCALE))

        return imgs

    def set_decode_settings(self):

        fs = cv2.FileStorage(self.CONFIG_FILE, cv2.FILE_STORAGE_READ)
        DISP_WIDTH = int(fs.getNode('disp_width').real())
        DISP_HEIGHT = int(fs.getNode('disp_height').real())
        STEP = int(fs.getNode('step').real())
        self.GC_STEP = int(STEP / 2)
        fs.release()

        gc_width = int((DISP_WIDTH - 1) / self.GC_STEP) + 1
        gc_height = int((DISP_HEIGHT - 1) / self.GC_STEP) + 1
        self.graycode = cv2.structured_light_GrayCodePattern.create(gc_width, gc_height)
        self.graycode.setBlackThreshold(self.BLACKTHR)
        self.graycode.setWhiteThreshold(self.WHITETHR)


    def decode_graycode(self,CAM_HEIGHT, CAM_WIDTH, gc_imgs, black, white):
        gc_map = np.zeros((CAM_HEIGHT, CAM_WIDTH, 2), np.int16)
        mask = np.zeros((CAM_HEIGHT, CAM_WIDTH), np.uint8)
        for y in range(CAM_HEIGHT):
            for x in range(CAM_WIDTH):
                if int(white[y, x]) - int(black[y, x]) <= self.BLACKTHR:
                    continue
                err, proj_pix = self.graycode.getProjPixel(gc_imgs, x, y)
                if not err:
                    gc_map[y, x, :] = np.array(proj_pix)
                    mask[y, x] = 255

        return gc_map, mask
    def filter(self,gc_map, mask, CAM_HEIGHT, CAM_WIDTH):
        self.logger.info('Applying smoothing filter ...')
        ext_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((self.FILTER * 2 + 1, self.FILTER * 2 + 1)))
        for y in range(CAM_HEIGHT):
            for x in range(CAM_WIDTH):
                if mask[y, x] == 0 and ext_mask[y, x] != 0:
                    sum_x = 0
                    sum_y = 0
                    cnt = 0
                    for dy in range(-self.FILTER, self.FILTER + 1):
                        for dx in range(-self.FILTER, self.FILTER + 1):
                            ty = y + dy
                            tx = x + dx
                            if ((dy != 0 or dx != 0) and ty >= 0 and ty < CAM_HEIGHT and tx >= 0 and tx < CAM_WIDTH and
                                    mask[ty, tx] != 0):
                                sum_x += gc_map[ty, tx, 0]
                                sum_y += gc_map[ty, tx, 1]
                                cnt += 1
                    if cnt != 0:
                        gc_map[y, x, 0] = np.round(sum_x / cnt)
                        gc_map[y, x, 1] = np.round(sum_y / cnt)
        return gc_map,ext_mask

    def filter_optimized(self, gc_map, mask, CAM_HEIGHT, CAM_WIDTH):
        """
        Applies a smoothing filter to gc_map based on a mask using optimized operations.

        Fills pixels in gc_map where mask is 0 but its morphological closing is non-zero,
        using the average of valid neighbors (mask != 0) within a defined window,
        excluding the center pixel itself.

        Args:
            gc_map: Input map (H, W, 2) - likely containing coordinates or similar data.
            mask: Input mask (H, W) - typically 0 for background/invalid, non-zero for foreground/valid.
            CAM_HEIGHT: Height of the images/maps.
            CAM_WIDTH: Width of the images/maps.

        Returns:
            tuple: (filtered_gc_map, ext_mask)
                   filtered_gc_map: The gc_map with holes filled.
                   ext_mask: The morphologically closed mask.
        """
        self.logger.info('Applying optimized smoothing filter ...')

        if self.FILTER <= 0:
            self.logger.warning('FILTER <= 0, skipping smoothing.')
            # Still calculate ext_mask for consistency if needed, using a minimal kernel
            ext_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((1, 1), dtype=np.uint8))
            return gc_map.copy(), ext_mask  # Return a copy to avoid modifying original map

        # 1. Perform morphological closing to get ext_mask (same as original)
        kernel_size_morph = self.FILTER * 2 + 1
        morph_kernel = np.ones((kernel_size_morph, kernel_size_morph), np.uint8)
        ext_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, morph_kernel)

        # 2. Identify target pixels to fill
        # Pixels where original mask is 0, but the closed mask is non-zero
        target_pixels_mask = (mask == 0) & (ext_mask != 0)

        # If no pixels need modification, return early
        if not np.any(target_pixels_mask):
            self.logger.info('No pixels to filter.')
            return gc_map.copy(), ext_mask  # Return a copy

        # 3. Prepare data for vectorized calculation
        # Ensure mask is binary (0 or 1) and float for calculations
        # Assuming mask uses 0 and some non-zero value (e.g., 255)
        mask_binary_float = (mask > 0).astype(np.float32)

        # Ensure gc_map is float for calculations to maintain precision during averaging
        gc_map_float = gc_map.astype(np.float32)

        # 4. Calculate sum and count of valid neighbors using cv2.boxFilter
        kernel_size_box = self.FILTER * 2 + 1
        box_kernel_shape = (kernel_size_box, kernel_size_box)

        # Calculate the sum of gc_map values where the mask is valid within the window
        # We multiply gc_map by the binary mask first, so invalid pixels contribute 0
        # Need to broadcast mask_binary_float to match gc_map_float channels
        masked_gc_map = gc_map_float * mask_binary_float[:, :, np.newaxis]
        sum_neighbors = cv2.boxFilter(masked_gc_map, -1, box_kernel_shape,
                                      normalize=False,
                                      borderType=cv2.BORDER_CONSTANT)  # Use BORDER_CONSTANT(0) to handle edges like loop bounds checks

        # Calculate the count of valid pixels (mask != 0) within the window
        count_neighbors = cv2.boxFilter(mask_binary_float, -1, box_kernel_shape,
                                        normalize=False, borderType=cv2.BORDER_CONSTANT)
        # Add channel dimension for broadcasting compatibility
        count_neighbors = count_neighbors[:, :, np.newaxis]

        # 5. Adjust sum and count to exclude the center pixel (matching original logic)
        # Subtract the center pixel's contribution if it was valid (i.e., where mask was > 0)
        adjusted_sum = sum_neighbors - masked_gc_map  # masked_gc_map is already 0 where mask is 0

        # Subtract 1 from the count if the center pixel was valid
        adjusted_count = count_neighbors - mask_binary_float[:, :, np.newaxis]

        # 6. Calculate the average value, avoiding division by zero
        # Initialize average_neighbors with zeros
        average_neighbors = np.zeros_like(adjusted_sum, dtype=np.float32)

        # Create a mask where the adjusted count is positive to avoid division by zero
        # Need to handle 2 channels in adjusted_count properly
        valid_count_mask = adjusted_count > 1e-6  # Use a small epsilon for float comparison

        # Perform division only where the count is positive
        # 6. Calculate the average value, avoiding division by zero
        # Initialize average_neighbors with zeros
        average_neighbors = np.zeros_like(adjusted_sum, dtype=np.float32)

        # Create a mask where the adjusted count is positive. Shape is (H, W, 1)
        valid_count_mask = adjusted_count > 1e-6 # Use a small epsilon for float comparison

        # Broadcast the mask to match the output shape (H, W, 2) for the 'where' clause
        where_mask_broadcast = np.broadcast_to(valid_count_mask, adjusted_sum.shape)

        # Use np.divide for safe division.
        # adjusted_count (H, W, 1) will be broadcast automatically during division against adjusted_sum (H, W, 2).
        # The 'where' clause ensures we only divide where the count is valid.
        # 'out=average_neighbors' places the result directly into the pre-allocated array.
        np.divide(adjusted_sum, adjusted_count, out=average_neighbors, where=where_mask_broadcast)

        # Pixels where where_mask_broadcast
        # 7. Update the gc_map only at the target pixels
        gc_map_filtered = gc_map.copy()  # Work on a copy

        # Round the calculated average values and convert back to original gc_map dtype
        update_values = np.round(average_neighbors[target_pixels_mask]).astype(gc_map.dtype)

        # Apply the updates
        gc_map_filtered[target_pixels_mask] = update_values

        self.logger.info('Optimized smoothing filter applied.')
        return gc_map_filtered, ext_mask

    def decode_pixel_vectorized(self, gc_map, mask, CAM_HEIGHT, CAM_WIDTH, ps_map_x1, ps_map_x2, ps_map_y1, ps_map_y2):
        """
        Vectorized version of decode_pixel using NumPy.
        NOTE: The sequential updates inside the original decode_pixel logic
              make direct vectorization very complex and potentially less readable.
              The logic below attempts to replicate it, but Numba (Approach 2)
              is often better suited for such cases.
        """
        self.logger.info('Decoding each pixel (Vectorized)...')

        # Create masks for valid pixels
        valid_mask = (mask != 0)
        # Get coordinates of valid pixels
        y_coords, x_coords = np.nonzero(valid_mask)

        # Extract data only for valid pixels into 1D arrays
        gc_valid = gc_map[y_coords, x_coords]  # Shape: (N, 2) where N is num valid pixels
        ps_x1_valid = ps_map_x1[y_coords, x_coords]  # Shape: (N,)
        ps_x2_valid = ps_map_x2[y_coords, x_coords]  # Shape: (N,)
        ps_y1_valid = ps_map_y1[y_coords, x_coords]  # Shape: (N,)
        ps_y2_valid = ps_map_y2[y_coords, x_coords]  # Shape: (N,)

        gc_x_valid = gc_valid[:, 0]
        gc_y_valid = gc_valid[:, 1]

        # --- Vectorized decode_pixel logic ---
        # This function is applied separately for X and Y dimensions
        def decode_vectorized(gc, ps1, ps2, gc_step, step):
            # Calculate difference 'dif' with wrap-around logic
            # This part is tricky to perfectly vectorize matching the original flow.
            # Let's approximate the angle difference calculation carefully.
            # Calculate difference: ps1 - ps2
            diff = ps1 - ps2

            # Adjust differences outside [-4pi/3, 4pi/3] ?? Original logic is complex.
            # Let's try to replicate the original conditional logic flow for 'dif'
            # Calculate d1=ps1-ps2, d2=ps2-ps1
            d1 = ps1 - ps2
            d2 = ps2 - ps1
            cond_gt = ps1 > ps2

            # Initialize dif array
            dif = np.zeros_like(ps1, dtype=np.float64)

            # Case 1: ps1 > ps2
            mask1a = cond_gt & (d1 > np.pi * 4 / 3)
            mask1b = cond_gt & ~mask1a
            dif[mask1a] = d2[mask1a] + 2 * np.pi
            dif[mask1b] = d1[mask1b]

            # Case 2: ps1 <= ps2 (using ~cond_gt)
            mask2a = (~cond_gt) & (d2 > np.pi * 4 / 3)
            mask2b = (~cond_gt) & ~mask2a
            # Original code: if ps2-ps1 > 4pi/3 => (ps1-ps2)+2pi = d1+2pi
            dif[mask2a] = d1[mask2a] + 2 * np.pi
            dif[mask2b] = d2[mask2b]

            # --- Calculate 'p' ---
            # This is the hardest part due to sequential updates in the original code.
            # Simple vectorization might not perfectly replicate if the second 'if'
            # depends on the modified 'p' from the first 'if'.
            # A Numba approach (see below) handles this naturally.
            # We attempt a direct translation, assuming conditions are checked on original 'p'.

            p = ps1.copy()  # Start with ps1

            is_even = (gc % 2 == 0)
            is_odd = ~is_even

            # --- Even gc ---
            # Condition 1
            cond_even_1 = is_even & (dif > np.pi / 6) & (p < 0)
            p_update_even1 = p + 2 * np.pi
            # Condition 2 (Applied potentially *after* update 1 in original)
            # Vectorizing this sequence dependency is complex. We apply based on original p for now.
            cond_even_2 = is_even & (dif > np.pi / 2) & (p < np.pi * 7 / 6)
            p_update_even2 = p + 2 * np.pi  # Update value for condition 2

            # --- Odd gc ---
            # Condition 1
            cond_odd_1 = is_odd & (dif > np.pi * 5 / 6) & (p > 0)
            p_update_odd1 = p - 2 * np.pi
            # Condition 2 (Applied potentially *after* update 1 in original)
            cond_odd_2 = is_odd & (dif < np.pi / 2) & (p < np.pi / 6)
            p_update_odd2 = p + 2 * np.pi  # Update value for condition 2

            # Apply updates (This simplified vectorization might differ from original sequential logic)
            p_final = p.copy()  # Start with base p = ps1

            # **Warning:** The following applies updates based on initial 'p'.
            # A truly sequential update requires multiple passes or Numba.

            # Even updates (apply second check based on original p)
            p_final = np.where(cond_even_1, p_update_even1, p_final)
            p_final = np.where(cond_even_2, p_update_even2, p_final)

            # Odd updates (apply second check based on original p)
            p_final = np.where(cond_odd_1, p_update_odd1, p_final)
            p_final = np.where(cond_odd_2, p_update_odd2, p_final)

            # Final adjustment for odd gc
            p_final = np.where(is_odd, p_final + np.pi, p_final)

            # Final calculation
            result = gc * gc_step + step * p_final / 3 / 2 / np.pi
            return result

        # Apply the vectorized function
        est_x_valid = decode_vectorized(gc_x_valid, ps_x1_valid, ps_x2_valid, self.GC_STEP, self.STEP)
        est_y_valid = decode_vectorized(gc_y_valid, ps_y1_valid, ps_y2_valid, self.GC_STEP, self.STEP)

        # --- Populate results ---
        # Initialize viz array (maybe with a background value if needed)
        viz = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.float64)  # Use float for est_x/y

        # Place calculated values into viz using advanced indexing
        viz[y_coords, x_coords, 0] = est_x_valid
        viz[y_coords, x_coords, 1] = est_y_valid
        viz[y_coords, x_coords, 2] = 128  # Constant value

        # Create res_list (more efficiently than appending)
        #res_list = list(zip(y_coords.tolist(), x_coords.tolist(), est_y_valid.tolist(), est_x_valid.tolist()))
        # Or using NumPy stacking (might be faster for very large lists):
        res_array = np.stack((y_coords, x_coords, est_y_valid, est_x_valid), axis=-1)
        res_list = res_array
        #res_list = res_array.tolist() # Convert rows to tuples

        # Convert viz to uint16 if required (potential data loss if values are large/negative)
        # Consider the range of est_x, est_y before converting type
        #viz = viz.astype(np.uint16)

        return viz, res_list

    # --- PyTorch implementation ---

    def decode_pixel_pytorch(self,gc_map_np, mask_np, CAM_HEIGHT, CAM_WIDTH,
                             ps_map_x1_np, ps_map_x2_np, ps_map_y1_np, ps_map_y2_np,
                             device=None):
        """
        PyTorch version of the decode_pixel logic.

        Args:
            gc_map_np (np.ndarray): Gray code map (H, W, 2). Should be integer type.
            mask_np (np.ndarray): Mask (H, W), non-zero for pixels to process.
            CAM_HEIGHT (int): Camera height.
            CAM_WIDTH (int): Camera width.
            ps_map_x1_np (np.ndarray): Phase shift map x1 (H, W). Float type.
            ps_map_x2_np (np.ndarray): Phase shift map x2 (H, W). Float type.
            ps_map_y1_np (np.ndarray): Phase shift map y1 (H, W). Float type.
            ps_map_y2_np (np.ndarray): Phase shift map y2 (H, W). Float type.
            GC_STEP (float): Gray code step value.
            STEP (float): Step value.
            device (torch.device, optional): Target device ('cuda', 'cpu').
                                             Defaults to CUDA if available, else CPU.

        Returns:
            tuple: (viz_tensor, res_list)
                - viz_tensor (torch.Tensor): Output visualization (H, W, 3) on the target device, uint16 type.
                - res_list (list): List of tuples (y, x, est_y, est_x) for valid pixels.
        """
        self.logger.info('Decoding each pixel (Torch)...')

        GC_STEP = self.GC_STEP
        STEP = self.STEP
        # --- Device Setup ---
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")  # Optional: for verification

        # --- Convert inputs to PyTorch tensors and move to device ---
        # Use float32 for phase maps and calculations for better GPU performance usually
        dtype = torch.float32
        gc_map = torch.from_numpy(gc_map_np).to(device=device, dtype=torch.int64)  # Keep GC integer
        mask = torch.from_numpy(mask_np).to(device=device)  # Type depends on input, bool/int fine
        ps_map_x1 = torch.from_numpy(ps_map_x1_np).to(device=device, dtype=dtype)
        ps_map_x2 = torch.from_numpy(ps_map_x2_np).to(device=device, dtype=dtype)
        ps_map_y1 = torch.from_numpy(ps_map_y1_np).to(device=device, dtype=dtype)
        ps_map_y2 = torch.from_numpy(ps_map_y2_np).to(device=device, dtype=dtype)
        self.logger.info('Decoding each pixel (Torch 2)...')

        # Constants as tensors or floats (float usually fine unless involved in tensor ops directly)
        gc_step_t = float(GC_STEP)
        step_t = float(STEP)
        pi_t = torch.pi  # Use torch.pi

        # --- Masking ---
        valid_mask = (mask != 0)
        # Get coordinates of valid pixels. nonzero returns a tuple of tensors.
        coords = torch.nonzero(valid_mask, as_tuple=True)  # (tensor_y_coords, tensor_x_coords)
        y_coords, x_coords = coords

        if y_coords.numel() == 0:
            # Handle case with no valid pixels
            print("Warning: No valid pixels found in the mask.")
            viz_empty = torch.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=torch.uint16, device=device)
            return viz_empty, []

        # --- Extract data for valid pixels ---
        # Shape: (N,) where N is num valid pixels
        ps_x1_valid = ps_map_x1[y_coords, x_coords]
        ps_x2_valid = ps_map_x2[y_coords, x_coords]
        ps_y1_valid = ps_map_y1[y_coords, x_coords]
        ps_y2_valid = ps_map_y2[y_coords, x_coords]
        # Shape: (N, 2) -> extract X (:,0) and Y (:,1) components
        gc_valid = gc_map[y_coords, x_coords]
        gc_x_valid = gc_valid[:, 0]  # Shape: (N,)
        gc_y_valid = gc_valid[:, 1]  # Shape: (N,)

        # --- Vectorized decode function (applied to valid pixels) ---
        def decode_tensor(gc, ps1, ps2):
            """ Applies the core decoding logic to tensors of valid pixel data. """

            # --- Calculate difference 'dif' with wrap-around ---
            # Replicating the original conditional logic using torch.where
            d1 = ps1 - ps2
            d2 = ps2 - ps1
            cond_gt = ps1 > ps2

            pi_4_3 = pi_t * 4 / 3
            two_pi = 2 * pi_t

            # Initialize dif tensor
            dif = torch.zeros_like(ps1)

            # Case 1: ps1 > ps2
            mask1a = cond_gt & (d1 > pi_4_3)
            mask1b = cond_gt & ~mask1a
            # If mask1a is true, use d2 + two_pi, else keep current value (0 initially)
            dif = torch.where(mask1a, d2 + two_pi, dif)
            # If mask1b is true, use d1, else keep current value (result from mask1a)
            dif = torch.where(mask1b, d1, dif)

            # Case 2: ps1 <= ps2
            mask2a = (~cond_gt) & (d2 > pi_4_3)
            mask2b = (~cond_gt) & ~mask2a
            # Original: (ps1 - ps2) + 2pi = d1 + 2pi
            # If mask2a is true, use d1 + two_pi, else keep current value (result from Case 1)
            dif = torch.where(mask2a, d1 + two_pi, dif)
            # If mask2b is true, use d2, else keep current value (result from mask2a)
            dif = torch.where(mask2b, d2, dif)

            # --- Calculate 'p' with sequential updates ---
            # We use multiple torch.where calls to mimic the sequential dependency
            p = ps1.clone()  # Start with ps1, clone to allow modification
            is_even = (gc % 2 == 0)  # Boolean tensor
            is_odd = ~is_even  # Boolean tensor

            # Constants for conditions
            pi_d_6 = pi_t / 6
            pi_d_2 = pi_t / 2
            pi_7_d_6 = pi_t * 7 / 6
            pi_5_d_6 = pi_t * 5 / 6

            # --- Even gc ---
            # Check Condition 1
            cond_even_1 = is_even & (dif > pi_d_6) & (p < 0)
            # Update p where condition 1 is true
            p = torch.where(cond_even_1, p + two_pi, p)

            # Check Condition 2 (uses the *updated* p from the previous step)
            cond_even_2 = is_even & (dif > pi_d_2) & (p < pi_7_d_6)
            # Update p where condition 2 is true
            p = torch.where(cond_even_2, p + two_pi, p)

            # --- Odd gc ---
            # Check Condition 1
            cond_odd_1 = is_odd & (dif > pi_5_d_6) & (p > 0)
            # Update p where condition 1 is true
            p = torch.where(cond_odd_1, p - two_pi, p)

            # Check Condition 2 (uses the *updated* p from the previous step)
            cond_odd_2 = is_odd & (dif < pi_d_2) & (p < pi_d_6)
            # Update p where condition 2 is true
            p = torch.where(cond_odd_2, p + two_pi, p)

            # --- Final offset for odd gc ---
            # Apply +pi where gc was odd (uses the p potentially updated multiple times)
            p = torch.where(is_odd, p + pi_t, p)

            # --- Final Calculation ---
            # Ensure gc is float for calculation if needed, or rely on type promotion
            result = gc.to(dtype) * gc_step_t + step_t * p / 3 / 2 / pi_t
            return result

        # --- Apply decoding to X and Y components ---
        est_x_valid = decode_tensor(gc_x_valid, ps_x1_valid, ps_x2_valid)
        est_y_valid = decode_tensor(gc_y_valid, ps_y1_valid, ps_y2_valid)

        # --- Populate Output Tensor 'viz' ---
        # Initialize viz with zeros (float for precision during assignment)
        viz_float = torch.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=dtype, device=device)

        # Place calculated est_x, est_y into the correct locations using valid coordinates
        viz_float[y_coords, x_coords, 0] = est_x_valid
        viz_float[y_coords, x_coords, 1] = est_y_valid
        # Set the constant value (e.g., intensity or confidence)
        viz_float[y_coords, x_coords, 2] = 128.0  # Use float literal

        # Convert final viz tensor to uint16 (consider potential precision loss/clamping)
        # Ensure values are in valid range for uint16 if necessary (e.g., clamp)
        # viz_float.clamp_(min=0, max=65535) # Optional clamping if needed before cast
        viz_uint16 = viz_float.to(torch.uint16)

        # --- Create Result List ---
        # Stack coordinates and results, transfer to CPU, convert to list of tuples
        res_data_gpu = torch.stack((y_coords.to(dtype), x_coords.to(dtype), est_y_valid, est_x_valid), dim=-1)
        res_data_cpu = res_data_gpu.cpu().numpy()  # Transfer to CPU and convert to NumPy
        res_list = [tuple(row) for row in res_data_cpu]  # Convert NumPy rows to tuples

        return viz_uint16, res_list

    def decode_pixel(self,gc_map,mask,CAM_HEIGHT,CAM_WIDTH, ps_map_x1, ps_map_x2, ps_map_y1, ps_map_y2):
        def decode_pixel(gc, ps1, ps2):
            dif = None
            if ps1 > ps2:
                if ps1 - ps2 > np.pi * 4 / 3:
                    dif = (ps2 - ps1) + 2 * np.pi
                else:
                    dif = ps1 - ps2
            else:
                if ps2 - ps1 > np.pi * 4 / 3:
                    dif = (ps1 - ps2) + 2 * np.pi
                else:
                    dif = ps2 - ps1

            p = None
            if gc % 2 == 0:
                p = ps1
                if dif > np.pi / 6 and p < 0:
                    p = p + 2 * np.pi
                if dif > np.pi / 2 and p < np.pi * 7 / 6:
                    p = p + 2 * np.pi
            else:
                p = ps1
                if dif > np.pi * 5 / 6 and p > 0:
                    p = p - 2 * np.pi
                if dif < np.pi / 2 and p < np.pi / 6:
                    p = p + 2 * np.pi
                p = p + np.pi
            return gc * self.GC_STEP + self.STEP * p / 3 / 2 / np.pi

        self.logger.info('Decoding each pixel ...')
        viz = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), np.uint16)
        res_list = []
        for y in range(CAM_HEIGHT):
            for x in range(CAM_WIDTH):
                if mask[y, x] != 0:
                    est_x = decode_pixel(gc_map[y, x, 0], ps_map_x1[y, x], ps_map_x2[y, x])
                    est_y = decode_pixel(gc_map[y, x, 1], ps_map_y1[y, x], ps_map_y2[y, x])
                    viz[y, x, :] = (est_x, est_y, 128)
                    res_list.append((y, x, est_y, est_x))
        return viz, res_list

    def decode_ps(self,ps_imgs_float):
        # Assuming ps_imgs is your input NumPy array of shape (12, height, width)
        # Example dummy data:
        # height, width = 100, 100
        # ps_imgs = np.random.rand(12, height, width).astype(np.uint8) * 255

        # 2. Select all 'pimg1', 'pimg2', and 'pimg3' images using slicing with steps
        #    This creates views, not copies usually, which is efficient.
        #    pimg1 corresponds to indices 0, 3, 6, 9
        #    pimg2 corresponds to indices 1, 4, 7, 10
        #    pimg3 corresponds to indices 2, 5, 8, 11


        # for all images im ps_imgs_float check if float32
        # if not convert to float32
        ps_imgs_float = np.array(ps_imgs_float, dtype=np.float32)


        pimg1_all = ps_imgs_float[0::3]  # Shape: (4, height, width)
        pimg2_all = ps_imgs_float[1::3]  # Shape: (4, height, width)
        pimg3_all = ps_imgs_float[2::3]  # Shape: (4, height, width)

        # 3. Perform the calculation in a single vectorized step
        #    NumPy applies the arctan2 element-wise across the arrays.
        #    The result 'ps_maps_combined' will have shape (4, height, width)
        ps_maps_combined = np.arctan2(np.sqrt(3.0) * (pimg1_all - pimg3_all),
                                      2.0 * pimg2_all - pimg1_all - pimg3_all)


        return ps_maps_combined[0], ps_maps_combined[1],ps_maps_combined[2],ps_maps_combined[3]

        # Now ps_map_x1, ps_map_x2, ps_map_y1, ps_map_y2 contain the results.
    def decode(self,imgs):




        ps_imgs = imgs[0:12]
        gc_imgs = imgs[12:]
        black = gc_imgs.pop()
        white = gc_imgs.pop()
        CAM_WIDTH = white.shape[1]
        CAM_HEIGHT = white.shape[0]

        self.logger.info('Decoding images ps ...')

        ps_map_x1, ps_map_x2, ps_map_y1, ps_map_y2 = self.decode_ps(ps_imgs)
        self.logger.info('Decoding images graycode ...')
        gc_map,mask = self.decode_graycode(CAM_HEIGHT, CAM_WIDTH, gc_imgs, black, white)

        if self.FILTER != 0:
            #gc_map,mask = self.filter(gc_map, mask, CAM_HEIGHT, CAM_WIDTH)
            gc_map,mask = self.filter_optimized(gc_map, mask, CAM_HEIGHT, CAM_WIDTH)

        viz, res_list= self.decode_pixel_vectorized(gc_map,mask,CAM_HEIGHT,CAM_WIDTH, ps_map_x1, ps_map_x2, ps_map_y1, ps_map_y2)
        #viz, res_list = self.decode_pixel_pytorch(gc_map,mask,CAM_HEIGHT,CAM_WIDTH, ps_map_x1, ps_map_x2, ps_map_y1, ps_map_y2)
        self.logger.info('Exporting result ...')

        if isinstance(viz, torch.Tensor):
            viz = viz.cpu().numpy()
        if isinstance(res_list, torch.Tensor):
            res_list= res_list.cpu().numpy()
        if not os.path.exists(self.OUTPUTDIR):
            os.mkdir(self.OUTPUTDIR)
        cv2.imwrite(self.OUTPUTDIR + '/vizualized.png', viz.astype(np.uint8))
        np.save(os.path.join(self.OUTPUTDIR, 'vizualized.npy'), viz)
        with open(self.OUTPUTDIR + '/camera2display.csv', mode='w') as f:
            f.write('camera_y, camera_x, display_y, display_x\n')
            for (cam_y, cam_x, disp_y, disp_x) in res_list:
                f.write(str(cam_y) + ', ' + str(cam_x) + ', ' + str(disp_y) + ', ' + str(disp_x) + '\n')

        self.logger.info('Done')


if __name__ == '__main__':
    # Set the parameters here
    width = 1280
    height = 720
    step = 32
    gamma = 1.0
    output_dir = 'output'
    black_thr = 15
    white_thr = 15
    filter_size = 2
    input_prefix = r"C:\Users\Seppe\PycharmProjects\phase\phase-shifting\sample_data\objectS_test\moved_fixedgray"
    config_file = r"C:\Users\Seppe\PycharmProjects\phase\phase-shifting\sample_data\objectS\config.xml"

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


    ps = PhaseShifting(width, height, step, gamma, output_dir, black_thr, white_thr, filter_size, input_prefix, config_file)
    #ps.generate()

    ps.set_decode_settings()
    imgs = ps.load_images_from_folder()
    ps.OUTPUTDIR = os.path.join(input_prefix, 'results')
    ps.decode(imgs)