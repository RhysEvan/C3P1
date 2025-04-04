from linecache import cache

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from llvmlite.ir import FastMathFlags

import C3P1.config.config as cfg
from numba import njit

class Decode_Gray():
    def __init__(self):
        """
        Class dedicated to decoding images
        
        Functions to call to:

        - scene_decoder: for captured data of an object with either 
        a single camera or two. If two are present be sure to include
        an identifier. Only L or R should be written in your identifier.

        - decode_process_calibration: for captured data to calibrate mono 
        setups with a single camera or two.
        """
        self.array_hor_masked = []
        self.array_vert_masked = []

    def scene_decoder(self, item, identifier = None, visualize = False):
        """
        scene_decoder: initalisation procedure of decoding
        assures right images are used for decoding.
        """
        if identifier is None:
            key = "pattern_"
        else:
            key = identifier+"_pattern_"

        mask = self.shadowmask_scenes(item[key+"white"],
                                      item[key+"black"])

        array_vert_masked = self.decode_graycode(item[key+"V"],
                                                 item[key+"V_I"],
                                                 mask)
        array_hor_masked = self.decode_graycode(item[key+"H"],
                                                item[key+"H_I"],
                                                mask)

        if visualize:
            self.show_mask(mask)
            self.show_gradient_image(self.array_hor_masked)

        return array_hor_masked, array_vert_masked

    def shadowmask_scenes(self, white, black):
        """
        Code that generates the mask for removing noise and none structured light field.
        """
        _, image_binary_all_white = cv.threshold(np.squeeze(white), cfg.decoder_threshold, cfg.decoder_maxvalue, cv.THRESH_BINARY)
        _, image_binary_all_black = cv.threshold(np.squeeze(black),
                                                 cfg.decoder_threshold, cfg.decoder_maxvalue,
                                                 cv.THRESH_BINARY)

        shadow_mask = np.empty(white.shape, dtype=object)

        shadow_mask = image_binary_all_white == image_binary_all_black

        return shadow_mask

    def decode_graycode(self, scan_instance, inverse_scan_instance, reference_mask):
        """
        Decodes scan_instance and applies mask to only keep necessary info.
        """
        mask = decode_process(scan_instance,
                              inverse_scan_instance)
        if reference_mask.all() is None:
            final_array = mask
        else:
            array_masked = np.ma.masked_array(mask, mask= reference_mask)
            final_array = np.ma.filled(array_masked, 0)
        return final_array

    def show_gradient_image(self, array):
        """
        Shows the decoded horizontal grid of pixels projected by the 
        projector into the view of the camera.
        """
        x = np.array(array, dtype=float)
        plt.imshow(x, cmap='jet')
        plt.title('Decoded')
        plt.show()

    def show_mask(self, mask):
        """
        Shows generated mask, the mask is an image which gets applied atop the
        decoded horizontal frame and vertical frame to only keep the necessary
        info. Which is the horizontal or vertical grid of pixel locations
        that is projected into the scene by the projector.
        """
        plt.imshow(mask, cmap='jet')
        plt.title('Mask')
        plt.show()

    def decode_process_calibration(self, instance, white, black):
        """
        Decoding process when calibrating, differs from standard procedures
        as the process of calibration includes the checkerboard in it, which
        would break the decoding unless there is a per pixel threshold map.
        """
        mask = np.full(instance.shape, 0, dtype=object)
        thresh = np.squeeze(0.5 * white + 0.5 * black)
        #currently known to be the most effienct and robust way of decoding averaging around
        #1.4s for all four instance when executed at once
        height, width = instance[0].shape[:2]
        num = len(instance)
        # Thresholding based on instance and instance_inverse
        for i, image in enumerate(instance):
            instance[i] = self.binarize(image, thresh)
        mask = instance
        # Compute the mask for the remaining bit planes
        for i in range(1, num):
            mask[i, :, :] = np.bitwise_xor(mask[i,:,:], mask[i-1,:,:])
        # Calculate the coefficient matrix
        coefficients = np.fromfunction(lambda n, y, x: 2 ** (num - 1 - n),
                                       (num, height, width), dtype=int)
        # Decode the final graycode image
        decoded_image = np.sum(mask * coefficients, axis=0)
        return decoded_image

    def binarize(self, src, thresh):
        """
        Transforms image into binary true or false values for graycode
        decoding.
        """
        imgs_thresh = thresh * np.ones_like(src)
        img_bin = np.empty_like(src, dtype=np.uint8)
        img_bin[src>=imgs_thresh] = True
        img_bin[src<imgs_thresh] = False
        return np.squeeze(img_bin)


@njit(parallel=True, cache=True)
def decode_process_jit(instance_np, instance_inverse_np):
    """
    Process in which all graycode frames are used to mathematically determine
    the vertical or horizontal pixel grid of the projector.
    
    This function expects numpy arrays directly, not lists or other structures
    """
    # Get dimensions
    num, height, width = instance_np.shape[:3]

    # Create mask array
    mask = np.zeros_like(instance_np, dtype=np.uint8)

    # Thresholding - compare each pixel in instance vs inverse
    for i in range(num):
        for y in range(height):
            for x in range(width):
                if instance_np[i, y, x] >= instance_inverse_np[i, y, x]:
                    mask[i, y, x] = 1
                else:
                    mask[i, y, x] = 0

    # XOR accumulation
    for i in range(1, num):
        for y in range(height):
            for x in range(width):
                mask[i, y, x] = mask[i, y, x] ^ mask[i-1, y, x]

    # Calculate final decoded image
    decoded_image = np.zeros((height, width), dtype=np.uint32)
    for i in range(num):
        coef = 2 ** (num - 1 - i)
        for y in range(height):
            for x in range(width):
                decoded_image[y, x] += mask[i, y, x] * coef

    return decoded_image

# outside of class because numba cannot handle self calls
def decode_process(instance, instance_inverse):
    """
    Wrapper function that prepares data for the JIT-compiled function.
    """
    # Convert to numpy arrays with explicit type
    instance_np = np.array(instance, dtype=np.uint8)
    instance_inverse_np = np.array(instance_inverse, dtype=np.uint8)

    # Call the JIT function
    return decode_process_jit(instance_np, instance_inverse_np)
