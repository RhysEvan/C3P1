import cv2 as cv
import numpy as np
from numpy import ma
import matplotlib.pyplot as plt

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
            self.gradient_image(self.array_hor_masked)

        return array_hor_masked, array_vert_masked

    def shadowmask_scenes(self, white, black):
        """
        Code that generates the mask for removing noise and none structured light field.
        Is currently hard coded to 55, should find a more robust procedure.
        """
        _, image_binary_all_white = cv.threshold(np.squeeze(white), 55, 255, cv.THRESH_BINARY)
        _, image_binary_all_black = cv.threshold(np.squeeze(black),
                                                 55, 255,
                                                 cv.THRESH_BINARY)

        shadow_mask = np.empty(white.shape, dtype=object)

        shadow_mask = image_binary_all_white == image_binary_all_black

        return shadow_mask

    def decode_graycode(self, scan_instance, inverse_scan_instance, reference_mask=None):
        """
        Decodes scan_instance and applies mask to only keep necessary info.
        """
        mask = self.decode_process(scan_instance,
                                   inverse_scan_instance)
        if reference_mask.all() == None:
            final_array = mask
        else:
            array_masked = ma.masked_array(mask, mask= reference_mask)
            final_array = ma.filled(array_masked, 0)
        return final_array

    def decode_process(self, instance, instance_inverse):
        """
        Process in which all graycode frames are used to mathematically determine
        the vertical or horizontal pixel grid of the projector.
        """

        mask = np.full(instance.shape, 0, dtype=object)
        #currently known to be the most effienct and robust way of decoding averaging around
        #1.4s for all four instance when executed at once
        height, width = instance[0].shape[:2]
        num = len(instance)

        # Thresholding based on instance and instance_inverse
        for i, image in enumerate(instance):
            instance[i] = np.where(image >= instance_inverse[i], 1, 0)  # Changed True/False to 1/0

        # Initialize the mask for the first bit plane
        mask= instance

        # Compute the mask for the remaining bit planes
        for i in range(1, num):
            mask[i, :, :] = np.bitwise_xor(mask[i,:,:], mask[i-1,:,:])

        # Calculate the coefficient matrix
        coefficients = np.fromfunction(lambda n, y, x: 2 ** (num - 1 - n),
                                       (num, height, width), dtype=int)

        # Decode the final graycode image
        decoded_image = np.sum(mask * coefficients, axis=0)

        return decoded_image

    def gradient_image(self, array):
        """
        Shows the decoded horizontal grid of pixels projected by the 
        projector into the view of the camera.
        """
        x = np.array(array, dtype=float)
        plt.imshow(x, cmap='jet')
        plt.title('decoded')
        plt.show()

    def show_mask(self, mask):
        """
        Shows generated mask, the mask is an image which gets applied atop the
        decoded horizontal frame and vertical frame to only keep the necessary
        info. Which is the horizontal or vertical grid of pixel locations
        that is projected into the scene by the projector.
        """
        plt.imshow(mask, cmap='jet')
        plt.title('mask')
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
