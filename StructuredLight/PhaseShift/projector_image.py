import os
import sys
import cv2 as cv
import numpy as np

np.set_printoptions(threshold=sys.maxsize)

class ProjectionPattern():
    def __init__(self, WIDTH, HEIGHT, PROJECTOR_DIRECTORY):
        self.width = WIDTH
        self.height = HEIGHT
        self.projector_directory = PROJECTOR_DIRECTORY

    def generate_images(self, periods,
                        number_of_images,
                        lower, upper):

        patterns = []

        patterns = self.create_sines(periods,
                                     number_of_images,
                                     lower, upper)

        if not os.path.exists(self.projector_directory):
            os.mkdir(self.projector_directory)

        if not os.path.exists(self.projector_directory + r"\gamma/"):
            os.mkdir(self.projector_directory + r"\gamma/")

        white = 255*np.ones((self.height, self.width), np.uint8)  # white
        black = np.zeros((self.height, self.width), np.uint8)     # black

        cv.imwrite(self.projector_directory + '/pattern_white.tiff', white)
        cv.imwrite(self.projector_directory + '\gamma/pattern_white.tiff', white)
        cv.imwrite(self.projector_directory + '/pattern_black.tiff', black)

        offset = round(360 / len(patterns))
        for i, pat in enumerate(patterns):
            cv.imwrite(self.projector_directory + '/pattern_' + str(i*offset).zfill(5) + '.tiff', pat)

        gammas = self.create_gamma()

        for i, pat in enumerate(gammas):
            cv.imwrite(self.projector_directory + '\gamma/pattern_' + str(i).zfill(5) + '.tiff', pat)


    def create_sines(self, periods, number_of_images, lower, upper):
        offset = 2 * np.pi / number_of_images
        # Create an array of x values
        x = np.linspace(0, periods*2*np.pi, self.width)
        # Create an empty array to store the sine waves
        sines = []
        # Fill the array with the sine waves
        for i in range(number_of_images):
            sine = np.sin(x + i*offset)
            sine = np.interp(sine, (sine.min(), sine.max()), (lower, upper))
            sine_array = np.vstack([sine] * self.height).astype(np.uint8)
            sines.append(sine_array)
        return sines

    def create_gamma(self):

        gamma_images = []

        for i in range(256):
            gamma = i*np.ones(self.width)
            gamma_m = np.vstack([gamma] * self.height).astype(np.uint8)
            gamma_images.append(gamma_m)

        return gamma_images

    def create_sines_with_gamma(self, periods, number_of_images, lower, upper, smoothed_gamma_curve):
        offset = 2 * np.pi / number_of_images
        x = np.linspace(0, periods * 2 * np.pi, self.width)
        sines = []

        for i in range(number_of_images):
            # Create a sine wave for the current image
            sine = np.sin(x + i * offset)

            # Normalize sine wave to the range [0, 1] before applying gamma correction
            normalized_sine = (sine - sine.min()) / (sine.max() - sine.min())

            # Apply the smoothed gamma curve to the normalized sine wave
            # We'll map the sine wave to indices of the gamma curve and apply the correction
            indices = (normalized_sine * (len(smoothed_gamma_curve) - 1)).astype(int)
            gamma_corrected_sine = smoothed_gamma_curve[indices]

            # Rescale the gamma-corrected sine wave back to the desired lower and upper range
            sine = np.interp(gamma_corrected_sine, (gamma_corrected_sine.min(), gamma_corrected_sine.max()), (lower, upper))

            # Create the 2D array for the sine wave
            sine_array = np.vstack([sine] * self.height).astype(np.uint8)
            sines.append(sine_array)

        return sines