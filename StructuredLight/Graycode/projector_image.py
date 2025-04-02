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
        self.graycode_height = int((HEIGHT-1))+1
        self.graycode_width = int((WIDTH-1))+1
        self.graycode = []

    def generate_images(self):
        self.graycode = cv.structured_light_GrayCodePattern.create(self.graycode_width,
                                                                   self.graycode_height)

        patterns = self.graycode.generate()[1]
        print(len(patterns))

        exp_patterns = []

        for pat in patterns:
            img = np.zeros((self.height, self.width), np.uint8)
            for y in range(self.height):
                for x in range(self.width):
                    img[y, x] = pat[int(y), int(x)]
            exp_patterns.append(img)

        white = 255*np.ones((self.height, self.width), np.uint8)  # white
        black = np.zeros((self.height, self.width), np.uint8)     # black

        cv.imwrite(self.projector_directory + '/pattern_white.tiff', white)
        cv.imwrite(self.projector_directory + '/pattern_black.tiff', black)

        if not os.path.exists(self.projector_directory):
            os.mkdir(self.projector_directory)

        VI = 0
        HI = 0
        H = 0
        V = 0

        for i, pat in enumerate(exp_patterns):
            if i > (len(exp_patterns)/2):
                if i%2 != 0:
                    cv.imwrite(self.projector_directory + '/pattern_H_I_' + str(HI).zfill(2) + '.tiff', pat)
                    HI += 1
                else:
                    cv.imwrite(self.projector_directory + '/pattern_H_' + str(VI).zfill(2) + '.tiff', pat)
                    VI += 1
            else:
                if i%2 != 0:
                    cv.imwrite(self.projector_directory + '/pattern_V_I_' + str(H).zfill(2) + '.tiff', pat)
                    H += 1
                else:
                    cv.imwrite(self.projector_directory + '/pattern_V_' + str(V).zfill(2) + '.tiff', pat)
                    V += 1
