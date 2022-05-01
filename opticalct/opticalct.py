"""The OpticalCT module."""
from pathlib import Path
import cv2
import numpy as np

class Dataset:
    projections = None

    def load(self, dir):
        self.dir = Path(dir)
        if not self.dir.exists:
            raise ValueError('directory ' + str(self.dir) + ' does not exist')
        files = sorted([f for f in self.dir.glob('*.tif')])
        length = len(files)
        if length == 0:
            raise ValueError('directory ' + str(self.dir) + ' does not contain any image')

        cvt2gray = cv2.IMREAD_UNCHANGED
        sample = cv2.imread(str(files[0]), cv2.IMREAD_UNCHANGED)
        if len(sample) > 2:
            cvt2gray = cv2.IMREAD_GRAYSCALE

        print(sample.shape)
        self.projections = np.empty((length, sample.shape[0], sample.shape[1]), dtype=sample.dtype)

        i = 0
        for f in files:
            self.projections[i] = cv2.imread(str(f), cvt2gray)
            i += 1
