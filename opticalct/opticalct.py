"""The OpticalCT module."""
from pathlib import Path
import cv2
import numpy as np
import re
import logging


class Dataset:
    projections = None
    dir = None
    format = '*.jpg'
    scale = 1.0
    invert = False

    def __init__(self, dir=None, **kwargs):
        if dir is None:
            return
        self.load(dir, **kwargs)

    def load(self, dir, **kw):
        self.format = kw.get('format', self.format)
        self.scale = kw.get('scale', self.scale)
        self.invert = kw.get('invert', self.invert)

        self.dir = Path(dir)
        if not self.dir.exists:
            raise ValueError('directory ' + str(self.dir) + ' does not exist')
        files = sorted([f for f in self.dir.glob(self.format)])
        length = len(files)
        if length == 0:
            raise ValueError('directory ' + str(self.dir) +
                             ' does not contain any image')

        cvt2gray = cv2.IMREAD_UNCHANGED
        if self.scale != 1.0:
            sample = cv2.resize(
                cv2.imread(str(files[0]), cvt2gray),
                (0, 0), fx=self.scale, fy=self.scale)
        else:
            sample = cv2.imread(str(files[0]), cvt2gray)
        if len(sample) > 2:
            cvt2gray = cv2.IMREAD_GRAYSCALE

        self.projections = np.empty((length, sample.shape[0], sample.shape[1]),
                                    dtype=sample.dtype)

        i = 0
        for f in files:
            logging.warning(str(f))
            if self.scale != 1.0:
                self.projections[i] = cv2.resize(
                    cv2.imread(str(f), cvt2gray),
                    (0, 0), fx=self.scale, fy=self.scale)
            else:
                self.projections[i] = cv2.imread(str(f), cvt2gray)
            if self.invert:
                self.projections[i] = 255 - self.projections[i]
            i += 1

