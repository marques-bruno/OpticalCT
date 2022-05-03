"""The OpticalCT module."""
from pathlib import Path
import cv2
from matplotlib import projections
import numpy as np
import re
import logging


class Dataset:
    """Store & manage a dataset"""
    sinogram = None
    projections = None
    dir = None
    format = '*.jpg'
    scale = 1.0
    invert = False

    def __init__(self, dir=None, **kwargs):
        """
        Create a Dataset loader.
        
        :param dir: The path to the directory containing the dataset.
        If provided, Dataset.load is called
        :type dir: str, optional
        :param kwargs scaling ratio to apply to the whole dataset
        """
        if dir is None:
            self.cleanup()
            return
        self.load(dir, **kwargs)

    def cleanup(self):
        """Delete data from object. Keeps settings (format, scale, invert)"""
        sinogram = None
        projections = None
        dir = None

    def load(self, dir, **kw):
        """
        Load the dataset stored in the given path.

        If needed, images will be converted to single 8-bit channel, grayscale images.
        scale will be applied if != 1.0 and frames will be inverted (negatives) if
        invert is True.
        Images in the directory must be indexed in their filenames, with or without
         trailing zeroes:
        - (1,2,3,...,10,11,12,...)
        - (001,002,...,010,011,...180,181...,999)
        Both formats are supported. Filename can contain alphabetic characters but
        numerical characters must only be used for frame indexing.

        :param path: The path to the directory containing the dataset
        :type path: str
        :param scale scaling ratio to apply to the whole dataset
        :type scale: int, optional
        :param invert: if set to True, load images as negatives: highly
        attenuated radiations on the detector should appear 'brighter' on a
        projected image.
        :type invert: bool
        """
        self.cleanup()

        self.format = kw.get('format', self.format)
        self.scale = kw.get('scale', self.scale)
        self.invert = kw.get('invert', self.invert)

        self.dir = Path(dir)
        if not self.dir.exists:
            raise ValueError('directory ' + str(self.dir) + ' does not exist')

        self.get_files()

        # read sample to get shape, depth, channels....
        cvt2gray = cv2.IMREAD_UNCHANGED
        if self.scale != 1.0:
            sample = cv2.resize(
                cv2.imread(str(self.files[0]), cvt2gray),
                (0, 0), fx=self.scale, fy=self.scale)
        else:
            sample = cv2.imread(str(self.files[0]), cvt2gray)
        if len(sample) > 2:
            cvt2gray = cv2.IMREAD_GRAYSCALE

        self.projections = np.empty((len(self.files), sample.shape[0], sample.shape[1]),
                                    dtype=sample.dtype)

        i = 0
        for f in self.files:
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

    def get_files(self):
        files = [f for f in self.dir.glob(self.format)]
        length = len(files)
        if length == 0:
            raise ValueError('directory ' + str(self.dir) +
                             ' does not contain any image')

        l_zeros = True
        regex = ''

        for i in range(len(str(length))):
            regex += '[0-9]'
        for f in files:
            if re.search(regex, f.name) is None:
                l_zeros = False
                break
        self.files = []
        if l_zeros is False:
            for i in range(1, length):

                fname = str(files[0]).replace(
                    re.findall(r'\d+', str(files[0]))[0], str(i))
                if Path(fname).exists():
                    self.files.append(Path(fname))
        else:
            self.files = sorted(files)

    def compute_sinogram(self):
        """Compute the sinogram of the loaded dataset."""
        if self.projections is None:
            raise ValueError('Cannot compute sinogram, projections not loaded.')
        
        # initial shape is (n_projections, nrows, ncols)
        # The inverse radon transform from skimage expects sinograms where each
        # column represents one slice (a row in the projection frames) at a
        # different angle (n_projections).
        # set must thus concatenate the projections into sinograms
        # shape (nrows, ncols, nproj)

        p_shape = list(self.projections.shape)
        s_shape = (p_shape[1], p_shape[2], p_shape[0])
        self.sinogram = np.empty(s_shape, dtype='uint8')

        for i in range(self.sinogram.shape[0]):
            self.sinogram[i] = np.rot90(self.projections[:, i])

    def compute_volume(self):
        pass
