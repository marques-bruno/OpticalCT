import unittest
from opticalct.opticalct import Dataset
from pathlib import Path
import numpy as np
import pytest
import os
import re


class TestOpticalCT(unittest.TestCase):

    def test_dataset_ctor(self):
        data = Dataset()
        assert data.projections is None
        assert data.dir is None

    def test_dataset_load_invalid_input(self):
        data = Dataset()
        with pytest.raises(ValueError):
            data.load('data/non-existentPath')

        with pytest.raises(ValueError):
            if not os.path.exists('/tmp/pytest_opticalct_emptyFolder'):
                os.mkdir('/tmp/pytest_opticalct_emptyFolder')
            data.load('/tmp/pytest_opticalct_emptyFolder')

    def test_dataset_load_lemon(self):
        data = Dataset()
        data.load('data/lemon')
        assert len(data.projections) == 199
        assert data.projections.shape == (199, 588, 878)

    def test_dataset_ctor_calls_load(self):
        data = Dataset()
        data = Dataset('data/small_matrix/projections', format='*.tif')
        assert len(data.projections) == 4

    def test_dataset_predefined_arguments(self):
        data = Dataset()
        dir = 'data/small_matrix/projections'
        data.format = '*.tif'
        data.load(dir)
        assert len(data.projections) == 4

    def test_dataset_load_scaled(self):
        data = Dataset()
        data.load('data/lemon', scale=0.5)
        assert data.projections.shape == (199, 294, 439)

    def test_dataset_load_small_matrix(self):
        data = Dataset()
        dataset_dir = 'data/small_matrix/projections'
        data.load(dataset_dir, format='*.tif')
        assert len(data.projections) == 4
        assert data.projections.shape == (4, 2, 5)
        assert data.projections.all() == np.array(
            [[[0, 48, 102, 101, 0], [0, 48, 102, 101, 0]], 
            [[64, 54, 75, 42, 16], [64, 54, 75, 42, 16]],
            [[0, 106, 94, 51, 0], [0, 106, 94, 51, 0]],
            [[0, 74, 130, 22, 25], [0, 74, 130, 22, 25]]], dtype='uint8').all()
        assert data.dir.is_dir() is True
        assert str(data.dir) == dataset_dir

    def test_dataset_load_small_matrix_invert(self):
        data = Dataset()
        dataset_dir = 'data/small_matrix/projections'
        data.load(dataset_dir, format='*.tif', invert=True)
        assert data.projections.all() == (255 - np.array(
            [[[0, 48, 102, 101, 0], [0, 48, 102, 101, 0]], 
            [[64, 54, 75, 42, 16], [64, 54, 75, 42, 16]],
            [[0, 106, 94, 51, 0], [0, 106, 94, 51, 0]],
            [[0, 74, 130, 22, 25], [0, 74, 130, 22, 25]]], dtype='uint8')).all()

    def test_dataset_get_files(self):
        data = Dataset()

        Path('/tmp/opticalct_standardNumbers').mkdir(parents=True, exist_ok=True)
        Path('/tmp/opticalct_leadingZeros').mkdir(parents=True, exist_ok=True)

        data.dir = Path('/tmp/opticalct_leadingZeros')
        for i in range(1, 101):
            Path(str(data.dir) + '/file' + str(i).zfill(3) + '.jpg').touch(exist_ok=True)
        data.get_files()
        for f in data.files:
            assert re.search('[0-9][0-9][0-9]', f.name) != None

        data.dir = Path('/tmp/opticalct_standardNumbers')
        for i in range(1, 101):
            Path(str(data.dir) + '/file' + str(i) + '.jpg').touch(exist_ok=True)
        data.get_files()
        assert data.files[1].name == 'file2.jpg'