import unittest
from opticalct.opticalct import Dataset
import numpy as np
import pytest
import os

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
        assert len(data.projections) == 200
        assert data.projections.shape == (200, 588, 878)

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
        assert data.projections.shape == (200, 294, 439)

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

    def test_dataset_load_check_frame_order(self):
        data = Dataset()
        data.load('data/lemon')