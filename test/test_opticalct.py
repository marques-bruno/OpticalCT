import unittest
from opticalct.opticalct import Dataset
import numpy as np
import pytest
import os

class TestOpticalCT(unittest.TestCase):

    def test_dataset_load(self):
        data = Dataset()
        assert data.projections is None

        with pytest.raises(ValueError):
            data.load('data/non-existentPath')

        with pytest.raises(ValueError):
            if not os.path.exists('/tmp/pytest_opticalct_emptyFolder'):
                os.mkdir('/tmp/pytest_opticalct_emptyFolder')
            data.load('/tmp/pytest_opticalct_emptyFolder')

        data.load('data/small_matrix/projections')
        assert len(data.projections) == 4
        assert data.projections.shape == (4, 2, 5)
        assert data.projections.all() == np.array(
            [[[0, 48, 102, 101, 0], [0, 48, 102, 101, 0]], 
            [[64, 54, 75, 42, 16], [64, 54, 75, 42, 16]],
            [[0, 106, 94, 51, 0], [0, 106, 94, 51, 0]],
            [[0, 74, 130, 22, 25], [0, 74, 130, 22, 25]]], dtype='uint8').all()

