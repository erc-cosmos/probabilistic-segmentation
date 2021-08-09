"""Tests for the readers."""


from readers import readAllMazurkaData, readMazurkaData
import numpy as np


def test_mazurka_collection_size():
    """Check the size of the mazurka collection."""
    data = list(readAllMazurkaData())

    assert len(data) == 46


def test_mazurka_read_interpret_count():
    """Check the number of interpretations for a piece."""
    data = readMazurkaData('data/beat_dyn/M06-1beat_dynNORM.csv')

    assert len(data) == 34


def test_mazurka_read_returns_arrays():
    """Check that the data is in numpy arrays."""
    data = readMazurkaData('data/beat_dyn/M06-1beat_dynNORM.csv')

    assert all(isinstance(datum, np.ndarray) for _, datum in data)
