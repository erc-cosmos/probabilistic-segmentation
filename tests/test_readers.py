"""Tests for the readers."""
from bayes_arcs import readers
import numpy as np


def test_mazurka_collection_size():
    """Check the size of the mazurka collection."""
    data = list(readers.read_all_mazurka_data())

    assert len(data) == 46


def test_mazurka_collection_tempo_size():
    """Check the size of the mazurka collection for tempo."""
    data = list(readers.read_all_mazurka_timings())

    assert len(data) == 46


def test_mazurka_read_interpret_count():
    """Check the number of interpretations for a piece."""
    data = readers.read_mazurka_data('data/beat_dyn/M06-1beat_dynNORM.csv')

    assert len(data) == 34


def test_mazurka_read_returns_arrays():
    """Check that the data is in numpy arrays."""
    data = readers.read_mazurka_data('data/beat_dyn/M06-1beat_dynNORM.csv')

    assert all(isinstance(datum, np.ndarray) for _, datum in data)


def test_mazurka_tempo_is_plausible():
    """Check that tempo values are plausible."""
    data = readers.read_mazurka_timings('data/beat_time/M06-1beat_time.csv')

    for _, (times, tempos) in data:
        for t in tempos:
            assert 10 < t < 600


def test_mazurka_seg_and_tempo_size():
    """Check that readAllMazurkaTimingsAndSeg returns the right number of series."""
    data = readers.read_all_mazurka_timings_and_seg()

    assert len(data) == 37


def test_mazurka_seg_and_loudness_size():
    """Check that readAllMazurkaDataAndSeg returns the right number of series."""
    data = readers.read_all_mazurka_data_and_seg()

    assert len(data) == 37
