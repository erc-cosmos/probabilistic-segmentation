"""Functions to write various types of data to disk."""
import csv
import os
from typing import Iterable, Mapping, Tuple, Union


def writeMarginals(filename: Union[os.PathLike, str], marginals: Iterable[float]) -> None:
    """Write a set of marginals to disk as csv."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(["Beat", "Credence of final"])
        writer.writerows(enumerate(marginals))


def writeMeasures(filename: Union[os.PathLike, str], measures: Mapping[str, Tuple[float, float, float, float]]) -> None:
    """Write a set of segmentation measures to disk."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(["feature", "piece", "performer", "",
                        "Quadratic/Brier score", "F1-score", "Recall", "Precision"])
        rows = [[*key.split('/'), *scores]
                for key, scores in measures.items()]
        writer.writerows(rows)
