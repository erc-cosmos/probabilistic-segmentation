""" Functions to write various types of data to disk """
import csv
import os


def writeMarginals(filename, marginals):
    """ Writes a set of marginals to disk as csv """
    os.makedirs(outputDir, exist_ok=True)
    with open(filename, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(["Beat", "Credence of final"])
        writer.writerows(enumerate(marginals))


def writeMeasures(filename, measures):
    """ Writes a set of segmentation measures to disk """
    if not os.path.isdir(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    with open(filename, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(["feature", "piece", "performer", "",
                        "Quadratic/Brier score", "F1-score", "Recall", "Precision"])
        rows = [list(key.split('/')) + list(scores)
                for key, scores in measures.items()]
        writer.writerows(rows)
