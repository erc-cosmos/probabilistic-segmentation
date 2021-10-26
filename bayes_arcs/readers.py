"""
Module for reading performance data or collections.

Data can be either raw or annotated.
Collections are either in the cosmonote format or MazurkaBL format.
"""

from collections import namedtuple
import csv
import os

import numpy as np
from scipy.interpolate import UnivariateSpline

CosmonoteLoudness = namedtuple("Loudness", ['time', 'loudness'])
CosmonoteAnnotation = namedtuple("Annotation", ['author', 'boundaries'])
CosmonoteAnnotationSet = namedtuple("AnnotationSet", ['audio', 'loudness', 'tempo'])
CosmonoteData = namedtuple("CosmonotePiece", ['piece_id', 'beats', 'tempo', 'loudness', 'annotations'])


def read_mazurka_data(filename, preprocess=None):
    """Read a MazurkaBL-formatted performance file with optional preprocessing."""
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file)
        # Read header
        interpret_ids = next(csv_reader)[3:]  # First 3 columns are not relevant to us
        # zip to read colum by column
        data = zip(*(map(float, row[3:]) for row in csv_reader))
        if preprocess is not None:
            data = zip(data, preprocess(data))
        else:
            data = map(np.array, data)
        return list(zip(interpret_ids, data))


def preprocess_timings(timings):
    """Map timings to tempo."""
    tempo = map(lambda time: 60/np.diff(time), timings)
    return tempo


def read_mazurka_timings(filename):
    """Read a tempo in MazurkaBL format."""
    return read_mazurka_data(filename, preprocess=preprocess_timings)


def read_all_mazurka_timings(dirpath="data/beat_time"):
    """Read all tempos in a directory in MazurkaBL format."""
    # Retrieve all mazurka files
    files = [os.path.join(dirpath, file) for file in os.listdir(dirpath) if os.path.splitext(file)[1] == ".csv"]
    # Read and return them
    return zip(files, map(read_mazurka_timings, files))


def read_all_mazurka_data(dirpath="data/beat_dyn", preprocess=None):
    """Read all of a directory in MazurkaBL format, with optional preprocessing."""
    # Retrieve all mazurka files
    files = [os.path.join(dirpath, file) for file in os.listdir(dirpath) if os.path.splitext(file)[1] == ".csv"]
    # Read and return them
    return list(zip(files, map(lambda f: read_mazurka_data(f, preprocess=preprocess), files)))


def read_mazurka_arc_segmentation(filename):
    """Read a segmentation in MazurkaBL format."""
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file)
        seg = [(line[0], [int(number) for number in line[1:] if number != '']) for line in csv_reader]
    return seg


def match_mazurka_segmentation(filename, dirpath="deaf_structure_tempo", data_type='tempo'):
    """Find the corresponding segmentation and read it."""
    if data_type == 'tempo':
        segbasename = os.path.basename(filename.replace("beat_time", "_man_seg"))
    elif data_type == 'loudness':
        segbasename = os.path.basename(filename.replace("beat_dynNORM", "_man_seg"))
    else:
        raise NotImplementedError("Unknown dataType: "+data_type)
    segfile = os.path.join(dirpath, segbasename)
    if os.path.isfile(segfile):
        return read_mazurka_arc_segmentation(segfile)
    else:
        return []


def read_all_mazurka_timings_and_seg(timing_path="data/beat_time", seg_path="data/deaf_structure_tempo"):
    """Read tempo segmentations and match them with tempo data."""
    all_data = []
    all_timings = read_all_mazurka_timings(timing_path)
    for filename, timings in all_timings:
        segmentations = match_mazurka_segmentation(filename, seg_path)
        for pid, seg in segmentations:
            tim = next((times for pid_match, times in timings if pid == pid_match), None)
            if tim is None:
                print("Warning: Encountered performer without a match in timings: "+pid+" in "+filename)
            else:
                all_data.append((filename, pid, tim, seg))
    return all_data


def read_all_mazurka_data_and_seg(timing_path="data/beat_dyn", seg_path="data/deaf_structure_loudness",
                                  preprocess=None, data_type='loudness'):
    """Read arbitrary segmentations and match them with arbitrary data."""
    all_perf = []
    all_data = read_all_mazurka_data(timing_path, preprocess=preprocess)
    for filename, timings in all_data:
        segmentations = match_mazurka_segmentation(filename, seg_path, data_type=data_type)
        for pid, seg in segmentations:
            tim = next((times for pid_match, times in timings if pid == pid_match), None)
            if tim is None:
                print("Warning: Encountered performer without a match in timings: "+pid+" in "+filename)
            else:
                all_perf.append((filename, pid, tim, seg))
    return all_perf


def read_cosmo_beats(filepath):
    """Read a beats file in Cosmonote format."""
    with open(filepath) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        beats = [float(e['time']) for e in csv_reader]
        return beats


def read_cosmo_loudness(filepath):
    """Read a loudness file in Cosmonote format."""
    with open(filepath) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        loud = [CosmonoteLoudness(float(e['Time']), float(e['Loudness_smooth'])) for e in csv_reader]
        return loud


def read_cosmo_annotation(filepath, strengths=(2, 3, 4)):
    """Read an annotation file in Cosmonote format, filtering to specified boundary strengths."""
    with open(filepath) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        annot = [float(e['Time']) for e in csv_reader if int(e['Strength']) in strengths]
        return annot


def preprocess_cosmo_loudness(loudness, beats):
    """Preprocess loudness by interpolating at beats positions."""
    x, y = zip(*loudness)
    interpol = UnivariateSpline(x, y, s=0)  # s=0 means no smoothing
    smoothed = interpol(beats)

    return smoothed


def preprocess_cosmo_annotation(annotation, beats):
    """Snap annotations to the closest beat and remove duplicate and implicit boundaries."""
    # Optimizable
    no_doubles = np.unique([(np.abs(np.array(beats) - boundary)).argmin() for boundary in annotation])
    # Ignore boundaries within 3 beats of the first or last beat
    no_extreme = [bound for bound in no_doubles if 3 <= bound < len(beats)-3]
    return no_extreme


def read_all_cosmo_data(source_path="data/Chopin Collection/"):
    """Collect data in Cosmonote format and assemble it in a single structure."""
    all_data = []
    for beat_base in os.listdir(os.path.join(source_path, "Beats_pruned")):
        if "excerpt_" not in beat_base:
            continue
        basefile = beat_base[:-10]
        piece_id = basefile

        beat_file = os.path.join(source_path, "Beats", f"{basefile}_beats.csv")
        beats = read_cosmo_beats(beat_file)
        tempo = [np.nan, *(60/np.diff(beats))]

        loudness_file = os.path.join(source_path, "Loudness", f"{basefile}_loudness.csv")
        loudness_raw = read_cosmo_loudness(loudness_file)
        loudness = preprocess_cosmo_loudness(loudness_raw, beats)

        annots_audio = []
        for annot_audio_base in os.listdir(os.path.join(source_path, "Annotations_tmp", "Audio")):
            if basefile+'-' in annot_audio_base:
                annot_audio_file = os.path.join(
                    source_path, "Annotations_tmp", "Audio", annot_audio_base)
                annotator = annot_audio_base.replace('-', '.').split('.')[1]
                annot_audio_raw = read_cosmo_annotation(annot_audio_file)
                annots_audio.append(CosmonoteAnnotation(annotator,
                                                        preprocess_cosmo_annotation(annot_audio_raw, beats)))

        annots_loudness = []
        for annot_loudness_base in os.listdir(os.path.join(source_path, "Annotations_tmp", "Loudness")):
            if basefile+'-' in annot_loudness_base:
                annot_loudness_file = os.path.join(
                    source_path, "Annotations_tmp", "Loudness", annot_loudness_base)
                annotator = annot_loudness_base.replace('-', '.').split('.')[1]
                annot_loudness_raw = read_cosmo_annotation(annot_loudness_file)
                annots_loudness.append(CosmonoteAnnotation(annotator,
                                                           preprocess_cosmo_annotation(annot_loudness_raw, beats)))

        annots_tempo = []
        for annot_tempo_base in os.listdir(os.path.join(source_path, "Annotations_tmp", "Tempo")):
            if basefile+'-' in annot_tempo_base:
                annot_tempo_file = os.path.join(source_path, "Annotations_tmp", "Tempo", annot_tempo_base)
                annotator = annot_tempo_base.replace('-', '.').split('.')[1]
                annotation_tempo_raw = read_cosmo_annotation(annot_tempo_file)
                annots_tempo.append(CosmonoteAnnotation(annotator,
                                                        preprocess_cosmo_annotation(annotation_tempo_raw, beats)))

        annot_set = CosmonoteAnnotationSet(annots_audio, annots_loudness, annots_tempo)
        data = CosmonoteData(piece_id, beats, tempo, loudness, annot_set)
        all_data.append(data)
    return all_data


if __name__ == "__main__":
    # filename = "M06-2beat_time.csv"
    # data = readMazurkaTimings(filename)
    # for d in data:
    #     print(d)

    # filename = "M06-2_seg_man.csv"
    # data = readMazurkaArcSegmentation(filename)
    # for d in data:
    #     print(d)
    # data = readAllMazurkaDataAndSeg()
    # for maz, pid, tim, seg in data:
    #     print(maz, pid, tim, seg)
    # a = readAllMultidim(["beat_dyn"], [None], ["Dyn"])
    # f1 = '/Users/guichaoua 1/Nextcloud/Workspace/ArcV2/data/Chopin Collection/Annotations_tmp/Audio/excerpt_2.csv'
    # print("Annots", read_cosmo_annotation(f1))
    # f2 = '/Users/guichaoua 1/Nextcloud/Workspace/ArcV2/data/Chopin Collection/Beats/excerpt_2_beats.csv'
    # print("Beats", read_cosmo_beats(f2))
    # f3 = '/Users/guichaoua 1/Nextcloud/Workspace/ArcV2/data/Chopin Collection/Loudness/excerpt_2_loudness.csv'
    # print("Loudness", read_cosmo_loudness(f3))

    # beats = read_cosmo_beats(f2)
    # raw_loud = read_cosmo_loudness(f3)
    # loud = preprocess_cosmo_loudness(raw_loud, beats)

    # import matplotlib.pyplot as plt
    # plt.plot(*zip(*raw_loud))
    # x, y = zip(*raw_loud)
    # interpol = UnivariateSpline(x, y, s=10000)
    # xsample = np.linspace(beats[0], beats[-1], 10000)
    # ysample = interpol(xsample)
    # plt.plot(xsample, ysample)
    # plt.plot(beats, loud)

    # raw_annot = read_cosmo_annotation(f1)
    # annot = preprocess_cosmo_annotation(raw_annot, beats)

    # bound_clocktime = [beats[idx] for idx in annot]
    # plt.vlines(bound_clocktime, ymin=np.min(loud), ymax=np.max(loud), colors="r")
    # plt.vlines(raw_annot, ymin=np.min(loud), ymax=np.max(loud), colors="b")
    # plt.vlines(beats, ymin=np.min(loud), ymax=0.5*np.max(loud), colors="k")
    # plt.show()
    # print(loud)

    data = read_all_cosmo_data()

    print(data)

    print("Done")
