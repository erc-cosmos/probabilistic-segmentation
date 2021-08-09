"""
Module for reading performance data or collections.

Data can be either raw or annotated.
Collections are either in the cosmonote format or MazurkaBL format.
"""

import csv
import numpy as np
import os
from collections import namedtuple
from scipy.interpolate import UnivariateSpline

CosmonoteLoudness = namedtuple("Loudness", ['time', 'loudness'])
CosmonoteAnnotation = namedtuple("Annotation", ['author', 'boundaries'])
CosmonoteAnnotationSet = namedtuple("AnnotationSet", ['audio', 'loudness', 'tempo'])
CosmonoteData = namedtuple("CosmonotePiece", ['piece_id', 'beats', 'tempo', 'loudness', 'annotations'])


def readMazurkaData(filename, preprocess=None):
    """Read a MazurkaBL-formatted performance file with optional preprocessing."""
    with open(filename) as csvFile:
        csvReader = csv.reader(csvFile)
        # Read header
        interpretIds = next(csvReader)[3:]  # First 3 columns are not relevant to us
        # zip to read colum by column
        data = zip(*(map(float, row[3:]) for row in csvReader))
        if preprocess is not None:
            data = preprocess(data)
        else:
            data = map(np.array, data)
        return list(zip(interpretIds, data))


def preprocessTimings(timings):
    """Map timings to tempo."""
    tempo = map(lambda time: 60/np.diff(time), timings)
    return tempo


def readMazurkaTimings(filename):
    """Read a tempo in MazurkaBL format."""
    return readMazurkaData(filename, preprocess=preprocessTimings)


def readAllMazurkaTimings(dirpath="data/beat_time"):
    """Read all tempos in a directory in MazurkaBL format."""
    # Retrieve all mazurka files
    files = [os.path.join(dirpath, file) for file in os.listdir(dirpath) if os.path.splitext(file)[1] == ".csv"]
    # Read and return them
    return zip(files, map(readMazurkaTimings, files))


def readAllMazurkaData(dirpath="data/beat_dyn", preprocess=None):
    """Read all of a directory in MazurkaBL format, with optional preprocessing."""
    # Retrieve all mazurka files
    files = [os.path.join(dirpath, file) for file in os.listdir(dirpath) if os.path.splitext(file)[1] == ".csv"]
    # Read and return them
    return list(zip(files, map(lambda f: readMazurkaData(f, preprocess=preprocess), files)))


def readMazurkaArcSegmentation(filename):
    """Read a segmentation in MazurkaBL format."""
    with open(filename) as csvFile:
        csvReader = csv.reader(csvFile)
        seg = [(line[0], [int(number) for number in line[1:] if number != '']) for line in csvReader]
    return seg


def matchMazurkaSegmentation(filename, dirpath="deaf_structure_tempo", dataType='tempo'):
    """Find the corresponding segmentation and read it."""
    if dataType == 'tempo':
        segbasename = os.path.basename(filename.replace("beat_time", "_man_seg"))
    elif dataType == 'loudness':
        segbasename = os.path.basename(filename.replace("beat_dynNORM", "_man_seg"))
    else:
        raise NotImplementedError("Unknown dataType: "+dataType)
    segfile = os.path.join(dirpath, segbasename)
    if os.path.isfile(segfile):
        return readMazurkaArcSegmentation(segfile)
    else:
        return []


def readAllMazurkaTimingsAndSeg(timingPath="data/beat_time", segPath="data/deaf_structure_tempo"):
    allData = []
    allTimings = readAllMazurkaTimings(timingPath)
    for filename, timings in allTimings:
        segmentations = matchMazurkaSegmentation(filename, segPath)
        for pID, seg in segmentations:
            tim = next((times for pIDmatch, times in timings if pID == pIDmatch), None)
            if tim is None:
                print("Warning: Encountered performer without a match in timings: "+pID+" in "+filename)
            else:
                allData.append((filename, pID, tim, seg))
    return allData


def readAllMazurkaDataAndSeg(timingPath="data/beat_dyn", segPath="data/deaf_structure_loudness",
                             preprocess=None, dataType='loudness'):
    allPerf = []
    allData = readAllMazurkaData(timingPath, preprocess=preprocess)
    for filename, timings in allData:
        segmentations = matchMazurkaSegmentation(filename, segPath, dataType=dataType)
        for pID, seg in segmentations:
            tim = next((times for pIDmatch, times in timings if pID == pIDmatch), None)
            if tim is None:
                print("Warning: Encountered performer without a match in timings: "+pID+" in "+filename)
            else:
                allPerf.append((filename, pID, tim, seg))
    return allPerf


def read_cosmo_beats(filepath):
    with open(filepath) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        beats = [float(e['time']) for e in csv_reader]
        return beats


def read_cosmo_loudness(filepath):
    with open(filepath) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        loud = [CosmonoteLoudness(float(e['Time']), float(e['Loudness_smooth'])) for e in csv_reader]
        return loud


def read_cosmo_annotation(filepath, strengths=(2, 3, 4)):
    with open(filepath) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        annot = [float(e['Time']) for e in csv_reader if int(e['Strength']) in strengths]
        return annot


def preprocess_cosmo_loudness(loudness, beats):
    x, y = zip(*loudness)
    # smoothing = len(loudness)
    interpol = UnivariateSpline(x, y, s=0)
    smoothed = interpol(beats)

    # smoothed = []
    # for curr_beat, next_beat in zip([0,*beats], beats):
    #     loud_interval = [y for x,y in loudness if curr_beat < x < next_beat]
    #     smoothed.append(np.mean(loud_interval))

    return smoothed
    # return (x,y)


def preprocess_cosmo_annotation(annotation, beats):
    # Optimizable
    no_doubles = np.unique([(np.abs(np.array(beats) - boundary)).argmin() for boundary in annotation])
    # Ignore boundaries within 3 beats of the first or last beat
    no_extreme = [bound for bound in no_doubles if 3 <= bound < len(beats)-3]
    return no_extreme


def read_all_cosmo_data(source_path="data/Chopin Collection/"):
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


def read_all_cosmonote_loudness(source_path="data/Chopin Collection/"):
    pass


def read_all_cosmo_annotation(source_path="data/Chopin Collection/"):
    pass


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
