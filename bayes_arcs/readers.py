"""
Module for reading performance data or collections.

Data can be either raw or annotated.
Collections are either in the cosmonote format or MazurkaBL format.
"""

from collections import namedtuple
import csv
import itertools as itt
import os
from typing import Any, List, NamedTuple, Sequence, Union

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

CosmonoteLoudness = namedtuple("Loudness", ['time', 'loudness'])
CosmonoteAnnotation = namedtuple("Annotation", ['author', 'boundaries'])
CosmonoteAnnotationSet = namedtuple("AnnotationSet", ['audio', 'loudness', 'tempo'])
CosmonoteData = namedtuple("CosmonotePiece", ['piece_id', 'beats', 'tempo', 'loudness', 'annotations'])


class CosmoPerf(NamedTuple):
    """Named tuple to hold a performance in cosmonote format."""

    perf_id: str
    perf_data: Any  # Union[pd.DataFrame, np.ndarray]


class CosmoPiece(NamedTuple):
    """Named tuple to hold a piece (with many interpretations) in cosmonote format."""

    piece_id: str
    piece_data: Sequence[CosmoPerf]


def read_mazurka_data(filename, preprocess=None):
    """Read a MazurkaBL-formatted performance file with optional preprocessing."""
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file)
        # Read header
        interpret_ids = next(csv_reader)[3:]  # First 3 columns are not relevant to us
        # zip to read colum by column
        data = list(zip(*(map(float, row[3:]) for row in csv_reader)))
        if preprocess is not None:
            data = zip(data, preprocess(data))
        else:
            data = map(np.array, data)
        return [CosmoPerf(perf_id, perf_data) for perf_id, perf_data in zip(interpret_ids, data)]


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
    return list(map(CosmoPiece, files, map(lambda f: read_mazurka_data(f, preprocess=preprocess), files)))


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


def load_mazurka_dataset_with_annot(timing_path: str = "data/beat_time", dyn_path: str = "data/beat_dyn",
                                    seg_time_path: str = "data/deaf_structure_tempo",
                                    seg_dyn_path: str = "data/deaf_structure_loudness") -> pd.DataFrame:
    """Load all data for the MazurkaBL dataset with structural annotations.

    Args:
        timing_path (str, optional): path to the timing data. Defaults to "data/beat_time".
        dyn_path (str, optional): path to the dynamics data. Defaults to "data/beat_dyn".
        seg_time_path (str, optional): path to timing-based segmentations. Defaults to "data/deaf_structure_tempo".
        seg_dyn_path (str, optional): path to dynamics-based segmentations. Defaults to "data/deaf_structure_loudness".

    Returns:
        pd.Dataframe: Dataframe containing for each performance, in that order:
            - the piece identifier
            - the interpret identifier
            - the tempo data
            - the tempo segmentation
            - the dynamics data
            - the dynamics segmentation
    """
    timings_data = read_all_mazurka_timings_and_seg(timing_path, seg_time_path)
    dyn_data = read_all_mazurka_data_and_seg(dyn_path, seg_dyn_path)
    return pd.DataFrame([(piece, interpret, tempo, tempo_seg, dyn, dyn_seg)
                         for ((piece, interpret, tempo, tempo_seg), (_piece2, interpret2, dyn, dyn_seg))
                         in itt.product(timings_data, dyn_data)
                         if interpret == interpret2],
                        columns=("file", "performer id", "tempo", "tempo segmentation",
                                 "loudness", "loudness segmentation"))


def load_mazurka_dataset(timing_path="data/beat_time", dyn_path="data/beat_dyn"):
    """Load all data for the MazurkaBL dataset."""
    pass


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


def tempo_outlier_correction_with_timings(timings: list[float], *,
                                          threshold: float = 2, max_passes: int = 3) -> np.ndarray:
    """Map timings to tempo, correcting for aberrant values.

    Args:
        see `:func:preprocess_tempo_outlier_correction`
    Returns:
        [np.ndarray]: Array of the corrected instantaneous tempos with their timings
    """
    timings = timings.copy()

    def locate_errors(timings, threshold: float = 1.5, frame_length: int = 10):
        def trailing_mean(timings, frame_length):
            mean_iois = [(timings[i]-timings[0])/(i) for i in range(1, frame_length)]
            mean_iois.extend(np.convolve(np.diff(timings), np.ones(frame_length), mode="valid")/frame_length)
            return mean_iois[:-1]

        def leading_mean(timings, frame_length):
            mean_iois = list(np.convolve(np.diff(timings), np.ones(frame_length), mode="valid")/frame_length)
            mean_iois.extend([(timings[-1]-timings[-i-1])/(i) for i in reversed(range(1, frame_length))])
            return mean_iois[1:]

        for j, (ioi, lead, trail) in enumerate(zip(np.diff(timings),
                                                   [*leading_mean(timings, frame_length=frame_length), np.nan],
                                                   [np.nan, *trailing_mean(timings, frame_length=frame_length)])):
            if ioi < min(lead, trail)/threshold:
                # A beat is abnormal if it's much faster than the leading or following beats
                yield j

    for _ in range(max_passes):
        changed = False
        for j in locate_errors(timings, threshold=threshold, frame_length=10):
            print(f"Aberrant value at index {j}")
            # Basic fix with linear interpolation
            if j == 0:
                timings[j+1] = (timings[j]+timings[j+2])/2
            elif j == len(timings)-2:
                timings[j] = (timings[j-1]+timings[j+1])/2
            else:
                interp1 = timings[j-1]
                interp2 = timings[j+2]
                timings[j] = (2*interp1+interp2)/3
                timings[j+1] = (2*interp2+interp1)/3
            changed = True
        if not changed:
            break
    return np.array([timings[1:], 60/np.diff(timings)])


def preprocess_tempo_outlier_correction(timings: list[float], *,
                                        return_timings: bool = False,
                                        threshold: float = 2, max_passes: int = 3) -> np.ndarray:
    """Map timings to tempo, correcting for aberrant values.

    Args:
        timings ([list[float]]): Timings sequence for a single piece
        threshold (float, optional): Maximum deviation factor from the average before correcting. Defaults to 3.
        max_passes (int, optional): Maximum iterations of correction. Defaults to 3.

    Returns:
        [np.ndarray]: Array of the corrected instantaneous tempos
    """
    _, tempo = tempo_outlier_correction_with_timings(timings, threshold=threshold, max_passes=max_passes)
    return tempo


def infer_data_type_from_filenames(filenames: list[str]):
    """Infer the data type from a set of file names.

    Args:
        filenames (list[str]): The file names to consider
    Returns:
        'tempo'|'beats'|'loudness'|'mixed'|'auto': The detected data type(s) or 'auto' if unsuccessful
    """
    # TODO
    return 'auto'


def read_cosmo_piece(piece_folder, data_type='auto', include_average=False):
    """Read a Cosmonote formatted set of performances.

    Args:
        piece_folder (str): Path to the folder containing the performances's data
        type ('tempo'|'beats'|'loudness'|'mixed'|'auto'): [NYI] Type of data to process. Defaults to 'auto'.
    """
    piece_data: List[CosmoPerf] = []
    perfs = sorted([f for f in os.listdir(piece_folder) if f.endswith('.csv')])
    for perf in perfs:
        data_path = os.path.join(piece_folder, perf)
        perf_id, _ = os.path.splitext(perf)
        perf_data = read_cosmo_perf(data_path, data_type)
        piece_data.append(CosmoPerf(perf_id, perf_data))
    if include_average:
        if data_type == "mixed":
            raise NotImplementedError("Average is not yet compatible with Mixed data")
        _, raw_data = zip(*piece_data)
        piece_data.append(CosmoPerf('Average', np.mean(raw_data, axis=0)))
    return piece_data


def read_cosmo_perf(data_path: str, data_type: str = 'mixed') -> Union[np.ndarray, pd.DataFrame]:
    """Read a performance from a cosmonote-formatted file.

    Args:
        data_path (str): path to the file to read
        data_type (str, optional): which column to access. Defaults to 'mixed'.

    Raises:
        NotImplementedError: If the data type is unknown

    Returns:
        Union[np.ndarray, pd.DataFrame]: The requested performance feature, or everything if data_type is mixed
    """
    data = pd.read_csv(data_path, index_col='count')
    if data_type == 'mixed':
        return data
    elif data_type == 'tempo':
        return np.array(data['tempo'])
    elif data_type == 'beats':
        return np.array(data['time'])
    else:
        raise NotImplementedError("Unsupported data type")


def read_cosmo_collection(main_folder, data_type='auto', include_average=False):
    """Read a Cosmonote formatted set of performances, grouped by piece.

    Args:
        main_folder ([type]): Path to the folder containing the collection's data
        type ('tempo'|'beats'|'loudness'|'mixed'|'auto', optional): [NYI] Type of data to process. Defaults to 'auto'.
    """
    pieces = sorted([f for f in os.listdir(main_folder)
                     if os.path.isdir(os.path.join(main_folder, f))])
    print(pieces)
    full_data = []
    for piece in pieces:
        piece_folder = os.path.join(main_folder, piece)
        piece_data = read_cosmo_piece(piece_folder, data_type=data_type, include_average=include_average)
        full_data.append((piece, piece_data))
    return full_data


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


def read_posterior(filepath: str):
    """Read a saved sequence of posteriors.

    Args:
        filepath (str): The path where the data was saved
    """
    df = pd.read_csv(filepath, names=["count", "posterior"], index_col="count", header=0)
    return df["posterior"]


def join_collections(collection1: Sequence[CosmoPiece], collection2: Sequence[CosmoPiece]) -> Sequence[CosmoPiece]:
    """Join data from two different features at the collection level.

    Args:
        collection1 (Sequence[CosmoPiece]): First feature data for the collection
        collection2 (Sequence[CosmoPiece]): Second feature for the collection

    Returns:
        Sequence[CosmoPiece]: Joined data
    """
    collection1 = sorted(collection1)
    collection2 = sorted(collection2)
    return [CosmoPiece(piece_name, join_pieces(piece_from_1, piece_from_2))
            for (piece_name, piece_from_1), (_piece_name, piece_from_2) in zip(collection1, collection2)]


def join_pieces(piece1: Sequence[CosmoPerf], piece2: Sequence[CosmoPerf]) -> Sequence[CosmoPerf]:
    """Join data from two different features at the piece level.

    Args:
        piece1 (Sequence[CosmoPerf]): First feature data for the piece
        piece2 (Sequence[CosmoPerf]): Second feature for the piece

    Returns:
        Sequence[CosmoPerf]: Joined data
    """
    return [CosmoPerf(pid1, list(zip(perf1, perf2)))
            for (pid1, perf1), (_pid2, perf2) in zip(sorted(piece1), sorted(piece2))]
