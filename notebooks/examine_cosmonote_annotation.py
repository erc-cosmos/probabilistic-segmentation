# %%
import csv
import itertools as itt

from bayes_arcs import default_priors
from bayes_arcs import dynamic_computation as dc
import length_priors
import matplotlib.pyplot as plt
import numpy as np
import readers
import scoring


# %%
cosmo_data = readers.read_all_cosmo_data()


# %%
cosmo_data[1].annotations


# %%
# Look at loudness vs itself and audio

def scoretime_to_clocktime(annotation, beats):
    return [beats[annot] for annot in annotation]


for piece in cosmo_data:
    plt.figure()
    # x,y = piece.loudness
    # plt.plot(x,y)
    plt.plot(piece.beats, piece.loudness)
    colors = ["k", "g", "b", "r", "c", "m"]
    ymin = np.nanmin(piece.loudness)
    ymax = np.nanmax(piece.loudness)
    for annotator, annotation in piece.annotations.loudness:
        yymin = ymin + (ymax-ymin) * (int(annotator)-1)/6
        yymax = ymin + (ymax-ymin) * (int(annotator))/6
        plt.vlines(scoretime_to_clocktime(annotation, piece.beats),
                   ymin=yymin, ymax=yymax, colors=colors[int(annotator)-1])
    for annotator, annotation in piece.annotations.audio:
        plt.vlines(scoretime_to_clocktime(annotation, piece.beats), ymin=ymin,
                   ymax=ymax, colors=colors[int(annotator)-1], linestyles='dotted')
    print(piece.piece_id)
    plt.title(f"Loudness curve and annotations—{piece.piece_id.replace('_',' ')}")
    plt.xlabel("Clock time")
    plt.ylabel("Loudness")
    plt.savefig(f"output/loudness/{piece.piece_id}.pdf")
    plt.show()


# %%


# %%
scores = [[[] for i in range(6)] for j in range(6)]

for piece in cosmo_data:
    # print([(piece.piece_id, author) for author,_ in piece.annotations.loudness])
    for (author1, annot1), (author2, annot2) in itt.permutations(piece.annotations.loudness, 2):
        # print(author1,author2)
        scores[int(author1)-1][int(author2)-1].append(scoring.f_measure(annot1, annot2, tolerance=3))


# %%
counts = np.array([[len(x) for x in y] for y in scores])
farray = np.array([[np.mean(x) for x in y] for y in scores])


# %%
print(counts)
print(farray)


# %%
scores_loud_audio = [([], []) for j in range(6)]

for piece in cosmo_data:
    # print([(piece.piece_id, author) for author,_ in piece.annotations.loudness])
    author_audio, annot_audio = piece.annotations.audio[0]
    for author1, annot1 in piece.annotations.loudness:
        # print(author1,author2)
        fmeasure = scoring.f_measure(annot1, annot_audio, tolerance=3)
        scores_loud_audio[int(author1)-1][0 if (author1 == author_audio) else 1].append(fmeasure)


# %%
for annotator, (self_fs, other_fs) in enumerate(scores_loud_audio):
    print(f"Annotator {annotator+1} scores:")
    print(f"\tself: {np.mean(self_fs)}")
    print(f"\tother: {np.mean(other_fs)}")
    print(f"\ttotal: {np.mean(self_fs+other_fs)}")


# %%
# Look at tempo vs itself and audio

for piece in cosmo_data:
    plt.figure()
    # x,y = piece.loudness
    # plt.plot(x,y)
    plt.plot(piece.beats, piece.tempo)
    colors = ["k", "g", "b", "r", "c", "m"]
    ymin = np.nanmin(piece.tempo)
    ymax = np.nanmax(piece.tempo)
    for annotator, annotation in piece.annotations.tempo:
        yymin = ymin + (ymax-ymin) * (int(annotator)-1)/6
        yymax = ymin + (ymax-ymin) * (int(annotator))/6
        plt.vlines(scoretime_to_clocktime(annotation, piece.beats),
                   ymin=yymin, ymax=yymax, colors=colors[int(annotator)-1])
    for annotator, annotation in piece.annotations.audio:
        plt.vlines(scoretime_to_clocktime(annotation, piece.beats), ymin=ymin,
                   ymax=ymax, colors=colors[int(annotator)-1], linestyles='dotted')
    print(piece.piece_id)
    plt.title(f"Tempo curve and annotations—{piece.piece_id.replace('_',' ')}")
    plt.xlabel("Clock time")
    plt.ylabel("Tempo")
    plt.savefig(f"output/tempo/{piece.piece_id}.pdf")
    plt.show()


# %%
scores = [[[] for i in range(6)] for j in range(6)]

for piece in cosmo_data:
    # print([(piece.piece_id, author) for author,_ in piece.annotations.loudness])
    for (author1, annot1), (author2, annot2) in itt.permutations(piece.annotations.tempo, 2):
        # print(author1,author2)
        scores[int(author1)-1][int(author2)-1].append(scoring.f_measure(annot1, annot2, tolerance=3))

counts = np.array([[len(x) for x in y] for y in scores])
farray = np.array([[np.mean(x) for x in y] for y in scores])
print(counts)
print(farray)


# %%
scores_loud_audio = [([], []) for j in range(6)]

for piece in cosmo_data:
    # print([(piece.piece_id, author) for author,_ in piece.annotations.loudness])
    author_audio, annot_audio = piece.annotations.audio[0]
    for author1, annot1 in piece.annotations.tempo:
        # print(author1,author2)
        fmeasure = scoring.f_measure(annot1, annot_audio, tolerance=3)
        scores_loud_audio[int(author1)-1][0 if (author1 == author_audio) else 1].append(fmeasure)

for annotator, (self_fs, other_fs) in enumerate(scores_loud_audio):
    print(f"Annotator {annotator+1} scores:")
    print(f"\tself: {np.mean(self_fs)}")
    print(f"\tother: {np.mean(other_fs)}")
    print(f"\ttotal: {np.mean(self_fs+other_fs)}")


# %%

segmentations = {}
for piece in cosmo_data:
    arc_prior = default_priors.arc_prior_tempo
    length_prior_params = default_priors.length_prior_params_tempo
    length_prior = length_priors.NormalLengthPrior(length_prior_params['mean'],
                                                   length_prior_params['stddev']*2,
                                                   range(len(piece.tempo)),
                                                   length_prior_params['maxLength']*2)
    posterior_marginals = dc.compute_boundary_posteriors(piece.tempo[1:], arc_prior, length_prior)
    segmentations[piece.piece_id] = posterior_marginals

    fig, ax1 = plt.subplots()
    ax1.plot(piece.tempo, color="r")  # Tempo input data
    plt.ylim(0, 300)

    ax2 = ax1.twinx()

    ax2.plot(posterior_marginals, 'k')  # Posterior Marginals
    plt.ylim(0, 1)
    plt.vlines(piece.annotations.audio[0].boundaries, ymin=0, ymax=1, colors="r", linestyle='dotted')  # Tempo seg
    plt.vlines(piece.annotations.tempo[0].boundaries, ymin=0, ymax=1, colors="b", linestyle='dotted')  # Dyn seg
    plt.title(f"Tempo estimation vs annotation—{piece.piece_id.replace('_',' ')}")


# %%
f_tempo = []
f_audio = []
for piece in cosmo_data:
    estimation, _ = scoring.marginal2guess(segmentations[piece.piece_id], tolerance=3, threshold=.5)
    f_tempo.append([scoring.f_measure(estimation, annotation.boundaries, tolerance=3)
                   for annotation in piece.annotations.tempo])
    f_audio.append([scoring.f_measure(estimation, annotation.boundaries, tolerance=3)
                   for annotation in piece.annotations.audio])
    print(piece.piece_id, f"Tempo match: {f_tempo[-1]}", f"Audio match: {f_audio[-1]}")


# %%
[print(value) for value in f_tempo]
print("Mean: ", np.mean([np.mean(value) for value in f_tempo]))

print(np.array(f_audio))
print("Mean: ", np.mean(f_audio))


# %%
for piece in cosmo_data:
    posterior_marginals = [0, *segmentations[piece.piece_id]]
    estimation, _ = scoring.marginal2guess([np.nan, *posterior_marginals], tolerance=3, threshold=.5)
    fig, ax1 = plt.subplots()
    ax1.plot(piece.tempo, color="r")  # Tempo input data
    plt.ylim(0, 1.1*np.nanmax(piece.tempo))
    colors = ["k", "g", "b", "r", "c", "m"]
    ymin = 0
    ymax = np.nanmax(piece.tempo)*1.1
    for annotator, annotation in piece.annotations.tempo:
        yymin = ymin + (ymax-ymin) * (int(annotator)-1)/6
        yymax = ymin + (ymax-ymin) * (int(annotator))/6
        plt.vlines(annotation, ymin=yymin, ymax=yymax, colors=colors[int(annotator)-1])
    for annotator, annotation in piece.annotations.audio:
        plt.vlines(annotation, ymin=ymin, ymax=ymax, colors=colors[int(annotator)-1], linestyles='dotted')
    ax2 = ax1.twinx()

    ax2.plot(posterior_marginals, 'k')  # Posterior Marginals
    plt.ylim(0, 1)
    # plt.vlines(piece.annotations.audio[0].boundaries, ymin=0, ymax=1, colors="r", linestyle='dotted')  # Tempo seg
    # plt.vlines(piece.annotations.tempo[0].boundaries, ymin=0, ymax=1, colors="b", linestyle='dotted')  # Dyn seg
    plt.vlines(estimation, ymin=0, ymax=1, colors="k", linestyles='dashed')
    plt.savefig(f"output/boundaries_tempo/{piece.piece_id}.pdf")
    plt.savefig(f"output/boundaries_tempo/{piece.piece_id}.png")
    plt.show()


# %%
f_tempo = []
f_audio = []
for piece in cosmo_data:
    estimation, _ = scoring.marginal2guess(segmentations[piece.piece_id], tolerance=3, threshold=.5)
    f_tempo.append([scoring.f_measure(scoretime_to_clocktime(estimation, piece.beats), scoretime_to_clocktime(
        annotation.boundaries, piece.beats), tolerance=3) for annotation in piece.annotations.tempo])
    f_audio.append([scoring.f_measure(scoretime_to_clocktime(estimation, piece.beats), scoretime_to_clocktime(
        annotation.boundaries, piece.beats), tolerance=3) for annotation in piece.annotations.audio])
    print(piece.piece_id, f"Tempo match: {f_tempo[-1]}", f"Audio match: {f_audio[-1]}")


# %%
[print(value) for value in f_tempo]
print("Mean: ", np.mean([np.mean(value) for value in f_tempo]))

print(np.array(f_audio))
print("Mean: ", np.mean(f_audio))


# %%

with open("output/segmentations_2021_05_17.csv", 'w') as csv_file:
    writer = csv.writer(csv_file)
    for excerpt, seg in segmentations.items():
        writer.writerow([excerpt, *seg])


# %%

segmentations_loud = {}
for piece in cosmo_data:
    if piece.piece_id != "excerpt_75":
        continue

    arc_prior = default_priors.arc_prior_loud
    length_prior_params = default_priors.length_prior_params_loud
    length_prior = length_priors.NormalLengthPrior(length_prior_params['mean'],
                                                   length_prior_params['stddev']*2,
                                                   range(len(piece.loudness)),
                                                   length_prior_params['maxLength']*2)
    posterior_marginals = dc.compute_boundary_posteriors(piece.loudness, arc_prior, length_prior)
    segmentations_loud[piece.piece_id] = posterior_marginals


# %% [markdown]
#  # Visualise loudness segmentation

# %%
for piece in cosmo_data:
    posterior_marginals = segmentations_loud[piece.piece_id]
    estimation, _ = scoring.marginal2guess([np.nan, *posterior_marginals], tolerance=3, threshold=.5)
    fig, ax1 = plt.subplots()
    ax1.plot(piece.loudness, color="r")  # Tempo input data
    plt.ylim(0, 1.1*np.nanmax(piece.loudness))
    colors = ["k", "g", "b", "r", "c", "m"]
    ymin = 0
    ymax = np.nanmax(piece.loudness)*1.1
    for annotator, annotation in piece.annotations.loudness:
        yymin = ymin + (ymax-ymin) * (int(annotator)-1)/6
        yymax = ymin + (ymax-ymin) * (int(annotator))/6
        plt.vlines(annotation, ymin=yymin, ymax=yymax, colors=colors[int(annotator)-1])
    for annotator, annotation in piece.annotations.audio:
        plt.vlines(annotation, ymin=ymin, ymax=ymax, colors=colors[int(annotator)-1], linestyles='dotted')

    plt.ylabel("Loudness")
    ax2 = ax1.twinx()

    ax2.plot(posterior_marginals, 'k')  # Posterior Marginals
    plt.ylim(0, 1)
    # plt.vlines(piece.annotations.audio[0].boundaries, ymin=0, ymax=1, colors="r", linestyle='dotted')  # Tempo seg
    # plt.vlines(piece.annotations.tempo[0].boundaries, ymin=0, ymax=1, colors="b", linestyle='dotted')  # Dyn seg
    plt.vlines(estimation, ymin=0, ymax=1, colors="k", linestyles='dashed')
    plt.title(f"Loudness estimation vs annotation—{piece.piece_id.replace('_',' ')}")
    plt.xlabel("Beats")
    plt.ylabel("Likelihood of boundary")
    plt.savefig(f"output/boundaries_loudness/{piece.piece_id}.pdf")
    plt.savefig(f"output/boundaries_loudness/{piece.piece_id}.png")
    plt.show()


# %%
f_loudness = []
f_loudness_to_audio = []
for piece in cosmo_data:
    estimation, _ = scoring.marginal2guess(segmentations_loud[piece.piece_id], tolerance=3, threshold=.5)
    f_loudness.append([scoring.f_measure(estimation, annotation.boundaries, tolerance=3)
                      for annotation in piece.annotations.loudness])
    f_loudness_to_audio.append([scoring.f_measure(estimation, annotation.boundaries, tolerance=3)
                               for annotation in piece.annotations.audio])
    print(piece.piece_id, f"Loudness match: {f_loudness[-1]}", f"Audio match: {f_loudness_to_audio[-1]}")


# %%
[print(value) for value in f_loudness]
print("Mean: ", np.mean([np.mean(value) for value in f_loudness]))

print(np.array(f_loudness_to_audio))
print("Mean: ", np.mean(f_audio))


# %%
for piece in cosmo_data:
    if piece.piece_id != "excerpt_75":
        continue

    posterior_marginals = segmentations_loud[piece.piece_id]
    estimation, _ = scoring.marginal2guess([np.nan, *posterior_marginals], tolerance=3, threshold=.5)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    plt.title(f"Loudness annotation—{piece.piece_id.replace('_',' ')}")
    plt.xlabel("Beats")
    plt.ylabel("Loudness")

    ax1.plot(piece.loudness, color="r")  # Tempo input data
    plt.ylim(0, 1.1*np.nanmax(piece.loudness))

    fig.savefig("output/buildup_example/1.png", dpi=300)
    fig.show()

    colors = ["k", "g", "b", "r", "c", "m"]
    ymin = 0
    ymax = np.nanmax(piece.loudness)*1.1
    for annotator, annotation in piece.annotations.loudness:
        yymin = ymin + (ymax-ymin) * (int(annotator)-1)/6
        yymax = ymin + (ymax-ymin) * (int(annotator))/6
        ax1.vlines(annotation, ymin=yymin, ymax=yymax, colors=colors[int(annotator)-1])

    fig.savefig("output/buildup_example/2.png", dpi=300)
    fig.show()

    for annotator, annotation in piece.annotations.audio:
        ax1.vlines(annotation, ymin=ymin, ymax=ymax, colors=colors[int(annotator)-1], linestyles='dotted')

    fig.savefig("output/buildup_example/3.png", dpi=300)
    fig.show()

    plt.title(f"Loudness estimation vs annotation—{piece.piece_id.replace('_',' ')}")
    ax2 = ax1.twinx()
    plt.ylabel("Likelihood of boundary")

    ax2.plot(posterior_marginals, 'k')  # Posterior Marginals
    plt.ylim(0, 1)
    # plt.vlines(piece.annotations.audio[0].boundaries, ymin=0, ymax=1, colors="r", linestyle='dotted')  # Tempo seg
    # plt.vlines(piece.annotations.tempo[0].boundaries, ymin=0, ymax=1, colors="b", linestyle='dotted')  # Dyn seg

    fig.savefig("output/buildup_example/4.png", dpi=300)
    fig.show()

    ax2.vlines(estimation, ymin=0, ymax=1, colors="k", linestyles='dashed')

    fig.savefig("output/buildup_example/5.png", dpi=300)
    fig.show()


# %%
for piece in cosmo_data:
    fig = plt.figure()
    plt.title(f"Tempo annotations—{piece.piece_id.replace('_',' ')}")
    plt.ylabel("Tempo (bpm)")
    plt.xlabel("Clocktime (s)")
    plt.plot(piece.beats, piece.tempo)
    colors = ["k", "g", "b", "r", "c", "m"]
    ymin = np.nanmin(piece.tempo)
    ymax = np.nanmax(piece.tempo)
    for annotator, annotation in piece.annotations.tempo:
        yymin = ymin + (ymax-ymin) * (int(annotator)-1)/6
        yymax = ymin + (ymax-ymin) * (int(annotator))/6
        plt.vlines(scoretime_to_clocktime(annotation, piece.beats),
                   ymin=yymin, ymax=yymax, colors=colors[int(annotator)-1])
    for annotator, annotation in piece.annotations.audio:
        plt.vlines(scoretime_to_clocktime(annotation, piece.beats), ymin=ymin,
                   ymax=ymax, colors=colors[int(annotator)-1], linestyles='dotted')
    fig.savefig(f"output/pure_annot/{piece.piece_id}_tempo.pdf", dpi=300)
    plt.close(fig)

    fig = plt.figure()
    plt.title(f"Loudness annotations—{piece.piece_id.replace('_',' ')}")
    plt.ylabel("Loudness")
    plt.xlabel("Clocktime (s)")
    plt.plot(piece.beats, piece.loudness)
    colors = ["k", "g", "b", "r", "c", "m"]
    ymin = np.nanmin(piece.loudness)
    ymax = np.nanmax(piece.loudness)
    for annotator, annotation in piece.annotations.loudness:
        yymin = ymin + (ymax-ymin) * (int(annotator)-1)/6
        yymax = ymin + (ymax-ymin) * (int(annotator))/6
        plt.vlines(scoretime_to_clocktime(annotation, piece.beats),
                   ymin=yymin, ymax=yymax, colors=colors[int(annotator)-1])
    for annotator, annotation in piece.annotations.audio:
        plt.vlines(scoretime_to_clocktime(annotation, piece.beats), ymin=ymin,
                   ymax=ymax, colors=colors[int(annotator)-1], linestyles='dotted')
    fig.savefig(f"output/pure_annot/{piece.piece_id}_loudness.pdf", dpi=300)
    plt.close(fig)


# %%
deltas = [[] for i in range(6)]
for piece in cosmo_data:
    ann_id = int(piece.annotations.audio[0].author) - 1
    clocktime_audio = scoretime_to_clocktime(piece.annotations.audio[0].boundaries, piece.beats)
    # means[ann_id].extend(list(np.diff(clocktimeAudio)))
    deltas[ann_id].extend(np.diff(clocktime_audio))
print([np.mean(x) for x in deltas], '\n', [np.std(x) for x in deltas])


# %%


# %%


# %%


# %%
