"""Set of default priors, tuned from a small sample of MazurkaBL performances."""

# Set for tempo
arcPriorTempo = {  # Set after first 10 each of M06-1 and M06-2
    # Gaussian priors on the parameters of ax^2 + bx + c
    'aMean': -181,
    'aStd': 93,
    'bMean': 159,
    'bStd': 106,
    'cMean': 107,
    'cStd': 31,
    'noiseStd': 18.1
}
lengthPriorParamsTempo = {
    'mean': 14.7,
    'stddev': 5.95,
    'maxLength': 30
}

# Set for loudness
arcPriorLoud = {  # Set after first 10 each of M06-1 and M06-2
    # Gaussian priors on the parameters of ax^2 + bx + c
    'aMean': -0.73,
    'aStd': 0.55,
    'bMean': 0.68,
    'bStd': 0.60,
    'cMean': 0.41,
    'cStd': 0.19,
    'noiseStd': 0.039
}
lengthPriorParamsLoud = {
    'mean': 11.8,
    'stddev': 5.53,
    'maxLength': 30
}
