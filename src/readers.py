#!/bin/python3

import csv
import numpy as np
import os
import itertools as itt

class Performance:
    def __init__(self, interpret, piece, dataType, data):
        self.interpret = interpret
        self.piece = piece
        self.data = {dataType:data}

    def addData(self, dataType, data):
        self.data[dataType]= data


class AnnotatedPerformance:
    def __init__(self, interpret, piece, dataType, data, annot):
        self.interpret = interpret
        self.piece = piece
        self.data = {dataType:data}
        self.annots = {dataType:annot}
    
    def addData(self, dataType, data, annot):
        self.data = {dataType:data}
        self.annots = {dataType:annot}

def readMazurkaData(filename, preprocess=None):
    with open(filename) as csvFile:
        csvReader = csv.reader(csvFile) 
        #Read header
        interpretIds = next(csvReader)[3:] # First 3 columns are not relevant to us
        #zip to read colum by column
        data = zip(*(map(float,row[3:]) for row in csvReader))
        if preprocess is not None:
            data = preprocess(data)
        else:
            data = map(np.array, data)
        return list(zip(interpretIds,data))

def preprocessTimings(timings):
    tempo = map(lambda time : 60/np.diff(time),timings)
    return tempo

def readMazurkaTimings(filename):
    return readMazurkaData(filename, preprocess=preprocessTimings)

def readAllMazurkaTimings(dirpath="beat_time"):
    # Retrieve all mazurka files
    files = [os.path.join(dirpath,file) for file in os.listdir(dirpath) if os.path.splitext(file)[1]==".csv"]
    # Read and return them
    return zip(files,map(readMazurkaTimings,files))

def readAllMazurkaData(dirpath="beat_dyn", preprocess=None):
    # Retrieve all mazurka files
    files = [os.path.join(dirpath,file) for file in os.listdir(dirpath) if os.path.splitext(file)[1]==".csv"]
    # Read and return them
    return list(zip(files,map(lambda f: readMazurkaData(f, preprocess=preprocess),files)))

def readMazurkaArcSegmentation(filename):
    with open(filename) as csvFile:
        csvReader = csv.reader(csvFile)
        seg = [ (line[0],[int(number) for number in line[1:] if number !='']) for line in csvReader]
    return seg

def matchMazurkaSegmentation(filename, dirpath="deaf_structure_tempo", dataType='tempo'):
    if dataType == 'tempo':
        segbasename = os.path.basename(filename.replace("beat_time","_man_seg"))
    elif dataType == 'loudness':
        segbasename = os.path.basename(filename.replace("beat_dynNORM","_man_seg"))
    else:
        raise NotImplementedError("Unknown dataType: "+dataType)
    segfile = os.path.join(dirpath,segbasename)
    if os.path.isfile(segfile):
        return readMazurkaArcSegmentation(segfile)
    else:
        return []

def readAllMazurkaTimingsAndSeg(timingPath = "beat_time", segPath = "deaf_structure_tempo"):
    allData = []
    allTimings = readAllMazurkaTimings(timingPath)
    for filename, timings in allTimings:
        segmentations = matchMazurkaSegmentation(filename,segPath)
        for pID, seg in segmentations:
            tim = next((times for pIDmatch,times in timings if pID == pIDmatch), None)
            if tim is None:
                print("Warning: Encountered performer without a match in timings: "+pID+" in "+filename)
            else:
                allData.append( (filename, pID, tim, seg) )
    return allData

def readAllMazurkaDataAndSeg(timingPath = "beat_dyn", segPath = "deaf_structure_loudness", preprocess=None, dataType='loudness'):
    allPerf = []
    allData = readAllMazurkaData(timingPath, preprocess=preprocess)
    for filename, timings in allData:
        segmentations = matchMazurkaSegmentation(filename,segPath,dataType=dataType)
        for pID, seg in segmentations:
            tim = next((times for pIDmatch,times in timings if pID == pIDmatch), None)
            if tim is None:
                print("Warning: Encountered performer without a match in timings: "+pID+" in "+filename)
            else:
                allPerf.append( (filename, pID, tim, seg) )
    return allPerf

def readAllMultidim(paths, preprocessors, dataTypes):
    it = iter(zip(paths,preprocessors,dataTypes))
    path, preprocessor, dataType = next(it)
    _,interpretData = zip(*readAllMazurkaData(path, preprocessor))
    myDict = dict(map(lambda i: (i[0],Performance(i[0], 'foo', dataType, i[1])),
                      itt.chain(*interpretData)))
    for path, preprocessor, dataType in it:
        _,interpretData = zip(*readAllMazurkaData(path, preprocessor))
        for interpret, data in itt.chain(interpretData):
            myDict[interpret].addData(dataType,data)
    return list(myDict.values())
    


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
    a = readAllMultidim(["beat_dyn"],[None],["Dyn"])
    print("Done")