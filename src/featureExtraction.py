import argparse
import logging
import os
import shutil
import sys
import re
import glob
from shutil import copyfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.utils import resample

import config
from feutils import FEUtils
from entropy import videoEntropyMatrix
from ipuseg import IPUdriver
from pos import POSfeatures, avgSentenceLength
from wavSplitter import duration

logger = logging.getLogger(__name__)

gregCorpusPath = config.PREV_CORPUS_PATH  # "/home/sameer/Projects/ACORFORMED/Data/corpus2017"
profBCorpusPath = config.CORPUS_PATH  # "/home/sameer/Projects/ACORFORMED/Data/Data"

feu = FEUtils()



def computePOStags(pathsList, splitratios, isSubject = True):
    crashlist = []
    """"["N01A-02-Casque-micro.E1-5.xra", "N02B-02-PC-micro.E1-5.xra", "E06F-03-Cave-micro.E1-5.xra",
                 "N15C-01-Casque-micro.E1-5.xra", "N03C-01-Casque-micro.E1-5.xra",
                 "N06F-05-PC-micro.E1-5.xra", "N06F-01-PC-micro.E1-5.xra", "N06F-04-Casque-micro.E1-5.xra",
                 "N06F-02-Casque-micro.E1-5.xra", "N21C-03-PC-micro.E1-5.xra", "N22D-02-PC-micro.E1-5.xra",
                 "N22D-02-PC-micro.E1-5.xra", "N22D-03-Cave-micro.E1-5.xra", "N04D-01-Casque-micro.E1-5.xra",
                 "E07A-02-Casque-micro.E1-5.xra"] # "E02B-02-Cave-micro.E2B-latin1.xra", (after N03C)"""
    for paths in pathsList:
        for path in paths:
            fileName, fileext = os.path.splitext(path)
            if (fileext == ".xra" and not any(
                    crashpath in path for crashpath in crashlist)):  # and path not in throughList):
                for wavPath in paths:
                    _, extWav = os.path.splitext(wavPath)
                    if (extWav == ".wav"):
                        # print path, wavPath
                        try:
                            POSfreqArr = POSfeatures(path, wavPath, splitratios, config.SPPAS_PATH, "1.8.6")
                            candidate, envType = feu.extract_info(path)
                            logger.debug('computePOStags: profBCorpusPat%s, candidate=%s, envType=%s, getFeaturesetFolderName(isSubject, splitratios)=%s',
                                         profBCorpusPath, candidate, envType, feu.get_featureset_folder_name(isSubject, splitratios))
                            dest = os.path.join(profBCorpusPath, candidate, envType, feu.get_featureset_folder_name(isSubject, splitratios), "pos.txt")
                            # make sure 'Features' path exists
                            if not os.path.exists(os.path.dirname(dest)): os.makedirs(os.path.dirname(dest))
                            np.savetxt(dest, POSfreqArr)
                        except Exception:
                            logger.exception("computePOStags: failed for %s", wavPath)


def computeSentenceLengths(pathsList, splitratios, isSubject = True):
    # code cleanup: computeSentenceLengths and computePOStags do redundant work. Could be improved
    crashlist = []
    """["N01A-02-Casque-micro.E1-5.xra", "N02B-02-PC-micro.E1-5.xra", "E06F-03-Cave-micro.E1-5.xra",
                 "N15C-01-Casque-micro.E1-5.xra", "N03C-01-Casque-micro.E1-5.xra",
                 "N06F-05-PC-micro.E1-5.xra", "N06F-01-PC-micro.E1-5.xra", "N06F-04-Casque-micro.E1-5.xra",
                 "N06F-02-Casque-micro.E1-5.xra", "N21C-03-PC-micro.E1-5.xra", "N22D-02-PC-micro.E1-5.xra",
                 "N22D-02-PC-micro.E1-5.xra", "N22D-03-Cave-micro.E1-5.xra", "N04D-01-Casque-micro.E1-5.xra",
                 "E07A-02-Casque-micro.E1-5.xra"] # "E02B-02-Cave-micro.E2B-latin1.xra","""
    for paths in pathsList:
        for path in paths:
            fileName, fileext = os.path.splitext(path)
            if (fileext == ".xra" and not any(
                    crashpath in path for crashpath in crashlist)):  # and path not in throughList):
                for wavPath in paths:
                    _, extWav = os.path.splitext(wavPath)
                    if (extWav == ".wav"):
                        # print path, wavPath
                        try:
                            sentenceLengthArray = avgSentenceLength(path, wavPath, splitratios, config.SPPAS_PATH, "1.8.6")
                            candidate, envType = feu.extract_info(path)
                            dest = os.path.join(profBCorpusPath, candidate, envType, feu.get_featureset_folder_name(isSubject, splitratios), "slength.txt")
                            np.savetxt(dest, sentenceLengthArray)
                        except Exception:
                            logger.exception("computeSentenceLengths() failed for %s", wavPath)


def computeEntropies(pathsList, splitratios, isSubject=True):
    # the function computes entropies for all unity files, replacing nan for non-working trackers
    for paths in pathsList:
        for path in paths:
            fileName, fileext = os.path.splitext(path)
            if (fileext == ".txt"):
                #envType = os.path.basename(os.path.normpath(os.path.dirname(os.path.dirname(path))))
                #candidate = os.path.basename(
                #    os.path.normpath(os.path.dirname(os.path.dirname(os.path.dirname(path)))))
                candidate, envType = feu.extract_info(path)
                logger.debug("computeEntropies: path " + path)
                logger.debug("computeEntropies: envTyp "+envType)
                logger.debug("computeEntropies: candidate " + candidate)
                try:
                    entArr = videoEntropyMatrix(path, splitratios)
                    dest = os.path.join(profBCorpusPath, candidate, envType, feu.get_featureset_folder_name(isSubject, splitratios), "entropy.txt")
                    if not os.path.exists(os.path.dirname(dest)): os.makedirs(os.path.dirname(dest))
                    np.savetxt(dest, entArr)
                except Exception:
                    logger.exception("computeEntropies() failed for %s / %s", envType, candidate)


def computeIPUlengths(pathsList, splitratios, isSubject=True):
    crashlist = ["N21C-03-PC-micro.wav", 'IPUtemp']
    for paths in pathsList:
        for path in paths:
            fileName, fileext = os.path.splitext(path)
            if (fileext == ".wav" and not any(crashpath in path for crashpath in crashlist)):
                #envType = os.path.basename(os.path.normpath(os.path.dirname(os.path.dirname(path))))
                #candidate = os.path.basename(os.path.normpath(os.path.dirname(os.path.dirname(os.path.dirname(path)))))
                candidate, envType = feu.extract_info(path)
                try:
                    entArr = IPUdriver(path, splitratios)
                    dest = os.path.join(profBCorpusPath, candidate, envType, feu.get_featureset_folder_name(isSubject, splitratios), "ipu.txt")
                    if not os.path.exists(os.path.dirname(dest)): os.makedirs(os.path.dirname(dest))
                    logger.debug("computeIPUlengths: saving %s", dest)
                    np.savetxt(dest, entArr)
                except Exception:
                    logger.exception("computeIPUlengths() failed for %s / %s / %s", envType, candidate, path)


def sum_nan_arrays(a, b):
    # from https://stackoverflow.com/questions/42209838/treat-nan-as-zero-in-numpy-array-summation-except-for-nan-in-all-arrays
    ma = np.isnan(a)
    mb = np.isnan(b)
    return np.where(ma & mb, np.NaN, np.where(ma, 0, a) + np.where(mb, 0, b))

"""
Traceback (most recent call last):
  File "featureExtraction.py", line 1133, in <module>
    main(sys.argv)
  File "featureExtraction.py", line 1124, in main
    computeFeatures(pathsList, splitratios, isSubject)
  File "featureExtraction.py", line 879, in computeFeatures
    removeNaN(splitratios, isSubject)
  File "featureExtraction.py", line 251, in removeNaN
    sums = sum_nan_arrays(sums, np.loadtxt(completePath))
  File "featureExtraction.py", line 228, in sum_nan_arrays
    return np.where(ma & mb, np.NaN, np.where(ma, 0, a) + np.where(mb, 0, b))
ValueError: operands could not be broadcast together with shapes (15,3) (15,)
"""

def updateFrequencies(current, newArr):
    # not an optimal solution, attempt improvement
    for i in range(newArr.shape[0]):
        for j in range(newArr.shape[1]):
            if (not np.isnan(newArr[i][j])):
                current[i][j] += 1
    return current


def removeNaN(splitratios, isSubject = True):
    # the following function replaces the nan values in the entropy matrices with average values
    sums = np.zeros((15, 3))  # this array eventually contains the sums of the values of the valid entries of each feature
    currentFreq = np.zeros((15, 3))  # supporting array to count the number of valid entries for each feature
    averages = np.zeros((15, 3))
    count = 0
    for dirs, subdirs, files in os.walk(profBCorpusPath, topdown=True, onerror=None,
                                        followlinks=False):  # this loop computes the sums and currentFreq arrays
        for file in files:
            if file == "entropy.txt" and os.path.basename(os.path.normpath(dirs)) == feu.get_featureset_folder_name(isSubject, splitratios):
                completePath = os.path.join(dirs, file)
                sums = sum_nan_arrays(sums, np.loadtxt(completePath))
                currentFreq = updateFrequencies(currentFreq, np.loadtxt(completePath))
                count += 1

    averages = np.divide(sums,
                         currentFreq)  # the average array is used in the next loop to replace nan values with corresponding averages

    for dirs, subdirs, files in os.walk(profBCorpusPath, topdown=True, onerror=None, followlinks=False):
        for file in files:
            if file == "entropy.txt" and os.path.basename(os.path.normpath(dirs)) == feu.get_featureset_folder_name(isSubject, splitratios):
                completePath = os.path.join(dirs, file)
                origMat = np.loadtxt(completePath)
                ma = np.isnan(origMat)

                modifiedMat = np.where(ma, averages, origMat)
                np.savetxt(os.path.join(dirs, "usableEntropies.txt"), modifiedMat) # todo handle features suffix


def extractClass(dframe, candidate, env):
    # the function maps a score in [1, 2) to class '1', a score in [2, 4) to class '2', and a score in [4, 5] to class '3'
    # yet to optimize
    logger.debug("extractClass(candidate=%s, env=%s)", candidate, env)
    for i in range(dframe.shape[0]):
        # print dframe["Candidate"][i], dframe["Environment"][i]
        if (dframe["Candidate"][i] == candidate) and (dframe["Environment"][i] == env):

            if (dframe["Value"][i] >= 1 and dframe["Value"][i] < 2.5):
                logger.debug("extractClass returns %s", (dframe["Value"][i], 1))
                return dframe["Value"][i], 1
            elif (dframe["Value"][i] >= 2.5 and dframe["Value"][i] < 3.5):
                logger.debug("extractClass returns %s", (dframe["Value"][i], 2))
                return dframe["Value"][i], 2
            else:
                logger.debug("extractClass returns %s", (dframe["Value"][i], 3))
                return dframe["Value"][i], 3
    logger.warn("extractClass: class not found in dataframe")


def movePOSfiles(pathsList, splitratios, isSubject=True):
    # following copies pos txt files to the features folder, which were generated in the asr-trans folder, to the Features folder
    for files in pathsList:
        for file in files:
            name, exten = os.path.splitext(file)
            if exten == ".xra":
                posFile = os.path.join(os.path.dirname(file), name + ".txt")
                if os.path.isfile(posFile):
                    envType = os.path.basename(
                        os.path.normpath(os.path.dirname(os.path.dirname(os.path.dirname(posFile)))))
                    candidate = os.path.basename(
                        os.path.normpath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(posFile))))))
                    dest = os.path.join(profBCorpusPath, candidate, envType, feu.get_featureset_folder_name(isSubject, splitratios), "pos.txt")
                    shutil.copy(posFile, dest)


def moveIPUfiles(splitratios, isSubject = True):
    for dirs, subdirs, files in os.walk(gregCorpusPath, topdown=True, onerror=None, followlinks=False):
        if (os.path.basename(os.path.normpath(dirs)) == feu.get_featureset_folder_name(isSubject, splitratios) and os.path.isfile(os.path.join(dirs, 'ipu.txt'))):
            envType = os.path.basename(
                os.path.normpath(os.path.dirname(os.path.dirname(os.path.join(dirs, 'ipu.txt')))))
            candidate = os.path.basename(
                os.path.normpath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.join(dirs, 'ipu.txt'))))))
            target = os.path.join(profBCorpusPath, candidate, envType, feu.get_featureset_folder_name(isSubject, splitratios))
            if os.path.exists(target):
                shutil.copy(os.path.join(dirs, "ipu.txt"), target)


def compressEntropy(entMat):
    head = np.mean(entMat[0:3, :], axis=0)
    leftW = np.mean(entMat[3:6, :], axis=0)
    rightW = np.mean(entMat[6:9, :], axis=0)
    leftE = np.mean(entMat[9:12, :], axis=0)
    rightE = np.mean(entMat[12:15, :], axis=0)

    compressed = np.vstack((head, leftW, rightW, leftE, rightE))
    return compressed


def prepareMatrix(splitratios, isSubject=True):
    """

    :param splitratios:
    :param isSubject:
    :return:
    """
    logger.info("prepareMatrix(isSubject=%s, splitratios=%s)", isSubject, splitratios)
    treated_samples_ids = set()
    # This function was used to traverse the corpus (specifically, the folders where the extracted features are stored) and prepare a feature matrix.
    featureMat = []

    candidateVec = []
    envVec = []
    expertVec = []
    durationVec = []

    classVec = []
    pScoreVec = []
    copClassVec = []
    copScoreVec = []
    logger.debug('prepareMatrix: retrieving features from folders "%s"', feu.get_featureset_folder_name(isSubject, splitratios))
    for dirs, subdirs, files in os.walk(profBCorpusPath, topdown=True, onerror=None, followlinks=False):
        try:
            if (os.path.basename(os.path.normpath(dirs)) == feu.get_featureset_folder_name(isSubject, splitratios)):
                logger.debug('prepareMatrix: treating path ' + dirs)
                entropyFile = os.path.join(dirs, "usableEntropies.txt")
                posFile = os.path.join(dirs, "pos.txt")
                slengthFile = os.path.join(dirs, "slength.txt")
                ipuFile = os.path.join(dirs, "ipu.txt")

                envType = os.path.basename(os.path.normpath(os.path.dirname(os.path.dirname(posFile))))
                candidate = os.path.basename(
                    os.path.normpath(os.path.dirname(os.path.dirname(os.path.dirname(posFile)))))
                candidateLevel = candidate[0]

                if not os.path.isfile(entropyFile):
                    logger.warn('prepareMatrix: missing entropy file %s for %s/%s', entropyFile, candidate, envType)
                elif not os.path.isfile(posFile):
                    logger.warn('prepareMatrix: missing parts of speech file %s for %s/%s', posFile, candidate, envType)
                elif not os.path.isfile(slengthFile):
                    logger.warn('prepareMatrix: missing sentence lengths file %s for %s/%s', slengthFile, candidate, envType)
                elif not os.path.isfile(ipuFile):
                    logger.warn('prepareMatrix: missing IPUs file %s for %s/%s', ipuFile, candidate, envType)
                else:
                    # print "in condition"
                    expert = 1

                    """
                    labels = pd.read_excel("/home/sameer/Projects/ACORFORMED/Data/presence.xlsx")
                    labelsCo = pd.read_excel("/home/sameer/Projects/ACORFORMED/Data/copresence.xlsx")
                    """
                    labels = pd.read_excel(os.path.join(profBCorpusPath, "presence.xlsx"))
                    labelsCo = pd.read_excel(os.path.join(profBCorpusPath, "copresence.xlsx"))

                    if (extractClass(labels, candidate, envType)) is None:
                        continue

                    if (extractClass(labelsCo, candidate, envType)) is None:
                        continue

                    if isSubject:
                        audioTarget = os.path.join(gregCorpusPath, candidate, envType, "data")
                    else:
                        audioTarget = os.path.join(config.TMP_PATH, candidate, envType)
                    logger.debug("prepareMatrix: searching wavs under %s", audioTarget)
                    found = glob.glob(os.path.join(audioTarget, '*.wav'))
                    #for file in os.listdir(audioTarget):
                        # print file, "here"
                        #if (file.endswith(".wav")):
                    if len(found) == 0 and isSubject:
                        # try in alternate directory
                        logger.debug("prepareMatrix: searching wavs under %s",
                                     os.path.join(profBCorpusPath, candidate, envType, 'Superviseur', 'session*', '*.wav'))
                        found = glob.glob(os.path.join(profBCorpusPath, candidate, envType, 'Superviseur', 'session*', '*.wav'))
                    if len(found) > 0:
                        file = found[0]
                        logger.debug('prepareMatrix: found wav %s', file)
                        #dur = duration(os.path.join(audioTarget, file))
                        dur = duration(file)
                        if (durationVec == []):
                            durationVec = [dur]
                        else:
                            durationVec = np.append(durationVec, dur)
                        #break

                    logger.debug("prepareMatrix: length of durationVec %d", len(durationVec))
                    logger.debug("durationVec %s", str(durationVec))

                    entMat = np.loadtxt(entropyFile)
                    compressedEntMat = compressEntropy(entMat)

                    posMat = np.loadtxt(posFile)
                    slengthMat = np.loadtxt(slengthFile)
                    ipuMat = np.loadtxt(ipuFile)
                    # print ipuMat
                    combinedMat = np.vstack((compressedEntMat, posMat, slengthMat, ipuMat))

                    adjecPOSVec = posMat[0, :]
                    advPOSVec = posMat[1, :]

                    sum1 = np.add(adjecPOSVec, advPOSVec)

                    conjPOSVec = posMat[3, :]
                    prepPOSVec = posMat[6, :]
                    pronPOSVec = posMat[7, :]

                    sum2 = np.add(conjPOSVec, prepPOSVec, pronPOSVec)

                    allSum = np.sum(posMat, axis=0)

                    # if there is no token we want ratio to be zero (not NaN because of division by zero)
                    ratio1 = np.divide(sum1, allSum, out=np.zeros_like(sum1), where=allSum!=0)
                    ratio2 = np.divide(sum2, allSum, out=np.zeros_like(sum1), where=allSum!=0)

                    combinedVec = combinedMat.flatten()

                    if (candidate[0] == 'N'):
                        expert = 0

                    if (featureMat == []):
                        featureMat = combinedVec
                        pScore, pClass = extractClass(labels, candidate, envType)
                        pScoreVec = [pScore]
                        classVec = [pClass]

                        copScore, copClass = extractClass(labelsCo, candidate, envType)
                        copScoreVec = [copScore]
                        copClassVec = [copClass]

                        candidateVec = [candidate]
                        envVec = [envType]
                        expertVec = [expert]

                        rat1Vec = ratio1
                        rat2Vec = ratio2
                    else:
                        # print "in this"
                        featureMat = np.vstack((featureMat, combinedVec))
                        pScore, pClass = extractClass(labels, candidate, envType)
                        copScore, copClass = extractClass(labelsCo, candidate, envType)

                        classVec = np.append(classVec, pClass)
                        copClassVec = np.append(copClassVec, copClass)
                        pScoreVec = np.append(pScoreVec, pScore)
                        copScoreVec = np.append(copScoreVec, copScore)

                        candidateVec = np.append(candidateVec, candidate)
                        envVec = np.append(envVec, envType)
                        expertVec = np.append(expertVec, expert)

                        rat1Vec = np.vstack((rat1Vec, ratio1))
                        rat2Vec = np.vstack((rat2Vec, ratio2))

                    treated_samples_ids.add((candidate, envType))

        except Exception as e:
            logger.exception('FAILED to treat %s', dirs)

    logger.debug("prepareMatrix: successfully treated %d samples" % len(treated_samples_ids))
    logger.debug("prepareMatrix: rejected samples: %s" % str(feu.all_samples_ids - treated_samples_ids))

    logger.debug("durationVec %d, pScoreVec %d, classVec %d, copScoreVec %d, copClassVec %d", len(durationVec),
                 len(pScoreVec), len(classVec), len(copScoreVec), len(copClassVec))
    classes = np.stack((np.transpose(durationVec), np.transpose(pScoreVec), np.transpose(classVec),
                        np.transpose(copScoreVec), np.transpose(copClassVec)), axis=-1)
    logger.debug('prepareMatrix: featureMat shape %s', len(featureMat))
    logger.debug('prepareMatrix: classes shape %s', len(classes))
    mat = np.hstack((featureMat, rat1Vec, rat2Vec, classes))
    logger.debug('prepareMatrix: mat shape %s', str(mat.shape))
    pdDump = pd.DataFrame(mat)
    dumpPath = os.path.join(os.path.dirname(profBCorpusPath), config.FEATURES_MATRIX
                            + feu.get_featureset_folder_name(isSubject, splitratios) + '.xlsx')



    pdDump.columns = ['Head_Entropy_Start', 'Head_Entropy_Mid', 'Head_Entropy_End', 'LeftWrist_Entropy_Start',
                      'LeftWrist_Entropy_Mid', 'LeftWrist_Entropy_End', 'RightWrist_Entropy_Start',
                      'RightWrist_Entropy_Mid', 'RightWrist_Entropy_End', 'LeftElbow_Entropy_Start',
                      'LeftElbow_Entropy_Mid', 'LeftElbow_Entropy_End', 'RightElbow_Entropy_Start',
                      'RightElbow_Entropy_Mid', 'RightElbow_Entropy_End', 'Freq_Adjective_Begin', 'Freq_Adjective_Mid',
                      'Freq_Adjective_End', 'Freq_Adverb_Begin', 'Freq_Adverb_Mid', 'Freq_Adverb_End',
                      'Freq_Auxiliary_Begin', 'Freq_Auxiliary_Mid', 'Freq_Auxiliary_End', 'Freq_Conjunction_Begin',
                      'Freq_Conjunction_Mid', 'Freq_Conjunction_End', 'Freq_Determiner_Begin', 'Freq_Determiner_Mid',
                      'Freq_Determiner_End', 'Freq_Noun_Begin', 'Freq_Noun_Mid', 'Freq_Noun_End',
                      'Freq_Preposition_Begin', 'Freq_Preposition_Mid', 'Freq_Preposition_End', 'Freq_Pronoun_Begin',
                      'Freq_Pronoun_Mid', 'Freq_Pronoun_End', 'Freq_Verb_Begin', 'Freq_Verb_Mid', 'Freq_Verb_End',
                      'Avg_SentenceLength_Begin', 'Avg_SentenceLength_Mid', 'Avg_SentenceLength_End',
                      'Avg_IPUlen_Begin', 'Avg_IPUlen_Middle', 'Avg_IPUlen_End', 'Ratio1_Begin', 'Ratio1_Mid',
                      'Ratio1_End', 'Ratio2_Begin', 'Ratio2_Mid', 'Ratio2_End', 'Duration', 'Presence Score',
                      'Presence Class', 'Co-presence Score', 'Co-presence Class']
    # compute average entropies columns
    pdDump['Avg_HandEntropy_Begin'] = pdDump[['LeftWrist_Entropy_Start', 'RightWrist_Entropy_Start',
                                             'LeftElbow_Entropy_Start', 'RightElbow_Entropy_Start']].mean(axis=1)
    pdDump['Avg_HandEntropy_Mid'] = pdDump[['LeftWrist_Entropy_Mid', 'RightWrist_Entropy_Mid', 'LeftElbow_Entropy_Mid',
                                           'RightElbow_Entropy_Mid']].mean(axis=1)
    pdDump['Avg_HandEntropy_End'] = pdDump[['LeftWrist_Entropy_End', 'RightWrist_Entropy_End', 'LeftElbow_Entropy_End',
                                           'RightElbow_Entropy_End']].mean(axis=1)
    pdDump = pdDump[['Head_Entropy_Start', 'Head_Entropy_Mid', 'Head_Entropy_End', 'LeftWrist_Entropy_Start',
                      'LeftWrist_Entropy_Mid', 'LeftWrist_Entropy_End', 'RightWrist_Entropy_Start',
                      'RightWrist_Entropy_Mid', 'RightWrist_Entropy_End', 'LeftElbow_Entropy_Start',
                      'LeftElbow_Entropy_Mid', 'LeftElbow_Entropy_End', 'RightElbow_Entropy_Start',
                      'RightElbow_Entropy_Mid', 'RightElbow_Entropy_End', 'Avg_HandEntropy_Begin',
                      'Avg_HandEntropy_Mid', 'Avg_HandEntropy_End', 'Freq_Adjective_Begin', 'Freq_Adjective_Mid',
                      'Freq_Adjective_End', 'Freq_Adverb_Begin', 'Freq_Adverb_Mid', 'Freq_Adverb_End',
                      'Freq_Auxiliary_Begin', 'Freq_Auxiliary_Mid', 'Freq_Auxiliary_End', 'Freq_Conjunction_Begin',
                      'Freq_Conjunction_Mid', 'Freq_Conjunction_End', 'Freq_Determiner_Begin', 'Freq_Determiner_Mid',
                      'Freq_Determiner_End', 'Freq_Noun_Begin', 'Freq_Noun_Mid', 'Freq_Noun_End',
                      'Freq_Preposition_Begin', 'Freq_Preposition_Mid', 'Freq_Preposition_End', 'Freq_Pronoun_Begin',
                      'Freq_Pronoun_Mid', 'Freq_Pronoun_End', 'Freq_Verb_Begin', 'Freq_Verb_Mid', 'Freq_Verb_End',
                      'Avg_SentenceLength_Begin', 'Avg_SentenceLength_Mid', 'Avg_SentenceLength_End',
                      'Avg_IPUlen_Begin', 'Avg_IPUlen_Middle', 'Avg_IPUlen_End', 'Ratio1_Begin', 'Ratio1_Mid',
                      'Ratio1_End', 'Ratio2_Begin', 'Ratio2_Mid', 'Ratio2_End', 'Duration', 'Presence Score',
                      'Presence Class', 'Co-presence Score', 'Co-presence Class']]
    pdDump.insert(0, 'Candidate', candidateVec)
    pdDump.insert(1, 'Environment', envVec)
    pdDump.insert(2, 'Expert', expertVec)

    pdDump.to_excel(dumpPath, index=False)
    logger.info("prepareMatrix: Saved matrix to %s", dumpPath)
    return mat


def graphTupleList(l):
    n = len(l)
    valList = []
    labelList = []
    for i in range(n):
        valList.append(l[i][0])
        labelList.append(l[i][1])

    fig, ax = plt.subplots()
    index = np.arange(n)
    bar_width = 0.5
    rects1 = ax.bar(index, tuple(valList), bar_width)
    plt.rc('font', size=12)
    ax.set_ylabel("Average decrease in node impurity")

    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(tuple(labelList), rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def randomForest_gridsearch(dataFile, modelTarget, upsample=False):
    logger.info("randomForest_gridsearch(dataFile={df}, modelTarget={mt})".format(df=dataFile, mt=modelTarget))

    from sklearn.model_selection import train_test_split

    samples = pd.read_excel(dataFile)
    samples = samples.rename(index=str,
                             columns={"Presence Class": "PresenceClass", "Co-presence Class": "CopresenceClass"})

    names = ("Expert", "Head_Entropy_Start", "Head_Entropy_Mid", "Head_Entropy_End", "Avg_HandEntropy_Begin",
             "Avg_HandEntropy_Mid", "Avg_HandEntropy_End", "Avg_SentenceLength_Begin", "Avg_SentenceLength_Mid",
             "Avg_SentenceLength_End", "Avg_IPUlen_Begin", "Avg_IPUlen_Middle", "Avg_IPUlen_End", "Ratio1_Begin",
             "Ratio1_Mid", "Ratio1_End", "Ratio2_Begin", "Ratio2_Mid", "Ratio2_End", "Duration")

    samples_split = []
    if (modelTarget == "presence"):
        samples_split.append(samples[samples.PresenceClass == 1])
        samples_split.append(samples[samples.PresenceClass == 2])
        samples_split.append(samples[samples.PresenceClass == 3])

    elif (modelTarget == "copresence"):
        samples_split.append(samples[samples.CopresenceClass == 1])
        samples_split.append(samples[samples.CopresenceClass == 2])
        samples_split.append(samples[samples.CopresenceClass == 3])
    else:
        sys.exit("Invalid input. Please pick between presence and copresence")

    maxClassSize = max(samples_split[0].shape[0], samples_split[1].shape[0], samples_split[2].shape[0])

    if upsample:
        upsampled = []
        # todo upsample with SMOTE algorithm ? https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html
        for idx, samples in enumerate(samples_split):
            if (samples.shape[0] == maxClassSize):
                upsampled.append(samples)
            else:
                logger.debug("resample: adding " + str(maxClassSize - samples.shape[0]) + " samples to class " + str(
                    idx + 1) + " to reach " + str(maxClassSize))
                upsampled.append(resample(samples, replace=True, n_samples=maxClassSize, random_state=None))

        balanced_set = pd.concat(upsampled)
        X = np.nan_to_num(balanced_set.as_matrix(names))

        if (modelTarget == "presence"):
            y = np.array(balanced_set["PresenceClass"].tolist())

        else:
            y = np.array(balanced_set["CopresenceClass"].tolist())

    else:

        X = np.nan_to_num(samples[list(names)])
        if modelTarget == "presence":
            y = samples.PresenceClass
        else:
            y = samples.CopresenceClass

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    logger.debug("X_train ", X_train.shape, "X_test", X_test.shape, "y_train", y_train.shape, "y_test", y_test.shape)

    forest = RandomForestClassifier()

    # print X.shape
    print(modelTarget + " random forest")
    from sklearn.model_selection import GridSearchCV  # todo move to beginning with other imports

    # n_estimators = np.concatenate((np.arange(1,10), np.arange(10,100,10)))
    n_estimators = np.arange(1, 100)
    class_weights = [None, 'balanced', 'balanced_subsample']
    folds = np.concatenate((np.arange(1, 10), np.arange(10, len(X), 10)))
    param_grid = dict(n_estimators=n_estimators, class_weight=class_weights)
    grid = GridSearchCV(estimator=forest, param_grid=param_grid,
                        scoring=['f1_macro', 'precision_macro', 'recall_macro'],
                        refit='precision_macro',
                        cv=20,
                        return_train_score=True,
                        verbose=1)

    grid = grid.fit(X_train, y_train)

    print("TEST Score : ", grid.score(X_test, y_test))

    results = grid.cv_results_
    # print("best params ", grid.best_params_)
    # print("best score ", grid.best_score_)

    return grid


def randomForest(dataFile, modelTarget):
    logger.info("randomForest(dataFile={df}, modelTarget={mt})".format(df=dataFile, mt=modelTarget))

    samples = pd.read_excel(dataFile)

    names = ("Expert", "Head_Entropy_Start", "Head_Entropy_Mid", "Head_Entropy_End", "Avg_HandEntropy_Begin",
             "Avg_HandEntropy_Mid", "Avg_HandEntropy_End", "Avg_SentenceLength_Begin", "Avg_SentenceLength_Mid",
             "Avg_SentenceLength_End", "Avg_IPUlen_Begin", "Avg_IPUlen_Middle", "Avg_IPUlen_End", "Ratio1_Begin",
             "Ratio1_Mid", "Ratio1_End", "Ratio2_Begin", "Ratio2_Mid", "Ratio2_End", "Duration", "Presence Class",
             "Co-presence Class")

    samples = samples[list(names)]
    # JB / not sure why computed matrix has different names...
    samples = samples.rename(index=str,
                             columns={"Presence Class": "PresenceClass", "Co-presence Class": "CopresenceClass"})

    samples_split = []
    if (modelTarget == "presence"):
        samples_split.append(samples[samples.PresenceClass == 1])
        samples_split.append(samples[samples.PresenceClass == 2])
        samples_split.append(samples[samples.PresenceClass == 3])

    elif (modelTarget == "copresence"):
        samples_split.append(samples[samples.CopresenceClass == 1])
        samples_split.append(samples[samples.CopresenceClass == 2])
        samples_split.append(samples[samples.CopresenceClass == 3])
    else:
        sys.exit("Invalid input. Please pick between presence and copresence")

    maxClassSize = max(samples_split[0].shape[0], samples_split[1].shape[0], samples_split[2].shape[0])

    upsampled = []
    # todo upsample with SMOTE algorithm ? https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html
    for samples in samples_split:
        if (samples.shape[0] == maxClassSize):
            upsampled.append(samples)
        else:
            upsampled.append(resample(samples, replace=True, n_samples=maxClassSize, random_state=None))

    balanced_set = pd.concat(upsampled)

    forest = RandomForestClassifier()
    sv = SVC()

    X = np.nan_to_num(balanced_set.as_matrix(names))

    if (modelTarget == "presence"):
        y = np.array(balanced_set["PresenceClass"].tolist())

    else:
        y = np.array(balanced_set["CopresenceClass"].tolist())

    # print X.shape
    print(modelTarget, "random forest")
    print("f1_macro", np.mean(cross_val_score(forest, X, y, cv=10, scoring="f1_macro")))
    print("precision_macro", np.mean(cross_val_score(forest, X, y, cv=10, scoring="precision_macro")))
    print("recall_macro", np.mean(cross_val_score(forest, X, y, cv=10, scoring="recall_macro")))

    print("\n", modelTarget, "SVM")
    print("f1_macro", np.mean(cross_val_score(sv, X, y, cv=10, scoring="f1_macro")))
    print("precision_macro", np.mean(cross_val_score(sv, X, y, cv=10, scoring="precision_macro")))
    print("recall_macro", np.mean(cross_val_score(sv, X, y, cv=10, scoring="recall_macro")))

    # preds = cross_val_predict(forest, X, y, cv=10)
    # print metrics.accuracy_score(y, preds)

    importanceMat = ([[0] * len(names)]) * 1000
    for i in range(1000):
        forest.fit(X, y)
        importanceMat[i] = forest.feature_importances_

    importanceArr = np.asarray(importanceMat)
    stdVec = np.std(importanceArr, axis=0)
    importanceVec = np.sum(importanceArr, axis=0) / 1000

    # dumpPath = "/home/sameer/Projects/ACORFORMED/Data/stats.xlsx"
    dumpPath = os.path.join(os.path.dirname(profBCorpusPath), config.STATS_MATRIX + '.xlsx')
    print("\n")
    descIndices = np.argsort(importanceVec)

    featureStats = np.vstack((importanceVec[descIndices[::-1]], stdVec[descIndices[::-1]]))
    pdDump = pd.DataFrame(featureStats)

    pdDump.columns = np.asarray(names)[descIndices[::-1]]

    print(np.asarray(names)[descIndices[::-1]])
    print(importanceVec[descIndices[::-1]])
    print(stdVec[descIndices[::-1]])
    pdDump.to_excel(dumpPath, index=False)


# graphTupleList (sorted(zip(map(lambda x: round(x, 4), forest.feature_importances_), names), reverse=True))

# writer = pd.ExcelWriter('/home/sameer/Projects/ACORFORMED/Data/upsampled.xlsx')
# balanced_set.to_excel(writer,'Sheet1')
# writer.save()

def presenceModels(dataFile):
    samples = pd.read_excel(dataFile)
    names = ("Avg_HandEntropy_End", "Avg_SentenceLength_End", "Avg_SentenceLength_Mid", "Ratio2_End", "Ratio1_Begin",
             "Head_Entropy_End")
    samples = samples[list(names)]

    samples_split = []
    samples_split.append(samples[samples.PresenceClass == 1])
    samples_split.append(samples[samples.PresenceClass == 2])
    samples_split.append(samples[samples.PresenceClass == 3])

    maxClassSize = max(samples_split[0].shape[0], samples_split[1].shape[0], samples_split[2].shape[0])

    upsampled = []

    for samples in samples_split:
        if (samples.shape[0] == maxClassSize):
            upsampled.append(samples)
        else:
            upsampled.append(resample(samples, replace=True, n_samples=maxClassSize, random_state=None))

    balanced_set = pd.concat(upsampled)

    forest = RandomForestClassifier()
    sv = SVC()
    X = np.nan_to_num(balanced_set.as_matrix(names))

    y = np.array(balanced_set["PresenceClass"].tolist())

    print("random forest")
    print("f1_macro", np.mean(cross_val_score(forest, X, y, cv=10, scoring="f1_macro")))
    print("precision_macro", np.mean(cross_val_score(forest, X, y, cv=10, scoring="precision_macro")))
    print("recall_macro", np.mean(cross_val_score(forest, X, y, cv=10, scoring="recall_macro")))

    print("\n", "SVM")
    print("f1_macro", np.mean(cross_val_score(sv, X, y, cv=10, scoring="f1_macro")))
    print("precision_macro", np.mean(cross_val_score(sv, X, y, cv=10, scoring="precision_macro")))
    print("recall_macro", np.mean(cross_val_score(sv, X, y, cv=10, scoring="recall_macro")))

    # preds = cross_val_predict(forest, X, y, cv=10)
    # print metrics.accuracy_score(y, preds)

    importanceMat = ([[0] * len(names)]) * 1000
    for i in range(1000):
        forest.fit(X, y)
        importanceMat[i] = forest.feature_importances_

    importanceArr = np.asarray(importanceMat)
    stdVec = np.std(importanceArr, axis=0)
    importanceVec = np.sum(importanceArr, axis=0) / 1000

    dumpPath = os.path.join(os.path.dirname(profBCorpusPath), config.STATS_PRESENCE_MATRIX + '.xlsx')
    print("\n")
    descIndices = np.argsort(importanceVec)

    featureStats = np.vstack((importanceVec[descIndices[::-1]], stdVec[descIndices[::-1]]))
    pdDump = pd.DataFrame(featureStats)

    pdDump.columns = np.asarray(names)[descIndices[::-1]]

    print(np.asarray(names)[descIndices[::-1]])
    print(importanceVec[descIndices[::-1]])
    print(stdVec[descIndices[::-1]])
    pdDump.to_excel(dumpPath, index=False)


def copresenceModels(dataFile):
    samples = pd.read_excel(dataFile)
    names = ("Duration", "Ratio2_Begin", "Avg_HandEntropy_Mid", "Avg_SentenceLength_Begin", "Head_Entropy_Mid", "Avg_IPUlen_End")
    samples = samples[list(names)]

    samples_split = []
    samples_split.append(samples[samples.CopresenceClass == 1])
    samples_split.append(samples[samples.CopresenceClass == 2])
    samples_split.append(samples[samples.CopresenceClass == 3])

    maxClassSize = max(samples_split[0].shape[0], samples_split[1].shape[0], samples_split[2].shape[0])

    upsampled = []

    for samples in samples_split:
        if (samples.shape[0] == maxClassSize):
            upsampled.append(samples)
        else:
            upsampled.append(resample(samples, replace=True, n_samples=maxClassSize, random_state=None))

    balanced_set = pd.concat(upsampled)

    forest = RandomForestClassifier()
    sv = SVC()
    X = np.nan_to_num(balanced_set.as_matrix(names))

    y = np.array(balanced_set["CopresenceClass"].tolist())

    print("random forest")
    print("f1_macro", np.mean(cross_val_score(forest, X, y, cv=10, scoring="f1_macro")))
    print("precision_macro", np.mean(cross_val_score(forest, X, y, cv=10, scoring="precision_macro")))
    print("recall_macro", np.mean(cross_val_score(forest, X, y, cv=10, scoring="recall_macro")))

    print("\n", "SVM")
    print("f1_macro", np.mean(cross_val_score(sv, X, y, cv=10, scoring="f1_macro")))
    print("precision_macro", np.mean(cross_val_score(sv, X, y, cv=10, scoring="precision_macro")))
    print("recall_macro", np.mean(cross_val_score(sv, X, y, cv=10, scoring="recall_macro")))

    # preds = cross_val_predict(forest, X, y, cv=10)
    # print metrics.accuracy_score(y, preds)

    importanceMat = ([[0] * len(names)]) * 1000
    for i in range(1000):
        forest.fit(X, y)
        importanceMat[i] = forest.feature_importances_

    importanceArr = np.asarray(importanceMat)
    stdVec = np.std(importanceArr, axis=0)
    importanceVec = np.sum(importanceArr, axis=0) / 1000

    dumpPath = os.path.join(os.path.dirname(profBCorpusPath), config.STATS_COPRESENCE_MATRIX + '.xlsx')
    print("\n")
    descIndices = np.argsort(importanceVec)

    featureStats = np.vstack((importanceVec[descIndices[::-1]], stdVec[descIndices[::-1]]))
    pdDump = pd.DataFrame(featureStats)

    pdDump.columns = np.asarray(names)[descIndices[::-1]]

    print(np.asarray(names)[descIndices[::-1]])
    print(importanceVec[descIndices[::-1]])
    print(stdVec[descIndices[::-1]])
    pdDump.to_excel(dumpPath, index=False)


def computeFeatures(pathsList, splitratios, isSubject=True):
    # Function to call all functions to compute features
    computePOStags(pathsList, splitratios, isSubject)
    computeSentenceLengths(pathsList, splitratios, isSubject)
    computeEntropies(pathsList, splitratios, isSubject)
    removeNaN(splitratios, isSubject)
    computeIPUlengths(pathsList, splitratios, isSubject)


def computeAveragedMatrix(dataFile, outputFile):
    print("computeAveragedMatrix(dataFile={df}, outputFile={of})".format(df=dataFile, of=outputFile))

    samples = pd.read_excel(dataFile)

    names = ("Expert", "Head_Entropy_Start", "Head_Entropy_Mid", "Head_Entropy_End", "Avg_HandEntropy_Begin",
             "Avg_HandEntropy_Mid", "Avg_HandEntropy_End", "Avg_SentenceLength_Begin", "Avg_SentenceLength_Mid",
             "Avg_SentenceLength_End", "Avg_IPUlen_Begin", "Avg_IPUlen_Middle", "Avg_IPUlen_End", "Ratio1_Begin",
             "Ratio1_Mid", "Ratio1_End", "Ratio2_Begin", "Ratio2_Mid", "Ratio2_End", "Duration")

    samples = samples[list(names)]

    samples['Head_Entropy_Avg'] = samples[['Head_Entropy_Start', 'Head_Entropy_Mid', 'Head_Entropy_End']].mean(axis=1)
    print(samples)


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)

def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, ' ', raw_html)
  return cleantext

def preprocess_agent_data(target_candidate=None, target_env=None):
    """

    :return:
    """
    # todo move imports at top of file
    from pydub import AudioSegment
    from os import listdir
    from os.path import isfile, join
    from xml.dom import minidom
    import unicodecsv as csv
    import re
    import subprocess

    logger.info("preprocess_agent_data()")

    # build paths
    outer_arr = {}
    for root, dirs, files in os.walk(config.CORPUS_PATH):

        subject, mode = feu.extract_info(root)
        #logger.debug('preprocess_agent_data: extracted %s / %s' % (subject, mode))
        if target_candidate is None or any(subject == filt for filt in target_candidate):
            if target_env is None or target_env[target_candidate == subject] == mode:
                #inner_arr = {}
                #logger.debug("preprocess_agent_data:searching %s ...", root)
                if os.path.basename(os.path.normcase(root)) == 'unity':
                    logger.debug("preprocess_agent_data:   found unity folder")
                    for file in files:
                        name, ext = os.path.splitext(file)
                        if ext == '.txt' and 'out_record' in name:
                            #inner_arr['rec'] = os.path.join(root, file)
                            out_rec = os.path.join(root, file)
                            logger.debug("preprocess_agent_data:    found out_record %s", out_rec)
                            m = re.search(r'([EN][\d]{1,2}[ABCDEF])[\\/](Casque|PC|Cave)[\\/]', out_rec)
                            if m is not None:
                                mode = m.group(2)
                                subject = m.group(1)
                                if outer_arr.get(subject, '?') == '?':
                                    outer_arr[subject] = {}
                                if outer_arr[subject].get(mode, '?') == '?':
                                    outer_arr[subject][mode] = {}
                                logger.debug("subject %s mode %s", subject, mode)
                                outer_arr[subject][mode]['rec'] = out_rec
                            else: logger.debug("invalid path")
                elif os.path.basename(os.path.normcase(root)).startswith('session-'):
                    logger.debug("preprocess_agent_data:    found session folder")
                    for file in files:
                        name, ext = os.path.splitext(file)
                        if ext == '.xml' and name.startswith('chat-history.'):
                            #inner_arr['chat'] = os.path.join(root, file)
                            chat_xml = os.path.join(root, file)
                            logger.debug("preprocess_agent_data:    found agent chat %s", chat_xml)
                            m = re.search(r'([EN][\d]{1,2}[ABCDEF])[\\/](Casque|PC|Cave)[\\/]', chat_xml)
                            if m is not None:
                                mode = m.group(2)
                                subject = m.group(1)
                                if outer_arr.get(subject, '?') == '?':
                                    outer_arr[subject] = {}
                                if outer_arr[subject].get(mode, '?') == '?':
                                    outer_arr[subject][mode] = {}
                                logger.debug("subject %s mode %s", subject, mode)
                                outer_arr[subject][mode]['chat'] = chat_xml
                            else: logger.debug("invalid path")
                #outer_arr.append(inner_arr)

    logger.debug("preprocess_agent_data: outer_arr %s", str(outer_arr))
    # Treating files
    for subject, modes in outer_arr.iteritems():
    #for files in outer_arr:
        logger.debug("preprocess_agent_data:  treating candidate %s", subject)
        for mode, files in modes.iteritems():
            try:
                logger.debug("preprocess_agent_data:  treating mode %s", mode)
                out_rec = files['rec']
                chat_xml = files['chat']
                out_rec_dir = os.path.dirname(out_rec)
                logger.debug("preprocess_agent_data: treating %s / %s ", subject, mode)

                tmp_dir = os.path.join(config.TMP_PATH, subject, mode)
                if not os.path.exists(tmp_dir): os.makedirs(tmp_dir)
                logger.debug("tmp_dir %s", tmp_dir)

                mydoc = minidom.parse(chat_xml)
                times = []
                texts = []
                # retrieve time alignments from chat-history
                session = mydoc.getElementsByTagName('session')[0]
                for item in session.childNodes:
                    if item.nodeType != item.TEXT_NODE:
                        if item.tagName == 'turn' and item.attributes['speaker'].value == 'greta':
                            times.append(int(item.attributes['startTime'].value))
                            text = item.childNodes[1].firstChild.data
                            print("text %s tags removed %s" % (text, cleanhtml(text)))
                            texts.append(cleanhtml(text))
                        """elif item.tagName == 'event':
                            times.append(int(item.attributes['startTime'].value))
                            texts.append('')"""

                logger.debug('preprocess_agent_data: found %d time steps', len(times))

                new_txt_file = os.path.join(tmp_dir, 'agent_texts.txt')
                np.savetxt(new_txt_file, texts, fmt='%s', encoding='utf8')

                wavdir = out_rec_dir
                wavfiles = [f for f in sorted_alphanumeric(listdir(wavdir)) if
                            isfile(join(wavdir, f)) and f.endswith(".wav")]
                print(wavfiles)
                print(len(wavfiles))

                if len(wavfiles) != len(times):
                    logger.warn("preprocess_agent_data: there should be as many wav files (%d) as IPUs (%d) !", len(wavfiles), len(times))
                    # first we remove empty IPUs if any
                    i = 0
                    for idx, text in enumerate(texts):
                        if texts[idx] == '' and len(times) > len(wavfiles):
                            texts.pop(idx)
                            times.pop(idx)
                        if len(times) == len(wavfiles):
                            break

                    # or we complete number of IPus with fake ones if necessary
                    while len(times) < len(wavfiles):
                        # complete missing time steps with arbitrary pause
                        times.append(times[-1] + len(wavfiles[len(times)]) + 1000)
                        texts.append(' ')

                wavs = []
                for wavfile in wavfiles:
                    logger.debug("loading wav %s", os.path.join(wavdir, wavfile))
                    song = AudioSegment.from_wav(os.path.join(wavdir, wavfile))
                    print(len(song))
                    wavs.append(song)

                # avoid overlaping sounds
                for idx, wav in enumerate(wavs):
                    if idx > 0:
                        print("idx %d %d:%s:%d %d:%s:%d " % (
                        idx, times[idx - 1], texts[idx - 1], len(wavs[idx - 1]),
                        times[idx], texts[idx], len(wavs[idx])))
                        if times[idx] < times[idx - 1] + len(wavs[idx - 1]):
                            times[idx] = times[idx-1]+len(wavs[idx-1]) + 300
                            print("   new time %d" % times[idx])
                    else:
                        print("idx %d %d:%s:%d " % (
                            idx, times[idx], texts[idx], len(wavs[idx])))

                newsound = AudioSegment.silent(duration=times[-1] + len(wavs[-1]))
                for idx, wav in enumerate(wavs):
                    newsound = newsound.overlay(wav, position=times[idx])
                    logger.debug('preprocess_agent_data: aligned wav %d at time %d with text %s' % (idx, times[idx], texts[idx]))
                new_wav_file = os.path.join(tmp_dir, 'agent_sound.wav')
                newsound.export(new_wav_file, format='wav')

                # generate CSV file
                #csvfile = open(os.path.join(tmp_dir, 'agent_transcription.csv'), 'wb')
                #csvwriter = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
                #for i, (time, text) in enumerate(zip(times, texts)):
                #    if i < len(wavfiles):
                #        print(i, time, text)
                #        tb = "{0:.2f}".format(time / 1000)
                #        te = "{0:.2f}".format((time + len(wavs[i])) / 1000)
                #        csvwriter.writerow(['ASR-Transcription', tb, te, text.encode('utf8')])
                #csvfile.close()

                # todo use sppas to convert csv file + wav to xra file
                sppas_cmd = os.path.join(config.SPPAS_2_PATH, 'sppas', 'bin', 'fillipus.py')
                new_xra_file = os.path.join(tmp_dir, 'alt_agent.xra')
                cmd = ['python', sppas_cmd, '-i', new_wav_file, '-t', new_txt_file, '-o', new_xra_file]
                logger.debug("Executing %s", subprocess.list2cmdline(cmd))
                logger.debug(subprocess.check_output(cmd))

                # backup xra
                shutil.copy(new_xra_file, os.path.join(tmp_dir, 'alt_agent_bak.xra'))

                # now convert to format expected by SPPAS 1.8.6
                xra_file = new_xra_file
                with open(xra_file, 'r') as file:
                    filedata = file.read()
                file.close()

                filedata = filedata.replace('<Location>', '<Location><Localization score="1.0">')
                filedata = filedata.replace('</Location>', '</Localization></Location>')
                filedata = filedata.replace('<Interval>', '<Timeinterval>')
                filedata = filedata.replace('</Interval>', '</Timeinterval>')
                filedata = filedata.replace('<Tag>', '<Text score="1.0">')
                filedata = filedata.replace('</Tag>', '</Text>')
                filedata = filedata.replace('tiername="Transcription"', 'tiername="ASR-Transcription"')
                filedata = re.sub(r'<End midpoint="([\d\.]*)" />', r'<End midpoint="\1" radius="0.0005" />', filedata)
                filedata = filedata.replace('</Tier>', '</Tier><Tier id="0" tiername="placeholder"></Tier>')

                # Write the file out again
                with open(xra_file, 'w') as file:
                    file.write(filedata)
                file.close()

                from agent import generate_xra
                generate_xra(texts, times, wavs, os.path.join(tmp_dir, 'agent.xra'))


            except Exception as e:
                logger.exception("FAILURE for %s / %s", subject, mode)



def main(argv):
    # cli parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--splits", nargs='+', type=float, required=False, default=config.DEFAULT_PHASE_SPLIT,
                        help="list of 3 phases ratio, summing to 1")
    parser.add_argument("--agent", help="Whether to compute subject or agent features", action='store_true')
    parser.add_argument("--candidate", nargs='+', type=str, required=False,
                        help='extract features of specified candidate(s) only', default=None)
    parser.add_argument("--env", nargs='+', type=str, required=False,
                        help='extract features for specified environment(s) only (same number and order as --candidate should be provided',
                        default=None)
    parser.add_argument("--pad", help='Whether to preprocess agent sounds or not', action='store_true')
    #parser.add_argument("--matrix", help="Whether to compute features matrix or not", action='store_true')
    #parser.add_argument("--features", help="Whether to compute features or not", action='store_true')
    #parser.add_argument("-o", "--obj", type=str, required=False, default=None,
    #                    help="'presence' or 'copresence' to perform related learning - no learning is performed if missing")
    parser.add_argument('-f', default=None)

    if not os.path.exists(config.LOG_PATH): os.makedirs(config.LOG_PATH)
    if not os.path.exists(config.TMP_PATH): os.makedirs(config.TMP_PATH)

    print("Logging to " + config.LOGFILE)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=config.LOGFILE, level=logging.DEBUG,
                        format='%(levelname)s : %(asctime)s : %(name)s : %(message)s')
    logger = logging.getLogger(__name__)


    logger.debug("Arguments {args}".format(args=sys.argv))
    args = parser.parse_args()
    logger.debug("Arguments parsed {args}".format(args=args))

    isSubject = not(args.agent)
    logger.debug("isSubject ? %s" % str(isSubject))
    targetCandidate = args.candidate
    logger.debug("candidate(s) %s" % targetCandidate)
    targetEnv = args.env
    logger.debug("environment(s) %s" % targetEnv)
    if targetCandidate is not None and targetEnv is not None and len(targetCandidate) != len(targetEnv):
        logger.warn("--env and --candidate do not have matching lengths")
        return

    doPreprocessAgentData = args.pad

    if isSubject:
        pathsList = feu.get_filtered_file_paths(targetCandidate, targetEnv)
    else:
        if doPreprocessAgentData:
            preprocess_agent_data(targetCandidate, targetEnv)
        pathsList = feu.get_filtered_file_paths_agent(targetCandidate, targetEnv)
    logger.debug("pathsList: " + str(pathsList))

    splitratios = args.splits
    logger.info('main: treating featureset "%s"', feu.get_featureset_folder_name(isSubject, splitratios))

    #pathsList = filePaths_agent()

    computeFeatures(pathsList, splitratios, isSubject)
    prepareMatrix(splitratios, isSubject)

    #randomForest(os.path.join(os.path.dirname(profBCorpusPath), config.FEATURES_MATRIX), args.obj)  # todo path

    logger.info("END")


if (__name__ == "__main__"):
    main(sys.argv)
