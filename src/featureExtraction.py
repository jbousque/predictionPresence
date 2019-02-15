from ipuseg import IPUdriver
from entropy import videoEntropyMatrix
import os
import sys
import shutil
from pos import POSfeatures, avgSentenceLength, PunctuatedFile
from wavSplitter import duration
import numpy as np
import pandas as pd
import scipy
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import config
import argparse
import logging

logger = logging.getLogger(__name__)

gregCorpusPath = config.PREV_CORPUS_PATH # "/home/sameer/Projects/ACORFORMED/Data/corpus2017"
profBCorpusPath = config.CORPUS_PATH # "/home/sameer/Projects/ACORFORMED/Data/Data"

def filePaths():
    #The function collects 4 files for each sample from the two sources in the paths below. The 4 files are: unity coordinates, xra transcription, wav participant mic audio, and the mp4 extracted from the video. It returns an array of arrays. Each outer array corresponds to a sample (a participant-environment combination) and each inner array contains four paths, one corresponding to each of the mentioned files. The output of this function is used by the functions which compute entropies, IPU lengths, sentence lengths, and POS tags.

    outerArr = []
    for subdir in os.listdir(gregCorpusPath):
        #print subdir
        if(os.path.isdir(os.path.join(gregCorpusPath, subdir))):
            for envDir in os.listdir(os.path.join(gregCorpusPath, subdir)):
                #print envDir
                innerArr = []
                if(os.path.isdir(os.path.join(gregCorpusPath, subdir,envDir))) and (os.path.isdir(os.path.join(profBCorpusPath, subdir, envDir))):
                    #print os.path.join(subdir, envDir)
                    for dirs, subdirs, files in os.walk(os.path.join(gregCorpusPath, subdir, envDir), topdown = True, onerror = None, followlinks = False):
                        for file in files:
                            name, exten = os.path.splitext(file)

                            if(os.path.basename(os.path.normpath(dirs)) == 'data') and (exten == ".wav"):
                                innerArr.append(os.path.join(dirs, file))
                                for trsFile in os.listdir(os.path.join(dirs, "asr-trans")):
                                    if trsFile.endswith(".xra"):
                                        innerArr.append(os.path.join(dirs, "asr-trans", trsFile))

                    for dirs, subdirs, files in os.walk(os.path.join(profBCorpusPath, subdir, envDir), topdown = True, onerror = None, followlinks = False):
                        for file in files:
                            name, exten = os.path.splitext(file)    

                            if(os.path.basename(os.path.normpath(dirs)) == 'Video') and (exten == ".mp4"):
                                #print os.path.join(dirs, file)
                                innerArr.append(os.path.join(dirs, file))
                            if(os.path.basename(os.path.normpath(dirs)) == 'Unity') and (exten == ".txt"):
                                #print os.path.join(dirs, file)
                                innerArr.append(os.path.join(dirs, file))                
                    outerArr.append(innerArr)
    return outerArr

def computePOStags(pathsList, splitratios):
    crashlist = ["N01A-02-Casque-micro.E1-5.xra", "N02B-02-PC-micro.E1-5.xra", "E06F-03-Cave-micro.E1-5.xra",
                 "N15C-01-Casque-micro.E1-5.xra", "N03C-01-Casque-micro.E1-5.xra", "E02B-02-Cave-micro.E2B-latin1.xra",
                 "N06F-05-PC-micro.E1-5.xra", "N06F-01-PC-micro.E1-5.xra", "N06F-04-Casque-micro.E1-5.xra",
                 "N06F-02-Casque-micro.E1-5.xra", "N21C-03-PC-micro.E1-5.xra", "N22D-02-PC-micro.E1-5.xra",
                 "N22D-02-PC-micro.E1-5.xra", "N22D-03-Cave-micro.E1-5.xra", "N04D-01-Casque-micro.E1-5.xra",
                 "E07A-02-Casque-micro.E1-5.xra"]
    for paths in pathsList:
        for path in paths:
            fileName, fileext = os.path.splitext(path)
            if(fileext == ".xra" and not any(crashpath in path for crashpath in crashlist)): # and path not in throughList):
                for wavPath in paths:
                    _, extWav = os.path.splitext(wavPath)
                    if(extWav == ".wav"):
                        #print path, wavPath
                        POSfreqArr = POSfeatures(path, wavPath, splitratios, config.SPPAS_PATH, "1.8.6")
                        
                        envType = os.path.basename(os.path.normpath(os.path.dirname(os.path.dirname(os.path.dirname(path)))))
                        candidate = os.path.basename(os.path.normpath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(path))))))
                        dest = os.path.join(profBCorpusPath, candidate, envType, "Features", "pos.txt")
                        # make sure 'Features' path exists
                        if not os.path.exists(os.path.dirname(dest)): os.makedirs(os.path.dirname(dest))
                        np.savetxt(dest, POSfreqArr)


def computeSentenceLengths(pathsList, splitratios):
    #code cleanup: computeSentenceLengths and computePOStags do redundant work. Could be improved
    crashlist = ["N01A-02-Casque-micro.E1-5.xra", "N02B-02-PC-micro.E1-5.xra", "E06F-03-Cave-micro.E1-5.xra",
                 "N15C-01-Casque-micro.E1-5.xra", "N03C-01-Casque-micro.E1-5.xra", "E02B-02-Cave-micro.E2B-latin1.xra",
                 "N06F-05-PC-micro.E1-5.xra", "N06F-01-PC-micro.E1-5.xra", "N06F-04-Casque-micro.E1-5.xra",
                 "N06F-02-Casque-micro.E1-5.xra", "N21C-03-PC-micro.E1-5.xra", "N22D-02-PC-micro.E1-5.xra",
                 "N22D-02-PC-micro.E1-5.xra", "N22D-03-Cave-micro.E1-5.xra", "N04D-01-Casque-micro.E1-5.xra",
                 "E07A-02-Casque-micro.E1-5.xra"]
    for paths in pathsList:
        for path in paths:
            fileName, fileext = os.path.splitext(path)
            if(fileext == ".xra" and not any(crashpath in path for crashpath in crashlist)):# and path not in throughList):
                for wavPath in paths:
                    _, extWav = os.path.splitext(wavPath)
                    if(extWav == ".wav"):
                        #print path, wavPath
                        sentenceLengthArray = avgSentenceLength(path, wavPath, splitratios, config.SPPAS_PATH, "1.8.6")
                        envType = os.path.basename(os.path.normpath(os.path.dirname(os.path.dirname(os.path.dirname(path)))))
                        candidate = os.path.basename(os.path.normpath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(path))))))
                        dest = os.path.join(profBCorpusPath, candidate, envType, "Features", "slength.txt")
                        np.savetxt(dest, sentenceLengthArray)
            

def computeEntropies(pathsList, splitratios):
    #the function computes entropies for all unity files, replacing nan for non-working trackers
    for paths in pathsList:
        for path in paths:
            fileName, fileext = os.path.splitext(path)
            if(fileext == ".txt"):
                entArr = videoEntropyMatrix(path, splitratios)

                nvType = os.path.basename(os.path.normpath(os.path.dirname(os.path.dirname(os.path.dirname(path)))))
                candidate = os.path.basename(os.path.normpath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(path))))))
                dest = os.path.join(profBCorpusPath, candidate, envType, "Features", "entropy.txt")
                np.savetxt(dest, entArr)

def computeIPUlengths(pathsList, splitratios):
    crashlist = ["N21C-03-PC-micro.wav"]
    for paths in pathsList:
        for path in paths:
            fileName, fileext = os.path.splitext(path)
            if(fileext == ".wav" and not any(crashpath in path for crashpath in crashlist)):
                entArr = IPUdriver(path, splitratios)
    
                envType = os.path.basename(os.path.normpath(os.path.dirname(os.path.dirname(path))))
                candidate = os.path.basename(os.path.normpath(os.path.dirname(os.path.dirname(os.path.dirname(path)))))
                dest = os.path.join(profBCorpusPath, candidate, envType, "Features", "ipu.txt")
                np.savetxt(dest, entArr)

def sum_nan_arrays(a, b):
    #from https://stackoverflow.com/questions/42209838/treat-nan-as-zero-in-numpy-array-summation-except-for-nan-in-all-arrays
    ma = np.isnan(a)
    mb = np.isnan(b)
    return  np.where(ma & mb, np.NaN, np.where(ma, 0, a) + np.where(mb, 0, b))

def updateFrequencies(current, newArr):
    #not an optimal solution, attempt improvement
    for i in range(newArr.shape[0]):
        for j in range(newArr.shape[1]):
            if(not np.isnan(newArr[i][j])):
                current[i][j] += 1
    return current

def removeNaN():
    #the following function replaces the nan values in the entropy matrices with average values
    sums = np.zeros((15, 3))            #this array eventually contains the sums of the values of the valid entries of each feature 
    currentFreq = np.zeros((15,3))      #supporting array to count the number of valid entries for each feature
    count = 0
    for dirs, subdirs, files in os.walk(profBCorpusPath, topdown = True, onerror = None, followlinks = False):  #this loop computes the sums and currentFreq arrays
        for file in files:
            if file == "entropy.txt" and os.path.basename(os.path.normpath(dirs)) == "Features":
                completePath = os.path.join(dirs, file)
                sums = sum_nan_arrays(averages, np.loadtxt(completePath))
                currentFreq = updateFrequencies(currentFreq, np.loadtxt(completePath))
                count += 1

    averages = np.divide(sums, currentFreq)     #the average array is used in the next loop to replace nan values with corresponding averages

    for dirs, subdirs, files in os.walk(profBCorpusPath, topdown = True, onerror = None, followlinks = False):
        for file in files:
            if file == "entropy.txt" and os.path.basename(os.path.normpath(dirs)) == "Features":
                completePath = os.path.join(dirs, file)
                origMat = np.loadtxt(completePath)
                ma = np.isnan(origMat)

                modifiedMat = np.where(ma, averages, origMat)
                np.savetxt(os.path.join(dirs, "usableEntropies.txt"), modifiedMat)

def extractClass(dframe, candidate, env):
    #the function maps a score in [1, 2) to class '1', a score in [2, 4) to class '2', and a score in [4, 5] to class '3'
    #yet to optimize
    print candidate, env
    for i in range(dframe.shape[0]):
        #print dframe["Candidate"][i], dframe["Environment"][i]
        if(dframe["Candidate"][i] == candidate) and (dframe["Environment"][i] == env):

            if(dframe["Value"][i] >= 1 and dframe["Value"][i] < 2.5):
                return dframe["Value"][i], 1
            elif(dframe["Value"][i] >= 2.5 and dframe["Value"][i] < 3.5):
                return dframe["Value"][i], 2
            else:
                return dframe["Value"][i], 3

def movePOSfiles(pathsList):
    #following copies pos txt files to the features folder, which were generated in the asr-trans folder, to the Features folder
    for files in pathsList:
        for file in files:
            name, exten = os.path.splitext(file)
            if exten == ".xra":
                posFile = os.path.join(os.path.dirname(file), name + ".txt")
                if os.path.isfile(posFile):
                    envType = os.path.basename(os.path.normpath(os.path.dirname(os.path.dirname(os.path.dirname(posFile)))))
                    candidate = os.path.basename(os.path.normpath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(posFile))))))
                    dest = os.path.join(profBCorpusPath, candidate, envType, "Features", "pos.txt")
                    shutil.copy(posFile, dest)

def moveIPUfiles():
    for dirs, subdirs, files in os.walk(gregCorpusPath, topdown = True, onerror = None, followlinks = False):
        if(os.path.basename(os.path.normpath(dirs)) == "Features" and os.path.isfile(os.path.join(dirs, 'ipu.txt'))):
            envType = os.path.basename(os.path.normpath(os.path.dirname(os.path.dirname(os.path.join(dirs, 'ipu.txt')))))
            candidate = os.path.basename(os.path.normpath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.join(dirs, 'ipu.txt'))))))
            target = os.path.join(profBCorpusPath, candidate, envType,"Features")
            if os.path.exists(target):
                print "yes"
                shutil.copy(os.path.join(dirs, "ipu.txt"), target)

def compressEntropy(entMat):
    head = np.mean(entMat[0:3, :], axis = 0)
    leftW = np.mean(entMat[3:6, :], axis = 0)
    rightW = np.mean(entMat[6:9, :], axis = 0)
    leftE = np.mean(entMat[9:12, :], axis = 0)
    rightE = np.mean(entMat[12:15, :], axis = 0)

    compressed = np.vstack((head, leftW, rightW, leftE, rightE))
    return compressed

def prepareMatrix():
    #This function was used to traverse the corpus (specifically, the folders where the extracted features are stored) and prepare a feature matrix.
    featureMat = []

    candidateVec = []
    envVec = []
    expertVec = []
    durationVec = []

    classVec = []
    pScoreVec = []
    copClassVec = []
    copScoreVec = []
    for dirs, subdirs, files in os.walk(profBCorpusPath, topdown = True, onerror = None, followlinks = False):
        if(os.path.basename(os.path.normpath(dirs)) == "Features"):
            entropyFile = os.path.join(dirs, "usableEntropies.txt")
            posFile = os.path.join(dirs, "pos.txt")
            slengthFile = os.path.join(dirs, "slength.txt")
            ipuFile = os.path.join(dirs, "ipu.txt")

            if(os.path.isfile(entropyFile) and os.path.isfile(posFile) and os.path.isfile(slengthFile) and os.path.isfile(ipuFile)):
                #print "in condition"
                expert = 1
                envType = os.path.basename(os.path.normpath(os.path.dirname(os.path.dirname(posFile))))
                candidate = os.path.basename(os.path.normpath(os.path.dirname(os.path.dirname(os.path.dirname(posFile)))))
                candidateLevel = candidate[0]
                
                """
                labels = pd.read_excel("/home/sameer/Projects/ACORFORMED/Data/presence.xlsx")
                labelsCo = pd.read_excel("/home/sameer/Projects/ACORFORMED/Data/copresence.xlsx")
                """
                labels = pd.read_excel(os.path.join(profBCorpusPath, "presence.xlsx"))
                labelsCo = pd.read_excel(os.path.join(profBCorpusPath, "copresence.xlsx"))
                

                if(extractClass(labels, candidate, envType)) is None:
                    continue

                if(extractClass(labelsCo, candidate, envType)) is None:
                    continue
                

                audioTarget = os.path.join(gregCorpusPath, candidate, envType, "data")

                for file in os.listdir(audioTarget):
                    #print file, "here"
                    if(file.endswith(".wav")):
                        #print file
                        dur = duration(os.path.join(audioTarget, file))
                        if(durationVec == []):
                            durationVec = [dur]
                        else:
                            durationVec = np.append(durationVec, dur)
                        break

                print len(durationVec)

                entMat = np.loadtxt(entropyFile)
                compressedEntMat = compressEntropy(entMat)

                posMat = np.loadtxt(posFile)
                slengthMat = np.loadtxt(slengthFile)
                ipuMat = np.loadtxt(ipuFile)
                #print ipuMat
                combinedMat = np.vstack((compressedEntMat, posMat, slengthMat, ipuMat))

                adjecPOSVec = posMat[0, :]
                advPOSVec = posMat[1, :]

                sum1 = np.add(adjecPOSVec, advPOSVec)

                conjPOSVec = posMat[3, :]
                prepPOSVec = posMat[6, :]
                pronPOSVec = posMat[7, :]

                sum2 = np.add(conjPOSVec,prepPOSVec, pronPOSVec)
                
                allSum = np.sum(posMat, axis = 0)

                ratio1 = np.divide(sum1, allSum)
                ratio2 = np.divide(sum2, allSum)


                combinedVec = combinedMat.flatten()

                if(candidate[0] == 'N'):
                    expert = 0

                if(featureMat == []):
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
                    #print "in this"
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

    classes = np.stack((np.transpose(durationVec), np.transpose(pScoreVec), np.transpose(classVec), np.transpose(copScoreVec), np.transpose(copClassVec)), axis = -1)
    print featureMat.shape
    print classes.shape
    mat =  np.hstack((featureMat, rat1Vec, rat2Vec, classes))

    pdDump = pd.DataFrame(mat)
    dumpPath = os.path.join(os.path.dirname(profBCorpusPath), config.FEATURES_MATRIX)
    pdDump.columns = ['Head_Entropy_Start', 'Head_Entropy_Mid', 'Head_Entropy_End', 'LeftWrist_Entropy_Start', 'LeftWrist_Entropy_Mid', 'LeftWrist_Entropy_End', 'RightWrist_Entropy_Start', 'RightWrist_Entropy_Mid', 'RightWrist_Entropy_End', 'LeftElbow_Entropy_Start', 'LeftElbow_Entropy_Mid', 'LeftElbow_Entropy_End', 'RightElbow_Entropy_Start', 'RightElbow_Entropy_Mid', 'RightElbow_Entropy_End', 'Freq_Adjective_Begin', 'Freq_Adjective_Mid', 'Freq_Adjective_End', 'Freq_Adverb_Begin', 'Freq_Adverb_Mid', 'Freq_Adverb_End','Freq_Auxiliary_Begin', 'Freq_Auxiliary_Mid', 'Freq_Auxiliary_End', 'Freq_Conjunction_Begin', 'Freq_Conjunction_Mid', 'Freq_Conjunction_End', 'Freq_Determiner_Begin', 'Freq_Determiner_Mid', 'Freq_Determiner_End', 'Freq_Noun_Begin', 'Freq_Noun_Mid', 'Freq_Noun_End', 'Freq_Preposition_Begin', 'Freq_Preposition_Mid', 'Freq_Preposition_End', 'Freq_Pronoun_Begin', 'Freq_Pronoun_Mid', 'Freq_Pronoun_End', 'Freq_Verb_Begin', 'Freq_Verb_Mid', 'Freq_Verb_End', 'Avg_SentenceLength_Begin', 'Avg_SentenceLength_Mid', 'Avg_SentenceLength_End', 'Avg_IPUlen_Begin', 'Avg_IPUlen_Middle', 'Avg_IPUlen_End', 'Ratio1_Begin', 'Ratio1_Mid', 'Ratio1_End', 'Ratio2_Begin', 'Ratio2_Mid', 'Ratio2_End', 'Duration', 'Presence Score', 'Presence Class', 'Co-presence Score', 'Co-presence Class']

    pdDump.insert(0, 'Candidate', candidateVec)
    pdDump.insert(1, 'Environment', envVec)
    pdDump.insert(2, 'Expert', expertVec)
    pdDump.to_excel(dumpPath, index = False)
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
    plt.rc('font', size= 12)
    ax.set_ylabel("Average decrease in node impurity")
    
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(tuple(labelList), rotation = 45, ha = "right")
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
             "Ratio1_Mid","Ratio1_End", "Ratio2_Begin", "Ratio2_Mid", "Ratio2_End", "Duration")


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
                print("resample: adding " + str(maxClassSize-samples.shape[0]) + " samples to class " + str(idx+1) + " to reach " + str(maxClassSize) )
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
    print("X_train ", X_train.shape, "X_test", X_test.shape, "y_train", y_train.shape, "y_test", y_test.shape)

    forest = RandomForestClassifier()

    #print X.shape
    print modelTarget, "random forest"
    from sklearn.model_selection import GridSearchCV # todo move to beginning with other imports

    #n_estimators = np.concatenate((np.arange(1,10), np.arange(10,100,10)))
    n_estimators = np.arange(1,100)
    class_weights = [None, 'balanced', 'balanced_subsample']
    folds = np.concatenate((np.arange(1,10), np.arange(10,len(X),10)))
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
    #print("best params ", grid.best_params_)
    #print("best score ", grid.best_score_)

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
        print modelTarget, "random forest"
        print "f1_macro", np.mean(cross_val_score(forest, X, y, cv=10, scoring="f1_macro"))
        print "precision_macro", np.mean(cross_val_score(forest, X, y, cv=10, scoring="precision_macro"))
        print "recall_macro", np.mean(cross_val_score(forest, X, y, cv=10, scoring="recall_macro"))

        print "\n", modelTarget, "SVM"
        print "f1_macro", np.mean(cross_val_score(sv, X, y, cv=10, scoring="f1_macro"))
        print "precision_macro", np.mean(cross_val_score(sv, X, y, cv=10, scoring="precision_macro"))
        print "recall_macro", np.mean(cross_val_score(sv, X, y, cv=10, scoring="recall_macro"))

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
        dumpPath = os.path.join(os.path.dirname(profBCorpusPath), config.STATS_MATRIX)
        print "\n"
        descIndices = np.argsort(importanceVec)

        featureStats = np.vstack((importanceVec[descIndices[::-1]], stdVec[descIndices[::-1]]))
        pdDump = pd.DataFrame(featureStats)

        pdDump.columns = np.asarray(names)[descIndices[::-1]]

        print np.asarray(names)[descIndices[::-1]]
        print importanceVec[descIndices[::-1]]
        print stdVec[descIndices[::-1]]
        pdDump.to_excel(dumpPath, index=False)

    #graphTupleList (sorted(zip(map(lambda x: round(x, 4), forest.feature_importances_), names), reverse=True))
    
   #writer = pd.ExcelWriter('/home/sameer/Projects/ACORFORMED/Data/upsampled.xlsx')
    #balanced_set.to_excel(writer,'Sheet1')
    #writer.save()

def presenceModels(dataFile):
    samples = pd.read_excel(dataFile)
    names = ("Avg_HandEntropy_End", "Avg_SentenceLength_End", "Avg_SentenceLength_Mid", "Ratio2_End", "Ratio1_Begin", "Head_Entropy_End")
    samples = samples[list(names)]


    samples_split = []
    samples_split.append(samples[samples.PresenceClass == 1])
    samples_split.append(samples[samples.PresenceClass == 2])
    samples_split.append(samples[samples.PresenceClass == 3])

    maxClassSize = max(samples_split[0].shape[0], samples_split[1].shape[0], samples_split[2].shape[0])

    upsampled = []

    for samples in samples_split:
        if(samples.shape[0] == maxClassSize):
            upsampled.append(samples)
        else:
            upsampled.append(resample(samples, replace=True, n_samples=maxClassSize, random_state=None))

    balanced_set = pd.concat(upsampled)

    forest = RandomForestClassifier()
    sv = SVC()
    X = np.nan_to_num(balanced_set.as_matrix(names))

    y = np.array(balanced_set["PresenceClass"].tolist())

    print "random forest"
    print "f1_macro", np.mean(cross_val_score(forest, X, y, cv=10, scoring = "f1_macro"))
    print "precision_macro", np.mean(cross_val_score(forest, X, y, cv=10, scoring = "precision_macro"))
    print "recall_macro", np.mean(cross_val_score(forest, X, y, cv=10, scoring = "recall_macro"))

    print "\n", "SVM"
    print "f1_macro", np.mean(cross_val_score(sv, X, y, cv=10, scoring = "f1_macro"))
    print "precision_macro", np.mean(cross_val_score(sv, X, y, cv=10, scoring = "precision_macro"))
    print "recall_macro", np.mean(cross_val_score(sv, X, y, cv=10, scoring = "recall_macro"))

    #preds = cross_val_predict(forest, X, y, cv=10)
    #print metrics.accuracy_score(y, preds)

    importanceMat = ([[0] * len(names)]) * 1000
    for i in range(1000):
        forest.fit(X, y)
        importanceMat[i] = forest.feature_importances_

    importanceArr = np.asarray(importanceMat)
    stdVec= np.std(importanceArr, axis = 0)
    importanceVec = np.sum(importanceArr, axis = 0)/1000


    dumpPath = os.path.join(os.path.dirname(profBCorpusPath), config.STATS_PRESENCE_MATRIX)
    print "\n"
    descIndices = np.argsort(importanceVec)
    
    featureStats = np.vstack((importanceVec[descIndices[::-1]], stdVec[descIndices[::-1]]))
    pdDump = pd.DataFrame(featureStats)

    pdDump.columns = np.asarray(names)[descIndices[::-1]]

    print np.asarray(names)[descIndices[::-1]]
    print importanceVec[descIndices[::-1]]
    print stdVec[descIndices[::-1]]
    pdDump.to_excel(dumpPath, index = False)

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
        if(samples.shape[0] == maxClassSize):
            upsampled.append(samples)
        else:
            upsampled.append(resample(samples, replace=True, n_samples=maxClassSize, random_state=None))

    balanced_set = pd.concat(upsampled)

    forest = RandomForestClassifier()
    sv = SVC()
    X = np.nan_to_num(balanced_set.as_matrix(names))

    y = np.array(balanced_set["CopresenceClass"].tolist())

    print "random forest"
    print "f1_macro", np.mean(cross_val_score(forest, X, y, cv=10, scoring = "f1_macro"))
    print "precision_macro", np.mean(cross_val_score(forest, X, y, cv=10, scoring = "precision_macro"))
    print "recall_macro", np.mean(cross_val_score(forest, X, y, cv=10, scoring = "recall_macro"))

    print "\n", "SVM"
    print "f1_macro", np.mean(cross_val_score(sv, X, y, cv=10, scoring = "f1_macro"))
    print "precision_macro", np.mean(cross_val_score(sv, X, y, cv=10, scoring = "precision_macro"))
    print "recall_macro", np.mean(cross_val_score(sv, X, y, cv=10, scoring = "recall_macro"))

    #preds = cross_val_predict(forest, X, y, cv=10)
    #print metrics.accuracy_score(y, preds)

    importanceMat = ([[0] * len(names)]) * 1000
    for i in range(1000):
        forest.fit(X, y)
        importanceMat[i] = forest.feature_importances_

    importanceArr = np.asarray(importanceMat)
    stdVec= np.std(importanceArr, axis = 0)
    importanceVec = np.sum(importanceArr, axis = 0)/1000


    dumpPath = os.path.join(os.path.dirname(profBCorpusPath), config.STATS_COPRESENCE_MATRIX)
    print "\n"
    descIndices = np.argsort(importanceVec)
    
    featureStats = np.vstack((importanceVec[descIndices[::-1]], stdVec[descIndices[::-1]]))
    pdDump = pd.DataFrame(featureStats)

    pdDump.columns = np.asarray(names)[descIndices[::-1]]

    print np.asarray(names)[descIndices[::-1]]
    print importanceVec[descIndices[::-1]]
    print stdVec[descIndices[::-1]]
    pdDump.to_excel(dumpPath, index = False)

def computeFeatures(pathsList, splitratios):
    #Function to call all functions to compute features
    computePOStags(pathsList, splitratios)
    computeSentenceLengths(pathsList, splitratios)
    computeEntropies(pathsList, splitratios)
    removeNaN()
    computeIPUlengths(pathsList, splitratios)

def computeAveragedMatrix(dataFile, outputFile):
    print("computeAveragedMatrix(dataFile={df}, outputFile={of})".format(df=dataFile, of=outputFile))

    samples = pd.read_excel(dataFile)

    names = ("Expert", "Head_Entropy_Start", "Head_Entropy_Mid", "Head_Entropy_End", "Avg_HandEntropy_Begin", "Avg_HandEntropy_Mid", "Avg_HandEntropy_End", "Avg_SentenceLength_Begin", "Avg_SentenceLength_Mid", "Avg_SentenceLength_End", "Avg_IPUlen_Begin", "Avg_IPUlen_Middle", "Avg_IPUlen_End", "Ratio1_Begin", "Ratio1_Mid", "Ratio1_End", "Ratio2_Begin", "Ratio2_Mid", "Ratio2_End", "Duration")

    samples = samples[list(names)]

    samples['Head_Entropy_Avg'] = samples[['Head_Entropy_Start', 'Head_Entropy_Mid', 'Head_Entropy_End']].mean(axis=1)
    print(samples)



def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)



def preprocess_agent_data():
    from pydub import AudioSegment
    from os import listdir
    from os.path import isfile, join
    from xml.dom import minidom
    import re
    import unicodecsv as csv

    logger.info("preprocess_agent_data()")

    mapping = {}
    out_rec_dir = None
    out_rec = None
    chat_xml = None
    chat_xml_dir = None
    mode = None
    subject = None
    for root, dirs, files in os.walk(config.CORPUS_PATH):
        if os.path.basename(os.path.normcase(root)) == 'unity':
            split_path = os.path.normpath(root).split(os.path.sep)
            for file in files:
                name, ext = os.path.splitext(file)
                if ext == '.txt' and 'out_record' in name:
                    out_rec = file
                    out_rec_dir = root
                    mode = split_path[-2]
                    subject = split_path[-3]
                    logger.debug('Found out record ' + out_rec + ' for ' + subject + ' mode ' + mode)
        elif os.path.basename(os.path.normcase(root)).startswith('session-'):
            for file in files:
                name, ext = os.path.splitext(file)
                if ext == '.xml' and name.startswith('chat-history-'):
                    chat_xml = file
                    chat_xml_dir = root
                    logger.debug('Found chat-history ' + chat_xml)
        if out_rec is not None and chat_xml is not None:
            print("Treating record ... ")

            mydoc = minidom.parse(chat_xml)

            times = []
            texts = []

            # retrieve time alignments from chat-history
            session = mydoc.getElementsByTagName('session')[0]
            for item in session.childNodes:
                if item.nodeType != item.TEXT_NODE:
                    if item.tagName == 'turn' and item.attributes['speaker'].value == 'greta':
                        times.append(int(item.attributes['startTime'].value))
                        texts.append(item.childNodes[1].firstChild.data)
                    elif item.tagName == 'event':
                        times.append(int(item.attributes['startTime'].value))
                        texts.append('')

            print(len(times))

            np.savetxt('newtexts.txt', texts, fmt='%s')

            wavdir = out_rec_dir
            wavfiles = [f for f in sorted_alphanumeric(listdir(wavdir)) if isfile(join(wavdir, f)) and f.endswith(".wav")]
            print(wavfiles)
            print(len(wavfiles))

            if len(wavfiles) != len(times):
                print("There should be as many wav files as IPUs !")

            wavs = []
            for wavfile in wavfiles:
                song = AudioSegment.from_wav(os.path.join(wavdir, wavfile))
                print(len(song))
                wavs.append(song)

            newsound = AudioSegment.silent(duration=times[-1] + len(wavfiles[-1]))
            for idx, wav in enumerate(wavs):
                newsound = newsound.overlay(wav, position=times[idx])
            newsound.export('newsound.wav', format='wav')

            # generate CSV file
            csvfile = open('newtranscription.csv', 'wb')
            csvwriter = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
            for i, (time, text) in enumerate(zip(times, texts)):
                print(i, time, text)
                tb = "{0:.2f}".format(time / 1000)
                te = "{0:.2f}".format((time + len(wavs[i])) / 1000)
                csvwriter.writerow(['ASR-Transcription', tb, te, text.encode('utf8')])
            csvfile.close()

            # todo use sppas to convert csv file + wav to xra file


            # todo

            out_rec = None
            chat_xml = None
            mode = None
            subject = None


def main(argv):

    # cli parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--splits", type=list, required=False, default=config.DEFAULT_PHASE_SPLIT, help="list of 3 phases ratio, summing to 1")
    parser.add_argument("-f", required=False, default="")
    args = parser.parse_args()
    print("Phases ratios: " + str(args.splits))

    print("Logging to "+config.LOGFILE)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=config.LOGFILE, level=logging.DEBUG, format='%(levelname)s : %(asctime)s : %(name)s : %(message)s')
    logger = logging.getLogger(__name__)

    pathsList = filePaths()
    print("pathsList: " + str(pathsList))
    splitratios = args.splits

    computeFeatures(pathsList, splitratios)
    prepareMatrix()
    
    randomForest(os.path.join(os.path.dirname(profBCorpusPath), config.FEATURES_MATRIX), "presence") # todo path

if(__name__ == "__main__"):
    main(sys.argv)


