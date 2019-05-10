import numpy as np
import sys
import math
import logging

logger = logging.getLogger(__name__)

class Pair():
    featureValue = 0.0
    classValue = 0

    def __init__(self, suppliedFeatureValue, suppliedClassValue):
        self.featureValue = float(suppliedFeatureValue)
        self.classValue = int(suppliedClassValue)

    def getClassValue(self):
        return self.classValue

    def getFeatureValue(self):
        return self.featureValue

    def __lt__(self, other):
        return self.featureValue < other.featureValue

    def __gt__(self, other):
        return self.featureValue > other.featureValue

    def __eq__(self, other):
        return self.featureValue == other.featureValue and self.classValue == other.classValue

    def __ne__(self, other):
        return self.featureValue != other.featureValue or self.classValue != other.classValue

class MdlDiscretizer:
    pairVector = None
    acceptedCutPoints = []
    possibleCutPoints = []
    numClasses = 0
    possibleCutPointsIdxInPairVector = []

    def __init__(self, suppliedFeatureValues, suppliedClassValues, suppliedNumClasses):
        """

        :param suppliedFeatureValues:
        :param suppliedClassValues:
        :param suppliedNumClasses:
        """
        logger.debug('__init__(suppliedFeatureValues=%s, suppliedClassValues=%s, suppliedNumClasses=%d)' % (suppliedFeatureValues, suppliedClassValues, suppliedNumClasses))
        self.pairVector = [] #Pair[suppliedFeatureValues.size()];
        for i in np.arange(len(suppliedFeatureValues)):
            self.pairVector.append(Pair(float(suppliedFeatureValues[i]), int(suppliedClassValues[i])))
            logger.debug("__init__ Pair %f %d" % (float(suppliedFeatureValues[i]), suppliedClassValues[i]))
        self.pairVector = sorted(self.pairVector)
        self.numClasses = suppliedNumClasses
        self.acceptedCutPoints = []
        self.possibleCutPointsIdxInPairVector = []
        self.possibleCutPoints = []
        self.computePossibleCutPoints()

        if not len(self.possibleCutPoints) == 0 :
            self.recursiveMDLDiscretization(0, len(self.pairVector) - 1)

        if not len(self.acceptedCutPoints) == 0:
            self.acceptedCutPoints = sorted(self.acceptedCutPoints)




    def computePossibleCutPoints(self):
        """

        :return:
        """
        logger.debug('computePossibleCutPoints()')
        previousValue = self.pairVector[0].getFeatureValue()
        previousClassSet = self.getClassList(0)

        for i in np.arange(1, len(self.pairVector)):
            currentValue = self.pairVector[i].getFeatureValue()
            logger.debug('computePossibleCutPoints: currentValue %f previousValue %f' % (currentValue, previousValue))
            if currentValue != previousValue:
                currentClassSet = self.getClassList(i)
                logger.debug('computePossibleCutPoints: a=%d, b=%d, c=%d, d=%d' % (len(previousClassSet), len(currentClassSet),
                    previousClassSet[0], currentClassSet[0]))
                if ((len(previousClassSet) > 1) or (len(currentClassSet) > 1) or (
                    previousClassSet[0] != currentClassSet[0])):
                    logger.debug('computePossibleCutPoints: adding cut point %f' % ((currentValue + previousValue) / 2.0))
                    self.possibleCutPoints.append(float((currentValue + previousValue) / 2.0))
                    logger.debug('computePossibleCutPoints: adding cut point idx %d' % (i-1))
                    self.possibleCutPointsIdxInPairVector.append(i - 1)

                previousClassSet = currentClassSet
                previousValue = currentValue

    def getClassList(self, index):
        """

        :param index:
        :return:
        """
        logger.debug('getClassList(index=%d)' % index)
        foundClasses = []
        foundClasses.append(self.pairVector[index].getClassValue())
        featureValue = self.pairVector[index].getFeatureValue()
        length = len(self.pairVector)
        index += 1


        while (index < length) and (featureValue == self.pairVector[index].getFeatureValue()):
            candidateClass = self.pairVector[index].getClassValue()
            try:
                found = foundClasses.index(int(candidateClass))
            except Exception:
                found = -1
            if found == -1:
                foundClasses.append(int(candidateClass))
            index += 1
        return foundClasses

    def recursiveMDLDiscretization(self, lowerIdx, upperIdx):
        """

        :param lowerIdx:
        :param upperIdx:
        :return:
        """
        logger.debug('recursiveMDLDiscretization(lowerIdx=%d, upperIdx=%d)' % (lowerIdx, upperIdx))
        lowerCutPointIdx = 0
        upperCutPointIdx = 0
        foundLower = False

        for i in np.arange(len(self.possibleCutPointsIdxInPairVector)):
            if  int(self.possibleCutPointsIdxInPairVector[i]) + 1 > lowerIdx \
                    and int(self.possibleCutPointsIdxInPairVector[i]) + 1 < upperIdx :
                if foundLower:
                    upperCutPointIdx = i
                else:
                    lowerCutPointIdx = i
                    upperCutPointIdx = i
                    foundLower = True
        logger.debug('recursive: foundLower %s' % str(foundLower))
        currentBestCutPoint = sys.float_info.min
        currentBestInfo = sys.float_info.min

        for j in np.arange(lowerCutPointIdx, upperCutPointIdx+1):
            limit = int(self.possibleCutPointsIdxInPairVector[j])

            PartitionInformation = float()
            LeftInformation = float()
            RightInformation = float()

            length = upperIdx - lowerIdx + 1
            PartitionInformation = self.computeEntropy(lowerIdx, upperIdx)
            LeftInformation = self.computeEntropy(lowerIdx, limit)
            RightInformation = self.computeEntropy(limit + 1, upperIdx)

            TrialPartitionEntropy = LeftInformation[0] * (limit + 1 - lowerIdx) + RightInformation[0] * (upperIdx - limit)
            TrialPartitionEntropy /= length

            InformationGain = PartitionInformation[0] - TrialPartitionEntropy

            InformationThreshold = np.log(length - 1) / np.log(2.0) + np.log(math.pow(3.0, PartitionInformation[1]) - 2.0) / np.log(2.0)
            InformationThreshold -= PartitionInformation[0] * PartitionInformation[1]
            InformationThreshold += LeftInformation[0] * LeftInformation[1] + RightInformation[0] * RightInformation[1]
            InformationThreshold /= length
            logger.debug('recursive: InformationThreshold=%f, InformationGain=%f, currentBestInfo=%s'
                         % (InformationThreshold, InformationGain,currentBestInfo))
            if (InformationGain > InformationThreshold) and (InformationGain > currentBestInfo):
                currentBestCutPoint = float(self.possibleCutPoints[j])
                currentBestInfo = InformationGain

        if currentBestInfo == sys.float_info.min:
            found = False
        else:
            logger.debug('recursive: adding accepted cut point %f' % currentBestCutPoint)
            self.acceptedCutPoints.append(float(currentBestCutPoint))

            cutPointIdxInPairVector = int(self.possibleCutPointsIdxInPairVector[self.possibleCutPoints
                                                                                    .index(float(currentBestCutPoint))])

            while self.pairVector[cutPointIdxInPairVector].getFeatureValue() == self.pairVector[
                (cutPointIdxInPairVector + 1)].getFeatureValue():
                cutPointIdxInPairVector += 1

            try:
                findCutpoint = self.possibleCutPoints.index(float(currentBestCutPoint))
            except Exception:
                findCutpoint = -1
            if findCutpoint > 0:
                self.recursiveMDLDiscretization(lowerIdx, cutPointIdxInPairVector)
            try:
                findCutpoint = self.possibleCutPoints.index(float(currentBestCutPoint))
            except Exception:
                findCutpoint = -1
            if findCutpoint < len(self.possibleCutPoints) - 1:
                self.recursiveMDLDiscretization(cutPointIdxInPairVector + 1, upperIdx)
            found = True
        logger.info('recursiveMDLDiscretization returns %s' % str(found))
        return found





    def computeEntropy(self, lowerBound, upperBound):
        """

        :param lowerBound:
        :param upperBound:
        :return:
        """

        ClassFrequencies = np.zeros((self.numClasses), dtype=int)
        ReturnedArray = np.zeros((2), dtype=float)

        for i in np.arange(lowerBound, upperBound + 1):
            ClassFrequencies[int(self.pairVector[i].getClassValue())] += 1

        N = upperBound - lowerBound + 1
        info = N * np.log(N)
        counter = 0
        for j in np.arange(0, len(ClassFrequencies)):
            if (ClassFrequencies[j] != 0):
                info -= ClassFrequencies[j] * np.log(ClassFrequencies[j])
                counter += 1
        info /= N * np.log(2.0)

        ReturnedArray[0] = info
        ReturnedArray[1] = counter
        return ReturnedArray


    def getAcceptedCutPoints(self):
        return self.acceptedCutPoints












