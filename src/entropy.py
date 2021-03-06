import numpy as np
from scipy.spatial import ConvexHull
from collections import defaultdict
import scipy
import logging

logger = logging.getLogger(__name__)

def coordArray(coordFile):
    coordArr = np.loadtxt(coordFile, skiprows=1)
    return coordArr


def splitArray(coordArray, splitRatios):
    numRows = coordArray.shape[0]
    splitIndex_1 = int(numRows * splitRatios[0])
    splitIndex_2 = int(numRows * (splitRatios[0] + splitRatios[1]))

    beginArr = coordArray[0: splitIndex_1, :]
    midArr = coordArray[splitIndex_1: splitIndex_2, :]
    endArr = coordArray[splitIndex_2:, :]



    return np.array([beginArr, midArr, endArr])


def labels(labelFile):
    labels = np.loadtxt(labelFile, dtype='string')
    return labels


def filteredDict(coordArr, allLabels,
                 subjectLabels):  # returns a dictionary of arrays where each array corersponds to the time series of a subject coordinate point
    d = defaultdict(list)

    for i in range(allLabels.size):
        if (allLabels[i] in subjectLabels):
            d[allLabels[i]] = coordArr[:, i]

    return d


def numTimeSteps(coordArr):
    return coordArr.shape[0]


def numLabels(labels):
    return labels.size


def vectorLength(vec):
    return np.sqrt(np.sum(np.square(vec)))


"""
def pathLength_plane_aux(dispVectors):
	pathLength_plane = 0
	for vec in dispVectors:
		pathLength_plane = pathLength_plane + vectorLength(vec)
	return pathLength_plane
"""


def pathLength_plane(points):
    x_points = points[:, 0]
    y_points = points[:, 1]

    xDiffs = np.diff(x_points)
    yDiffs = np.diff(y_points)

    vecLengths = [vectorLength([xDiffs[i], yDiffs[i]]) for i in range(xDiffs.size)]
    return sum(vecLengths)


def entropy(pointMatrix):
    try:
        logger.debug('entropy: pointMatrix=%s' % str(pointMatrix))
        hull = ConvexHull(pointMatrix)
        vertices = np.append(hull.vertices, hull.vertices[0])

        hull_x = pointMatrix[:, 0][vertices]
        hull_y = pointMatrix[:, 1][vertices]

        hullPts = np.column_stack((hull_x, hull_y))
        hullPerimeter = pathLength_plane(hullPts)

        tarMovement = pathLength_plane(pointMatrix)

        entropy = np.log(2 * tarMovement / hullPerimeter)

        # print "Absolute length: ", tarMovement
        # print "Hull points: ", hullPts
        # print "Hull perimeter: ", hullPerimeter

        return entropy
    except scipy.spatial.qhull.QhullError as e:
        #TODO print input that causes this
        logger.exception('Convex Hull Error')
        return np.NaN



"""
def headEntropies(fileName):
	allLabels = labels("coordLabels")
	subjectLabels = labels("subjectLabels")
	coords = coordArray(fileName)

	coordsDic = filteredDict(coords, allLabels, subjectLabels)

	Head_xy_array = np.column_stack((coordsDic["HeadSubject_posx"], coordsDic["HeadSubject_posy"]))
	Head_yz_array = np.column_stack((coordsDic["HeadSubject_posy"], coordsDic["HeadSubject_posz"]))
	Head_zx_array = np.column_stack((coordsDic["HeadSubject_posz"], coordsDic["HeadSubject_posx"]))

	entropyDict = defaultdict(list)

	entropyDict["HeadSubject_xy"] = entropy(Head_xy_array)
	entropyDict["HeadSubject_yz"] = entropy(Head_yz_array)
	entropyDict["HeadSubject_zx"] = entropy(Head_zx_array)

	return entropyDict
"""


def targetEntropies(target, coordsDic):
    label_x = target + "_posx"
    label_y = target + "_posy"
    label_z = target + "_posz"

    xy_array = np.column_stack((coordsDic[label_x], coordsDic[label_y]))
    yz_array = np.column_stack((coordsDic[label_y], coordsDic[label_z]))
    zx_array = np.column_stack((coordsDic[label_z], coordsDic[label_x]))

    # print xy_array
    return np.array([entropy(xy_array), entropy(yz_array), entropy(zx_array)])


def segmentEntropyMatrix(coords, isSubject):
    """
    takes the coordinate array of one segment and returns the corresponding 15 x 1 entropy matrix, with the
    first 5 values representing the five entropy values of the first plane, followed by 5 valuesfor the second plane,
    and then for the third plane
    :param coords:
    :return:
    """
    allLabels = labels("coordLabels")
    if isSubject:
        type='subject'
    else:
        type='agent'
    typeLabels = labels("{type}Labels".format(type=type))
    coordsDic = filteredDict(coords, allLabels, typeLabels)

    if isSubject:
        featMat = targetEntropies("HeadSubject", coordsDic)
        featMat = np.vstack([featMat, targetEntropies("LeftWristSubject", coordsDic)])
        featMat = np.vstack([featMat, targetEntropies("RightWristSubject", coordsDic)])
        featMat = np.vstack([featMat, targetEntropies("LeftElbowSubject", coordsDic)])
        featMat = np.vstack([featMat, targetEntropies("RightElbowSubject", coordsDic)])
    else:
        featMat = targetEntropies("Head", coordsDic)
        # for agent we don't have wrist, using hand instead ...
        featMat = np.vstack([featMat, targetEntropies("LeftHand", coordsDic)])
        featMat = np.vstack([featMat, targetEntropies("RightHand", coordsDic)])
        # for agent we don't have elbow, using arm instead ...
        featMat = np.vstack([featMat, targetEntropies("LeftArm", coordsDic)])
        featMat = np.vstack([featMat, targetEntropies("RightArm", coordsDic)])

    featMat = np.reshape(featMat, 15)
    return featMat

def videoEntropyMatrix(fileName, splitRatios, isSubject):
    coordArr = coordArray(fileName)
    splitArr = splitArray(coordArr, splitRatios)

    completeEntropyMatrix = np.array([])

    if splitArr[0].size > 0:
        completeEntropyMatrix = segmentEntropyMatrix(splitArr[0], isSubject)
    else:
        completeEntropyMatrix = np.zeros((15))

    if splitArr[1].size > 0:
        completeEntropyMatrix = np.column_stack([completeEntropyMatrix, segmentEntropyMatrix(splitArr[1], isSubject)])
    else:
        completeEntropyMatrix = np.column_stack([completeEntropyMatrix, np.zeros((15))])

    if splitArr[2].size > 0:
        completeEntropyMatrix = np.column_stack([completeEntropyMatrix, segmentEntropyMatrix(splitArr[2], isSubject)])
    else:
        completeEntropyMatrix = np.column_stack([completeEntropyMatrix, np.zeros((15))])

    print(completeEntropyMatrix)
    print(completeEntropyMatrix.shape)
    return completeEntropyMatrix


# videoEntropyMatrix('/home/sameer/Projects/ACORFORMED/Data/Data/N12F/PC/Unity/N12F-PC-Unity-out_record_DATE17-3-2_10-28-55.txt', [0.15, 0.70, 0.15])

# videoEntropyMatrix("E7A-Casque-Unity-out_record_DATE17-6-23_11-54-23.txt", [0.15, 0.70, 0.15])

# videoEntropyMatrix("E7A-Cave-Unity-out_record_DATE17-6-23_11-37-13.txt", [0.15, 0.70, 0.15])

# videoEntropyMatrix("E7A-Casque-Unity-out_record_DATE17-6-23_11-54-23.txt", [0.15, 0.70, 0.15])
