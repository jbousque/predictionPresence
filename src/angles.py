from __future__ import division
import numpy as np
import logging
import pandas as pd
import math

from pyquaternion import Quaternion

points_subject = ['HeadSubject', 'LeftWristSubject', 'RightWristSubject', 'LeftElbowSubject', 'RightElbowSubject']
points_agent = ['Head', 'LeftHand', 'RightHand', 'LeftArm', 'RightArm']
points_labels = ['Head', 'LeftWrist', 'RightWrist', 'LeftElbow', 'RightElbow']

from feutils import FEUtils

logger = logging.getLogger(__name__)
feu = FEUtils()

def quaternion_to_euler(w, x, y, z):
    sqw = w*w
    sqx = x*x
    sqy = y*y
    sqz = z*z
    unit = sqx + sqy + sqz + sqw # if normalised is one, otherwise is correction factor
    test = x*y + z*w
    if test > 0.499*unit: # singularity at north pole
        heading = 2 * math.atan2(x,w)
        attitude = math.pi/2
        bank = 0
        return heading, attitude, bank

    if test < -0.499*unit: # singularity at south pole
        heading = -2 * math.atan2(x,w)
        attitude = -math.pi/2
        bank = 0
        return heading, attitude, bank

    heading = math.atan2(2*y*w-2*x*z , sqx - sqy - sqz + sqw)
    attitude = math.asin(2*test/unit)
    bank = math.atan2(2*x*w-2*y*z , -sqx + sqy - sqz + sqw)
    return heading, attitude, bank


def angles(out_record, split_ratios, isSubject=True):
    """
    Compute angular speed on 3 axis for 5 body parts.
    :param out_record:
    :param split_ratios:
    :param isSubject:
    :return:
    """
    logger.info('angles(out_record=%s, split_ratios=%s, isSubject=%s' % (str(out_record), str(split_ratios), str(isSubject)))
    df = pd.read_csv(out_record, sep='\t')

    duration = df['chrono'].values[-1]
    intervals = feu.get_intervals(duration, split_ratios)
    logger.debug('angles: intervals %s' % intervals)

    if isSubject:
        points = points_subject
    else:
        points = points_agent

    labels = []
    angles_array = None

    for idx, (left, right) in enumerate(zip(intervals.left.values, intervals.right.values)):

        logger.debug('angles: interval #%d ]%s, %s]' % (idx, str(left), str(right)))

        if idx == 0:
            suffix = 'Start'
        elif idx == 1:
            suffix = 'Mid'
        else:
            suffix = 'End'

        data = df[df['chrono'] >= left]
        data = data[data['chrono'] <= right]

        for idx, point in enumerate(points):

            accel = np.zeros((len(data), 3), dtype='float64')
            #angles = np.zeros((len(df), 3), dtype='float64')
            pyaw, ppitch, proll = 0, 0, 0
            for i in np.arange(len(data)):
                row = data.iloc[i]
                yaw, pitch, roll = quaternion_to_euler(row[point + '_quaw'], row[point + '_quax'], row[point + '_quay'], row[point + '_quaz'])
                yaw, pitch, roll = abs(yaw), abs(pitch), abs(roll)
                if i == 0:
                    pyaw, ppitch, proll = yaw, pitch, roll
                accel[i, :] = [yaw - pyaw, pitch - ppitch, roll - proll]
                #angles[i, :] = [yaw, pitch, roll]
                pyaw, ppitch, proll = yaw, pitch, roll

            means = np.mean(accel, axis=0)
            stds = np.std(accel, axis=0)
            if angles_array is None:
                angles_array = np.hstack([means, stds])
            else:
                angles_array = np.hstack([angles_array, means, stds])
            labels.append('MeanYawAngularSpeed_%s_%s' % (points_labels[idx], suffix))
            labels.append('MeanPitchAngularSpeed_%s_%s' % (points_labels[idx], suffix))
            labels.append('MeanRollAngularSpeed_%s_%s' % (points_labels[idx], suffix))
            labels.append('StdYawAngularSpeed_%s_%s' % (points_labels[idx], suffix))
            labels.append('StdPitchAngularSpeed_%s_%s' % (points_labels[idx], suffix))
            labels.append('StdRollAngularSpeed_%s_%s' % (points_labels[idx], suffix))

    result =  pd.DataFrame(data=angles_array, index=labels).T
    # compute 'compressed' features (averaged over 'hand' body part)
    for feat in ['Yaw', 'Pitch', 'Roll']:
        for suffix in ['Start', 'Mid', 'End']:
            cols = ['Mean%sAngularSpeed_%s_%s' % (feat, point, suffix) for point in points_labels[1:]]
            result['Avg_Mean%sAngularSpeed_Hand_%s' % (feat, suffix)] = result[cols].mean(axis=1)
            cols = ['Std%sAngularSpeed_%s_%s' % (feat, point, suffix) for point in points_labels[1:]]
            result['Avg_Std%sAngularSpeed_Hand_%s' % (feat, suffix)] = result[cols].mean(axis=1)
    logger.info('angle: return %s' % result)

    return result

