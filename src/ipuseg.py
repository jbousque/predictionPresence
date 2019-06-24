import sys
import os
import pandas as pd
import numpy as np
from pydub import AudioSegment
from wavSplitter import splitInThree
import config
import logging
import pickle
from pandas import IntervalIndex

from feutils import FEUtils

sp_globPath = config.SPPAS_SRC_PATH
sys.path.append(sp_globPath)
from annotations.IPUs.ipusseg import sppasIPUs

logger = logging.getLogger(__name__)
feu = FEUtils()

def convertToMono(audioFileLoc, audioFileName):
	"""

	:param audioFileLoc:
	:param audioFileName:
	:return:
	"""
	logger.debug('convertToMono(audioFileLoc=%s, audioFileName=%s)', audioFileLoc, audioFileName)
	audioFile = os.path.join(audioFileLoc, audioFileName)
	fileName, fileExt = os.path.splitext(audioFileName)

	stereo = AudioSegment.from_wav(audioFile)
	mono = stereo.set_channels(1)

	monoFileName = os.path.join(audioFileLoc, fileName + "_mono" + fileExt)
	mono.export(monoFileName, format="wav")
	logger.debug('convertToMono returns %s', monoFileName)
	return monoFileName

def avgIPU_length(audioFileLoc, audioFileName):
	"""

	:param audioFileLoc:
	:param audioFileName:
	:return:
	"""
	logger.debug('avgIPU_length(audioFileLoc=%s, audioFileName=%s)', audioFileLoc, audioFileName)

	f = convertToMono(audioFileLoc, audioFileName)
	#f = os.path.join(audioFileLoc, audioFileName)
	out_dir = os.path.join(audioFileLoc, "outputDir", audioFileName)

	if not os.path.exists(os.path.dirname(out_dir)): os.makedirs(os.path.dirname(out_dir))

	IPUobj = sppasIPUs()
	try:
		IPUobj.run(audiofile = f, trsinputfile=None, trstieridx=None, ntracks=None, diroutput=out_dir, tracksext="xra", trsoutput="output.xra")
		index_file = os.path.join(out_dir, "index.txt")

		summary = pd.read_csv(index_file, sep=' ', header=None)
		summary = summary.dropna()
		print(summary)

		IPU_lengths = summary[1] - summary[0]
		return_val = IPU_lengths.mean()
	except ZeroDivisionError:
		logger.warn('avgIPU_length: SPPAS returned ZeroDivisionError meaning there is no IPU in this segment')
		return_val = 0

	logger.debug('avgIPU_length returns %s' % return_val)
	return return_val


def IPU_length(audioFileLoc, audioFileName):
	"""

	:param audioFileLoc:
	:param audioFileName:
	:return:
	"""
	logger.debug('IPU_length(audioFileLoc=%s, audioFileName=%s)', audioFileLoc, audioFileName)

	f = convertToMono(audioFileLoc, audioFileName)
	#f = os.path.join(audioFileLoc, audioFileName)
	out_dir = os.path.join(audioFileLoc, "outputDir", audioFileName)

	if not os.path.exists(os.path.dirname(out_dir)): os.makedirs(os.path.dirname(out_dir))

	IPUobj = sppasIPUs()
	try:
		IPUobj.run(audiofile = f, trsinputfile=None, trstieridx=None, ntracks=None, diroutput=out_dir, tracksext="xra", trsoutput="output.xra")
		index_file = os.path.join(out_dir, "index.txt")

		summary = pd.read_csv(index_file, sep=' ', header=None)
		summary = summary.dropna()
		return_val = summary
	except ZeroDivisionError:
		logger.warn('IPU_length: SPPAS returned ZeroDivisionError meaning there is no IPU found')
		return_val = 0

	logger.debug('IPU_length returns %s' % str(return_val))
	return return_val

def IPUdriver(audioFilePath, splitUp, isSubject):
	"""

	:param audioFilePath:
	:param splitUp:
	:return:
	"""
	logger.debug('IPUdriver(audioFilePath=%s, splitUp=%s)', audioFilePath, str(splitUp))

	if isSubject:
		segment = AudioSegment.from_file(audioFilePath)
		duration = segment.duration_seconds
		logger.debug('IPUdriver: duration %d' % duration)
		intervals = feu.get_intervals(duration, splitUp)
		logger.debug('IPUdriver: intervals %s' % str(intervals))
		summary = IPU_length(os.path.dirname(audioFilePath), os.path.basename(audioFilePath))
		nb = len(summary)
		avgIPUarr = np.zeros((nb, 3))
		for i in np.arange(nb):
			begin = summary.iloc[i, 0]
			end = summary.iloc[i, 1]
			it = feu.get_interval(intervals, begin, end)
			logger.debug('IPUdriver: interval %d for begin %d end %d (midpoint %f)' % (it, begin, end, (end + begin) / 2))
			avgIPUarr[i, it] = (end - begin)  # ipu in seconds
		avgIPUarr = np.mean(avgIPUarr, axis=0)

	else:
		candidate, envType = feu.extract_info(audioFilePath)
		f_agent = open(os.path.join(config.TMP_PATH, candidate, envType, 'agent_speech_vectors.pkl'))
		[agent_times, agent_markers] = pickle.load(f_agent)
		subject_paths = feu.get_filtered_file_paths([candidate], [envType])
		subject_wav_path = None
		for potential_subject_wav_path in subject_paths[0]:
			_, ext = os.path.splitext(potential_subject_wav_path)
			if (ext == ".wav"):
				subject_wav_path = potential_subject_wav_path
		if subject_wav_path is not None:
			# Note: slowness of retrieving doctor wav only to retrieve duration of interaction ...
			logger.debug('IPUdriver: found doctor wav %s' % subject_wav_path)
			segment = AudioSegment.from_file(subject_wav_path)
			duration = segment.duration_seconds * 1000
		else:
			duration = agent_times[-1] * 1000

		logger.debug('IPUdriver: duration %d' % duration)
		logger.debug('IPUdriver: last time tick %d' % agent_times[-1])

		intervals = feu.get_intervals(duration, splitUp)
		logger.debug('IPUdriver: intervals %s' % str(intervals))

		nb = int(len(agent_times) / 2)
		avgIPUarr = np.zeros((nb, 3))
		for i in np.arange(nb):
			begin = agent_times[i*2]
			end = agent_times[i*2+1]
			it = feu.get_interval(intervals, begin, end)
			logger.debug('IPUdriver: interval %d for begin %d end %d (midpoint %f)' % (it, begin, end, (end-begin)/2))
			avgIPUarr[i, it] = (end - begin) / 1000 # ipu in seconds
		avgIPUarr = np.mean(avgIPUarr, axis=0)



	print(avgIPUarr)
	print(avgIPUarr[0], avgIPUarr[1], avgIPUarr[2])
	logger.debug('IPUdriver returns %s', str(avgIPUarr))
	return avgIPUarr



#compute standard deviation


#avgIPU_length("/home/sameer/Projects/ACORFORMED/Data/Data/E6F/Casque/Video", "E6F-Casque-Video-00025.wav")
#convertToMono("/home/sameer/Projects/ACORFORMED/Data/Data/E6F/Casque/Video", "E6F-Casque-Video-00025.wav")
#IPUdriver("/home/sameer/Projects/ACORFORMED/Data/corpus2017/E7A/Casque/data/E7A-02-Casque-micro.wav", [0.15, 0.70, 0.15])