from __future__ import division
import sys
import os
import subprocess
from pydub import AudioSegment
from collections import defaultdict
import numpy as np
import config
import logging
import shutil
import pickle

from pandas import IntervalIndex


from feutils import FEUtils

logger = logging.getLogger(__name__)
feu = FEUtils()

sp_globPath = config.SPPAS_SRC_PATH
greg_path = config.SPPAS_GREG_SRC_PATH


sys.path.append(sp_globPath)
sys.path.append(greg_path)

import annotationdata.aio
from annotationdata import Transcription
from annotationdata import Tier
from annotationdata import Annotation
from annotationdata import Label
from annotationdata import TimePoint
from annotationdata import TimeInterval

def alignmentFile(transcriptionFile, wavFile, sppaspath, sppasver):
    """

    :param transcriptionFile: xra transcription file name
    :param wavFile: wav file name
    :param sppaspath:
    :param sppasver:
    :return: eaf transcription aligned with audio
    """
    logger.info('alignmentFile(transcriptionFile=%s, wavFile=%s, sppaspath=%s, sppasver=%s', transcriptionFile, wavFile, sppaspath, sppasver)
    fileName, fileExt = os.path.splitext(transcriptionFile)
    elanFile = os.path.join(fileName + ".eaf")
    if os.path.isfile(elanFile) and not config.FORCE_OW:
        logger.debug('alignmentFile: %s already exists', elanFile)
    else:
        cmd = ["python", os.path.join(greg_path, "sppas_afterClosedCap.py"), transcriptionFile, "-D", sppaspath, "-V", sppasver]
        logger.debug("Executing %s", subprocess.list2cmdline(cmd))
        logger.debug(subprocess.check_output(cmd))

    trs = annotationdata.aio.read(elanFile)

    #The SPPAS normalization routine works on the tier with the name 'transcription', or the first tier with the string 'trans' occuring in the name.
    #Hence, the tier named 'ASR-Revised', on which normalization and subsequent processing is desired, is renamed. So is the tier 'ASR-Transcription', which is to be left untouched.

    endValue = -1
    for tier in trs:
        logger.debug('alignmentFile: Tier %s' % tier.GetName())
        if(tier.GetName() == "ASR-Transcription"):
            tier.SetName("ASR-Orig")
        if(tier.GetName() == "ASR-Revised"):
            tier.SetName("transcription")
    endValue = (trs[0].GetAllPoints()[-1].GetValue() + (trs[0].GetAllPoints()[-1].GetRadius() * 2)) * 1000
    logger.debug('alignmentFile: EndValue of Tier ASR-Transcription %s' % endValue)
    annotationdata.aio.write(elanFile, trs)

    wavFileName, wavExt = os.path.splitext(wavFile)
    tokFileName = wavFileName +"-token.eaf"
    phonFileName = wavFileName + "-phon.eaf"
    alignFileName = wavFileName + "-palign.eaf"

    # sometimes transcription ends slightly after sound duration
    song = AudioSegment.from_wav(wavFile)
    logger.debug('alignmentFile: Sound duration %d' % len(song))
    if len(song)  < endValue:
        logger.warn('alignmentFile: increasing sound duration to match transcription')
        newsound = AudioSegment.silent(endValue)
        newsound = newsound.overlay(song, position=0)
        # backup original and remove it from being taken in the loop
        if os.path.isfile(wavFile+'-orig'):
            os.remove(wavFile+'-orig')
        os.rename(wavFile, wavFile+'-orig')
        newsound.export(wavFile, format='wav')
        song = newsound

    # silents parts of sound where doctor does not speech to improve detection of pos (only for doctor not for agent,
    # as aligned transcription for agent is not very good).
    if not os.path.isfile(wavFile+'-orig2') and os.path.basename(transcriptionFile).startswith('agent'):
        subject, mode = feu.extract_info(wavFile)
        logger.debug('alignmentFile: extracted %s / %s' % (subject, mode))
        tier = trs.Find('transcription')
        if tier is not None:
            # retrieve doctor wav file
            file_paths = feu.get_filtered_file_paths([subject], [mode])
            wavPath = None
            for potential_wav_path in file_paths[0]:
                _, extWav = os.path.splitext(potential_wav_path)
                if (extWav == ".wav"):
                    wavPath = potential_wav_path
            if wavPath is not None:
                logger.debug('alignmentFile: found doctor wav %s' % wavPath)
                new_song = None #AudioSegment.silent(len(song))
                doc_song = AudioSegment.from_wav(wavPath) - 40
                beg = 0
                beg_radius = 0
                inside = False
                logger.debug('alignmentFile: length %d' % len(song))
                for idx, point in enumerate(tier.GetAllPoints()):
                    end = point.GetValue() * 1000
                    end_radius = point.GetRadius() * 1000
                    logger.debug(
                        "alignmentFile: overlay range %d-%d..%d+%d" % (beg, beg_radius, end, end_radius))
                    segment = song[beg:end]
                    if not inside:
                        segment = doc_song[beg:end]
                        # new_song = new_song.overlay(song[beg-beg_radius:end+end_radius], position=beg-beg_radius)
                        # lower the sound out of doctor speech
                        #new_song = new_song.overlay(song[beg-beg_radius:end+end_radius] - 100, position=beg-beg_radius)
                    else:
                        segment = song[beg:end] - 10
                    if idx == 0:
                        new_song = segment
                    else:
                        new_song = new_song + segment
                    beg = end
                    beg_radius = end_radius
                    inside = not(inside)
                # backup original and remove it from being taken in the loop
                os.rename(wavFile, wavFile+'-orig2')
                new_song.export(wavFile, format='wav')

    #Subprocess call to create tokenized transcription file.
    if os.path.isfile(tokFileName) and not config.FORCE_OW:
        logger.debug('alignmentFile: %s already exists', tokFileName)
    else:
        logger.debug("alignmentFile: tokenizing " + subprocess.list2cmdline(["python", os.path.join(sppaspath, "sppas", "bin", "tokenize.py"), '-r', os.path.join(sppaspath, "resources", "vocab", "fra.vocab"), "-i", elanFile, "-o", tokFileName]))
        logger.debug(subprocess.check_output(["python", os.path.join(sppaspath, "sppas", "bin", "tokenize.py"), '-r', os.path.join(sppaspath, "resources", "vocab", "fra.vocab"), "-i", elanFile, "-o", tokFileName]))

    if os.path.isfile(phonFileName) and not config.FORCE_OW:
        logger.debug('alignmentFile: %s already exists', phonFileName)
    else:
        logger.debug('alignmentFile: phonetizing ' + subprocess.list2cmdline(["python", os.path.join(sppaspath, "sppas", "bin", "phonetize.py"), '-r', os.path.join(sppaspath, "resources", "dict", "fra.dict"), "-i", tokFileName, "-o", phonFileName]))
        #Subprocess call to create phonetized transcription file.
        logger.debug(subprocess.check_output(["python", os.path.join(sppaspath, "sppas", "bin", "phonetize.py"), '-r', os.path.join(sppaspath, "resources", "dict", "fra.dict"), "-i", tokFileName, "-o", phonFileName]))

    if os.path.isfile(alignFileName) and not config.FORCE_OW:
        logger.debug('alignmentFile: %s already exists', alignFileName)
    else:
        logger.debug('alignmentFile: aligning ' + subprocess.list2cmdline(["python", os.path.join(sppaspath, "sppas", "bin", "alignment.py"), '-w', wavFile, "-i", phonFileName, "-I", tokFileName, "-o", alignFileName, "-r", os.path.join(sppaspath, "resources", "models", "models-fra")]))
        #Subprocess call to create transcription file aligned with audio.
        logger.debug(subprocess.check_output(["python", os.path.join(sppaspath, "sppas", "bin", "alignment.py"), '-w', wavFile, "-i", phonFileName, "-I", tokFileName, "-o", alignFileName, "-r", os.path.join(sppaspath, "resources", "models", "models-fra")]))

    return alignFileName

def POStaggedFile(alignFileName):
    #Inputs 	: eaf transcription aligned with audio
    #Output 	: eaf transcription with a tier for part-of-speech labels
    logger.info("POStaggedFile(alignFileName=%s)", alignFileName)
    # use 'token' instead of 'TokensAlign' ?
    cmd = [config.MARSATAG_COMMAND, '-cli', '-pt', "TokensAlign", "-oral", "-P", "-p", "lpl-oral", "-r", "elan-lite", "-w", "elan-lite", "-in-ext", ".eaf", "--out-ext", "-marsatag.eaf", alignFileName]
    logger.debug("POStaggedFile: Executing cmd [%s]" % ' '.join(cmd))
    #logger.debug("POStaggedFile: %s" % str(subprocess.list2cmdline(cmd)))
    logger.debug(subprocess.check_output(cmd))
    # todo put back "lpl-oral-no-punct" instead of "lpl-oral"


    fileName, fileExt = os.path.splitext(alignFileName)
    logger.info("POStaggedFile return " + os.path.join(fileName + "-marsatag.eaf"))
    return os.path.join(fileName + "-marsatag.eaf")

def PunctuatedFile(alignFileName):
    """

    :param alignFileName: eaf transcription aligned with audio
    :return: eaf transcription with a tier for part-of-speech and punctuation labels
    """
    logger.info("PunctuatedFile(alignFileName=%s)", alignFileName)
    # use 'token' instead of 'TokensAlign' ?
    cmd = [config.MARSATAG_COMMAND, '-cli', '-pt', "TokensAlign", "-oral", "-P", "-p", "lpl-oral", "-r", "elan-lite", "-w", "elan-lite", "-in-ext", ".eaf", "--out-ext", "-marsatagPunc.eaf", alignFileName]
    logger.debug("PunctuatedFile: " + subprocess.list2cmdline(cmd))
    logger.info(subprocess.check_output(cmd))
    # todo put back "lpl-oral-with-punct" instead of "lpl-oral"

    fileName, fileExt = os.path.splitext(alignFileName)
    logger.info("PunctuatedFile return %s", os.path.join(fileName + "-marsatagPunc.eaf"))
    return os.path.join(fileName + "-marsatagPunc.eaf")

def fix_eaf_parent_ref(eafFileName):
    """
    For any reason in eaf generated by marsatag / postagger, tiers miss 'PARENT_REF' information ...
    :param alignFileName:
    :return:
    """
    logger.debug('fix_eaf_parent_ref('+eafFileName+')')

    with open(eafFileName, 'r') as file:
        filedata = file.read()
    file.close()

    # Replace the target string
    # todo ugly way to fix eaf produced by marsatag ...
    filedata = filedata.replace('<TIER TIER_ID="tokens" LINGUISTIC_TYPE_REF="S-Tokens">',
                                '<TIER TIER_ID="tokens" LINGUISTIC_TYPE_REF="S-Tokens" PARENT_REF="TokensAlign">')
    filedata = filedata.replace('<TIER TIER_ID="category" LINGUISTIC_TYPE_REF="S-Tag">',
                                '<TIER TIER_ID="category" LINGUISTIC_TYPE_REF="S-Tag" PARENT_REF="tokens">')
    filedata = filedata.replace('<TIER TIER_ID="lemma" LINGUISTIC_TYPE_REF="S-Tag">',
                                '<TIER TIER_ID="lemma" LINGUISTIC_TYPE_REF="S-Tag" PARENT_REF="tokens">')
    filedata = filedata.replace('<TIER TIER_ID="morpho" LINGUISTIC_TYPE_REF="S-Tag">',
                                '<TIER TIER_ID="morpho" LINGUISTIC_TYPE_REF="S-Tag" PARENT_REF="tokens">')

    # Write the file out again
    with open(eafFileName, 'w') as file:
        file.write(filedata)
    file.close()


def avgSentenceLength(transcriptionFile, wavFile, splitUp, sppaspath, sppasver):
    """
    Inputs 	:	xra transcription
                wav audio
                phase-wise split of the interaction in the form of a list with 3 elements adding to 1
                SPPAS path
                SPPAS version
    #Output :	3-element numpy array containing average sentence lengths of each phase
    """
    logger.info('avgSentenceLength(transcriptionFile=%s, wavFile=%s, splitUp=%s, sppaspath=%s, sppasver=%s)',
                transcriptionFile, wavFile, splitUp, sppaspath, sppasver)
    segment = AudioSegment.from_file(wavFile)
    duration = segment.duration_seconds

    #splitPoint_1 : time (in seconds) at which the first phase ends and the second begins
    splitPoint_1 = (splitUp[0]) * duration

    #splitPoint_2 : time (in seconds) at which the second phase ends and the third begins
    splitPoint_2 = (splitUp[0] + splitUp[1]) * duration

    taggedTransFile = PunctuatedFile(alignmentFile(transcriptionFile, wavFile, sppaspath, sppasver))

    fix_eaf_parent_ref(taggedTransFile)

    trs = annotationdata.aio.read(taggedTransFile)
    tier = trs.Find("category", case_sensitive=False)

    #sentenceLength : temporary variable counting the length of each sentence from one punctuation annotation to the next
    sentenceLength = 0

    #sentenceCount : 3-element array storing the number of sentences in each phase
    sentenceCount = np.zeros(3)

    #numWords : 3-element array storing the number of words in each phase
    numWords = np.zeros(3)

    # temporary variable remembering beginning time of current sentence
    sentenceBegin = -1

    intervals = IntervalIndex.from_tuples([(0, splitPoint_1), (splitPoint_1, splitPoint_2), (splitPoint_2, duration)])

    for annotation in tier:
        if(annotation.GetLabel().GetValue() == "punctuation"):
            # retrieve in which phase this avg sentence length should be added
            idx = get_interval(intervals, sentenceBegin, annotation.GetLocation().GetBeginMidpoint())
            sentenceCount[idx] += 1
            numWords[idx] = numWords[idx] + sentenceLength
            sentenceLength = 0
            sentenceBegin = -1

        else:
            sentenceLength += 1
            if sentenceBegin == -1:
                sentenceBegin = annotation.getLocation().getBeginMidpoint()

    #avgLengths : 3-element array storing the average sentence length in each phase
    avgLengths = np.zeros(3)

    for i in range(3):
        if(sentenceCount[i] == 0):
            avgLengths[i] = 0
        else:
            avgLengths[i] = numWords[i] / sentenceCount[i]
    return avgLengths


def POSfreq(taggedTransFile, wavFile, splitUp):
    """
    Takes in the file generated by MarsaTag, along with a 3-element array for splitup, and returns a list of
    dicionaries with keys as parts-of-speech and values as their frequencies.
    :param taggedTransFile:
    :param wavFile:
    :param splitUp:
    :return:
    """
    logger.debug('POSfreq(taggedTransFile=%s, wavFile=%s, splitUp=%s)', taggedTransFile, wavFile, str(splitUp))
    dictList = []
    for i in range(3):
        dictList.append(defaultdict(list))
    segment = AudioSegment.from_file(wavFile)
    duration = segment.duration_seconds
    #duration_ms = duration * 1000

    splitPoint_1 = (splitUp[0]) * duration
    splitPoint_2 = (splitUp[0] + splitUp[1]) * duration

    logger.debug('POSfreq: split points %d / %d', splitPoint_1, splitPoint_2)

    trs = annotationdata.aio.read(taggedTransFile)

    for tier in trs:
        if(tier.GetName() == "category"):
            break

    for annotation in tier:
        #logger.debug(annotation.GetLocation().GetBeginMidpoint() )
        #logger.debug(annotation.GetLabel().GetValue())

        if(annotation.GetLocation().GetBeginMidpoint() < splitPoint_1):
            if(not dictList[0][annotation.GetLabel().GetValue()]):
                dictList[0][annotation.GetLabel().GetValue()] = 1
            else:
                dictList[0][annotation.GetLabel().GetValue()] += 1
        elif(annotation.GetLocation().GetBeginMidpoint() < splitPoint_2):
            if(not dictList[1][annotation.GetLabel().GetValue()]):
                dictList[1][annotation.GetLabel().GetValue()] = 1
            else:
                dictList[1][annotation.GetLabel().GetValue()] += 1
        else:
            if(not dictList[2][annotation.GetLabel().GetValue()]):
                dictList[2][annotation.GetLabel().GetValue()] = 1
            else:
                dictList[2][annotation.GetLabel().GetValue()] += 1

    logger.debug('POSfreq returns %s', str(dictList))
    return dictList

def POSfeatures(transcriptionFile, wavFile, splitUp, sppaspath, sppasver):
    """
    Inputs 	:	xra transcription
                wav audio
                phase-wise split of the interaction in the form of a list with 3 elements adding to 1
                SPPAS path
                SPPAS version
    #Output :	9x3 numpy array containing the frequency of each POS tag in each phase (with rows arranged in lexicographic order of tag name)
    """
    logger.info('POSfeatures(transcriptionFile=%s, wavFile=%s, splitUp=%s, sppaspath=%s, sppasver=%s)', transcriptionFile, wavFile, splitUp, sppaspath, sppasver)
    posFile = POStaggedFile(alignmentFile(transcriptionFile, wavFile, sppaspath, sppasver))
    fix_eaf_parent_ref(posFile)
    POSdict = POSfreq(posFile, wavFile, splitUp)
    features = np.zeros((9, 3))

    for i in range(3):
        if(POSdict[i]["adjective"]):
            features[0][i] = POSdict[i]["adjective"]
        if(POSdict[i]["adverb"]):
            features[1][i] = POSdict[i]["adverb"]
        if(POSdict[i]["auxiliary"]):
            features[2][i] = POSdict[i]["auxiliary"]
        if(POSdict[i]["conjunction"]):
            features[3][i] = POSdict[i]["conjunction"]
        if(POSdict[i]["determiner"]):
            features[4][i] = POSdict[i]["determiner"]
        if(POSdict[i]["noun"]):
            features[5][i] = POSdict[i]["noun"]
        if(POSdict[i]["preposition"]):
            features[6][i] = POSdict[i]["preposition"]
        if(POSdict[i]["pronoun"]):
            features[7][i] = POSdict[i]["pronoun"]
        if(POSdict[i]["verb"]):
            features[8][i] = POSdict[i]["verb"]

    logger.debug('POSfeatures returns %s', str(features))

    return features

def get_interval(intervals, begin, end):
    return intervals.get_loc((begin+end) / 2)

def answerDelays(transcriptionFile, wavPath, splitratios, isSubject):

    # BD/ED : begin/end Doctor
    # BA/ES : begin/end Agent
    if isSubject:
        BD = 'BD'
        ED = 'ED'
        BA = 'BA'
        EA = 'EA'
    else:
        BD = 'BA'
        ED = 'EA'
        BA = 'BD'
        EA = 'ED'

    candidate, envType = feu.extract_info(wavPath)
    segment = AudioSegment.from_file(wavPath)
    duration = segment.duration_seconds * 1000
    splitPoint_1 = splitratios[0] * duration
    splitPoint_2 = (splitratios[0] + splitratios[1]) * duration
    intervals = IntervalIndex.from_tuples([(0, splitPoint_1), (splitPoint_1, splitPoint_2), (splitPoint_2, duration)])
    logger.debug('answerDelays: split intervals %s' % str(intervals))

    f_agent = open(os.path.join(config.TMP_PATH, candidate, envType, 'agent_speech_vectors.pkl'))
    [agent_times, agent_markers] = pickle.load(f_agent)

    # retrieve aligned elan file from doctor, if any
    subject_paths = feu.get_filtered_file_paths([candidate], [envType])
    subject_wav_path = None
    for potential_subject_wav_path in subject_paths[0]:
        _, ext = os.path.splitext(potential_subject_wav_path)
        if (ext == ".wav"):
            subject_wav_path = potential_subject_wav_path
    if subject_wav_path is not None:
        logger.debug('answerDelays: found doctor wav %s' % subject_wav_path)
        fileName, fileExt = os.path.splitext(subject_wav_path)
        output = fileName + '-doctor-speech.pkl'
        logger.debug('answerDelays: checking %s' % output)
        if not os.path.isfile(output):
            elanFile = fileName + '-palign.eaf'
            logger.debug('answerDelays: opening elan transcription %s' % elanFile)
            trs = annotationdata.aio.read(elanFile)
            tier = trs.Find("Activity", case_sensitive=False)
            doctor_times = []
            doctor_markers = []
            for annotation in tier:
                if(annotation.GetLabel().GetValue() == "speech"):
                    logger.debug('answerDelays: annotation %s' % str(annotation))
                    object_methods = [method_name for method_name in dir(annotation) if callable(getattr(annotation, method_name))]
                    logger.debug('methods %s ' % str(object_methods))
                    doctor_times.append(int(annotation.GetLocation().GetBeginMidpoint() * 1000))
                    doctor_times.append(int(annotation.GetLocation().GetEndMidpoint() * 1000))
                    doctor_markers.append('BD')
                    doctor_markers.append('ED')
            if doctor_times and doctor_markers:
                f_output = open(output, 'w')
                obj = [doctor_times, doctor_markers]
                pickle.dump(obj, f_output)
                f_output.close()

        else:
            f_output = open(output, 'r')
            obj = pickle.load(f_output)
            doctor_times = obj[0]
            doctor_markers = obj[1]
            f_output.close()

        logger.debug('answerDelays: subject times %s' % (str(doctor_times)))
        logger.debug('answerDelays: subject markers %s' % (str(doctor_markers)))
        logger.debug('answerDelays: agent times %s' % (str(agent_times)))
        logger.debug('answerDelays: agent markers %s' % (str(agent_markers)))
        times = np.array(agent_times + doctor_times)
        markers = np.array(agent_markers + doctor_markers)
        order_idx = np.argsort(times)
        logger.debug('answerDelays: order %s' % str(order_idx))
        logger.debug('answerDelays: all ordered times %s' % str(times[order_idx]))
        logger.debug('answerDelays: all ordered markers %s' % str(markers[order_idx]))

        last_EA = None
        last_EA_idx = None
        delays = [[] for _ in np.arange(len(splitratios))]
        delays_B = []
        for i, idx in enumerate(order_idx):
            logger.debug('answerDelays: #%d idx %d time %s marker %s last_EA %s last_EA_idx %s'
                         % (i, idx, str(times[idx]), str(markers[idx]), str(last_EA), str(last_EA_idx)))
            if markers[idx] == BA:
                logger.debug('answerDelays: (%s) new talk at idx %d time %s' % (BA, idx, str(times[idx])))
                # new talk
                last_EA = None
                last_EA_idx = None
            if markers[idx] == EA:
                logger.debug('answerDelays: (%s) end of talk at idx %d time %s' % (EA, idx, str(times[idx])))
                # last time his talk ended
                last_EA = times[idx]
                last_EA_idx = idx

            # answer ?
            if markers[idx] == BD:
                # occurred end of other speaker segment
                if last_EA is not None:
                    # this is the 'first' answer following this other speaker segment
                    if not delays_B or delays_B[-1] is not last_EA_idx:
                        logger.debug('answerDelays: (%s) new answer (last_EA %s, delays_B[-1] %s, last_EA_idx %s'
                                     % (BD, str(last_EA), str(delays_B[-1]) if delays_B else '[]', str(last_EA_idx)))
                        # actor ended speech at last_EA and answer came at times[idx]
                        delays_B.append(last_EA_idx)
                        it = get_interval(intervals, last_EA, times[idx])
                        delays[it].append(times[idx] - last_EA)
                else:
                    # delay is 0, end of previous segment not reached, so consider current index
                    # (BD) as time of previous end of talk segment
                    delays_B.append(idx)
                    it = get_interval(intervals, times[idx], times[idx])
                    delays[it].append(0)
            logger.debug('answerDelays: update delays_B %s, delays %s' % (str(delays_B), str(delays)))
    delays = np.nan_to_num([np.mean(item) for item in delays])
    logger.debug('answerDelays: return %s' % str(delays))
    return delays