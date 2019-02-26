from __future__ import division
import os
import subprocess
import sys
from pydub import AudioSegment
import config


def trimEnd(origFile, threshold, chunk_size):		#chunk_size is in milliseconds
    segment = AudioSegment.from_file(origFile)												#alternatively, to find duration of the audio file, use the ffprobe command : "ffprobe -i SilenceRemoved.wav -show_entries format=duration -v quiet -of csv="p=0""
    duration = segment.duration_seconds
    duration_ms = duration * 1000

    currentChunkStart = int(duration_ms - chunk_size)

    while segment[currentChunkStart : currentChunkStart + chunk_size].dBFS < threshold and currentChunkStart > 0:
    	currentChunkStart = currentChunkStart - chunk_size

    s = segment[0 : currentChunkStart + chunk_size]	
    s.export("endTrimmed.wav", format = "wav")

    return 	s 			#returns an audio segment

def splitInThree(originalAudio, splitUp, removeBip=False):		#splitUp would be a list of three numbers telling where to split the original audio file
    """
    Splits audio in 3 segments with relative lengths given by splitUp array.
    :param originalAudio: 
    :param splitUp: 
    :param removeBip: Trims 12 ms from end of audio (to remove bip, if necessary)
    :return: Nothing, creates 3 files begin.wav, middle.wav, end.wav in IPUtemp directory.
    """
    # the cutoff threshold of-12 is customized for the beep at the end of the videos,
    # run this only if IPUs are to be calculated for the interaction audio file, not needed
    # for the mic input audio file
    if removeBip:
        seg = trimEnd(originalAudio, -12, 1000)
    segment = AudioSegment.from_file(originalAudio)
    duration = segment.duration_seconds
    duration_ms = duration * 1000
    splitPoint_1 = (splitUp[0]) * duration_ms
    splitPoint_2 = (splitUp[0] + splitUp[1]) * duration_ms

    segment_begin = segment[0:splitPoint_1]
    segment_mid = segment[splitPoint_1:splitPoint_2]
    segment_end = segment[splitPoint_2:duration_ms]

    directory = os.path.dirname(originalAudio)

    IPUdir = os.path.join(directory, "IPUtemp")
    if not os.path.exists(IPUdir):
        os.makedirs(IPUdir)

    begin = segment_begin.export(os.path.join(IPUdir, "begin.wav"), format = "wav")
    middle = segment_mid.export(os.path.join(IPUdir, "middle.wav"), format = "wav")
    end = segment_end.export(os.path.join(IPUdir, "end.wav"), format = "wav")

def duration(originalAudio):
    return AudioSegment.from_file(originalAudio).duration_seconds
