from __future__ import division
import sys
import os
import subprocess
from pydub import AudioSegment
from collections import defaultdict
import numpy as np
import config
import logging


logger = logging.getLogger(__name__)

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

def generate_xra(texts, times, wavs, output):
	trs = Transcription()
	tier = trs.NewTier('ASR-Transcription')

	for idx, wav in enumerate(wavs):
		time_begin = times[idx]
		time_end = time_begin + len(wavs[idx])

		p1 = TimePoint(time_begin / 1000, radius=0.0025)
		p2 = TimePoint(time_end / 1000, radius=0.0025)
		t = TimeInterval(p1, p2)
		l = Label(texts[idx])
		ann = Annotation(t, l)
		tier.Add(ann)

	tier2 = trs.NewTier('placeholder')

	annotationdata.aio.write(output, trs)


"""<Tier id="0" tiername="placeholder"></Tier>

	<Media id="cdf14143-2d1f-4e47-8b2d-76a0b09d4286" mimetype="audio/wav" url="C:\IAAA\TER\tmp\E2B\Casque\agent_sound.wav">
		<Tier id="1605174a-ec7a-4664-8494-18e1490ee08f" />
	</Media>"""