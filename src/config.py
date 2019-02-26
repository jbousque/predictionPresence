"""
A python configuration file containing only variables (properties) declarations.
"""

import os
import random
import numpy as np
from datetime import datetime

ROOT_PATH = os.path.join('C:\\', 'Users', 'bousquet', 'DEV', 'TER')

# Corpus
CORPUS_PATH = os.path.join(ROOT_PATH, 'data', 'ACORFORMED', 'Data') # profBCorpusPath
PREV_CORPUS_PATH = os.path.join(ROOT_PATH, 'data', 'CorpusHMPassation') # gregCorpusPath

# all produced files shall go here
OUT_PATH = os.path.join(ROOT_PATH, '_output')
LOG_PATH = os.path.join(OUT_PATH, 'logs')
TMP_PATH = os.path.join(OUT_PATH, 'tmp')

# Tools

TOOLS_PATH = os.path.join('C:\\', 'Users', 'bousquet', 'TOOLS')

#  SPPAS

SPPAS_1 = 'SPPAS-1.8.6'
SPPAS_2 = 'SPPAS'
SPPAS_PATH = os.path.join(TOOLS_PATH, SPPAS_1)
SPPAS_SRC_PATH = os.path.join(SPPAS_PATH, 'sppas', 'src')
SPPAS_2_PATH = os.path.join('C:\\', SPPAS_2)
SPPAS_2_SRC_PATH = os.path.join(SPPAS_2_PATH, 'sppas', 'src')

SPPAS_GREG_SRC_PATH = os.path.join(PREV_CORPUS_PATH, 'tools')

#  MarsaTag

MARSATAG_PATH = os.path.join(TOOLS_PATH, 'MarsaTag')
MARSATAG_COMMAND = os.path.join(MARSATAG_PATH, 'MarsaTag-UI.bat')

#  pyAudioAnalysis

PYAA_SRC_PATH = os.path.join(TOOLS_PATH, 'pyAudioAnalysis', 'pyAudioAnalysis', 'pyAudioAnalysis')

# Behavioural

FORCE_OW = True

DEFAULT_PHASE_SPLIT = [0.15, 0.70, 0.15]
NO_PHASE_SPLIT = [0, 1, 0]

CURDATE = datetime.now().strftime("%Y-%m%d_%H-%M-%S")

LOGFILE = os.path.join(LOG_PATH, 'predpres-'+CURDATE+'.log')

FEATURES_MATRIX = "matrix-" + CURDATE

STATS_MATRIX = "stats-"+CURDATE
STATS_PRESENCE_MATRIX = "statsPres-"+CURDATE
STATS_COPRESENCE_MATRIX = "statsCopres-"+CURDATE


seed = 12
random.seed(seed)
np.random.seed(seed)