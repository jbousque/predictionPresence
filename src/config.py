"""
A python configuration file containing only variables (properties) declarations.
"""

import os

ROOT_PATH = os.path.join('C:\\', 'IAAA' ,'TER')

# Corpus

CORPUS_PATH = os.path.join(ROOT_PATH, 'DSM', 'ACORFORMED', 'Data')
PREV_CORPUS_PATH = os.path.join(ROOT_PATH, 'Cloud', 'CorpusHMPassation')

# Tools

TOOLS_PATH = os.path.join(ROOT_PATH, 'resources', 'tools')

#  SPPAS

SPPAS_1 = 'SPPAS-1.8.6'
SPPAS_2 = 'SPPAS-2.0-2019-01-08'
SPPAS_PATH = os.path.join(TOOLS_PATH, SPPAS_1 , 'sppas')
SPPAS_SRC_PATH = os.path.join(SPPAS_PATH, 'src')

SPPAS_GREG_SRC_PATH = "/home/sameer/Downloads/Gregoire SPPAS Scripts/" # todo find Greg scripts for SPPAS bootstrap

#  MarsaTag

MARSATAG_PATH = os.path.join(TOOLS_PATH, 'MarsaTag')
MARSATAG_COMMAND = os.path.join(MARSATAG_PATH, 'MarsaTag-UI.bat')

#  pyAudioAnalysis

PYAA_SRC_PATH = os.path.join(TOOLS_PATH, 'pyAudioAnalysis', 'pyAudioAnalysis', 'pyAudioAnalysis')