"""
A python configuration file containing only variables (properties) declarations.
"""

import os

ROOT_PATH = os.path.join('C:\\', 'IAAA' ,'TER')

DATA_PATH = os.path.join(ROOT_PATH, 'DSM', 'ACORFORMED', 'Data')

SPPAS_1 = 'SPPAS-1.8.6'
SPPAS_2 = 'SPPAS-2.0-2019-01-08'
SPPAS_PATH = os.path.join(ROOT_PATH, 'resources', 'tools', SPPAS_1 , 'sppas')
SPPAS_SRC_PATH = os.path.join(SPPAS_PATH, 'src')

PYAA_SRC_PATH = os.path.join('C:\\', 'IAAA', 'TER', 'resources', 'tools', 'pyAudioAnalysis', 'pyAudioAnalysis', 'pyAudioAnalysis')