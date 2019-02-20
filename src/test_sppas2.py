import config
import sys
import os

sys.path.append(config.SPPAS_2_PATH)
from sppas.src.anndata import sppasRW

taggedTransFile = os.path.join(config.PREV_CORPUS_PATH, 'E10D', 'Casque', 'data', 'E10D-01-Casque-micro-palign-marsatag.eaf')
xraFile = os.path.join(config.PREV_CORPUS_PATH, 'E10D', 'Casque', 'data', 'asr-trans', 'E10D-01-Casque-micro.E1-5.xra')

parser = sppasRW(taggedTransFile)
trs = parser.read2()