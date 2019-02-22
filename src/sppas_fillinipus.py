import config
import sys
import os

sys.path.append(config.SPPAS_2_PATH)

from sppas.src.annotations.FillIPUs import sppasFillIPUs


path = 'C:\\IAAA\\TER\\tmp\\E05E\\PC'

filler = sppasFillIPUs()
filler.run(os.path.join(path, 'agent_sound.wav'))