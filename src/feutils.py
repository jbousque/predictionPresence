import re
import logging
import os
import pickle
import pandas as pd
import numpy as np

import config

logger = logging.getLogger(__name__)

class FEUtils():
    """

    """

    all_samples_ids = set()

    def __init__(self):
        self.all_samples_ids = set()

    def get_intervals(self, duration, split_ratios):
        prev_split_point = 0
        tuples_array = []
        for point in np.cumsum(split_ratios[:-1]):
            split_point = point * duration
            tuples_array.append((prev_split_point, split_point))
            prev_split_point = split_point
        tuples_array.append((split_point, duration))
        return pd.IntervalIndex.from_tuples(tuples_array)

    def extract_info(self, path):
        """

        :param path:
        :return:
        """
        mode = None
        subject = None
        m = re.search(r'([EN][\d]{1,2}[ABCDEF])[\\/](Casque|PC|Cave)[\\/]', path)
        if m is None:
            m = re.search(r'([EN][\d]{1,2}[ABCDEF])[\\/](Casque|PC|Cave)$', path)
        if m is not None:
            mode = m.group(2)
            subject = m.group(1)
        return subject, mode

    def get_featureset_folder_name(self, isSubject, phasesSplit):
        """
        Formats and returns name of feature folder for given featureset.
        :param isSubject: True if medic, False if agent (greta)
        :param phasesSplit: split ratios for phases, or None.
        :return: formatted folder name for featureset.
        """
        #logger.debug('getFeaturesetFolderName(isSubject=%s, phasesSplit=%s)', isSubject, phasesSplit)
        featuresetName = 'Features'
        if not isSubject:
            featuresetName += '-agent'
        if phasesSplit is None or (phasesSplit[0] == 0 and phasesSplit[2] == 0):
            featuresetName += '-nophase'
        else:
            featuresetName += '-%d%d%d' % ( phasesSplit[0]*100, phasesSplit[1]*100, phasesSplit[2]*100 )
        #logger.debug('getFeaturesetFolderName returns %s', featuresetName)
        return featuresetName

    def get_filtered_file_paths(self, target_candidate=None, target_env=None):
        """

        :param target_candidate:
        :param target_env:
        :return:
        """

        # The function collects 4 files for each sample from the two sources in the paths below. The 4 files are: unity coordinates, xra transcription, wav participant mic audio, and the mp4 extracted from the video. It returns an array of arrays. Each outer array corresponds to a sample (a participant-environment combination) and each inner array contains four paths, one corresponding to each of the mentioned files. The output of this function is used by the functions which compute entropies, IPU lengths, sentence lengths, and POS tags.

        outerArr = []
        logger.info('filePaths: filter_samples %s', target_candidate)

        for root, dirs, files in os.walk(config.PREV_CORPUS_PATH, topdown=True, followlinks=False, onerror=None):
            if root.count(os.sep) - config.PREV_CORPUS_PATH.count(os.sep) >= 3:
                del dirs[:]
            else:
                subject, mode = self.extract_info(root)
                if (target_candidate is None or any(subject == filt for filt in target_candidate)) and (
                        target_env is None or target_env[target_candidate == subject] == mode):
                    self.all_samples_ids.add((subject, mode))
        logger.debug("filePaths: All samples ids %s" % str(self.all_samples_ids))

        for subdir in os.listdir(config.PREV_CORPUS_PATH):
            if os.path.isdir(os.path.join(config.PREV_CORPUS_PATH, subdir))\
                    and (target_candidate is None or any(subdir == filt for filt in target_candidate)):
                for envDir in os.listdir(os.path.join(config.PREV_CORPUS_PATH, subdir)):
                    # print envDir
                    innerArr = []
                    foundWav = False

                    if os.path.isdir(os.path.join(config.PREV_CORPUS_PATH, subdir, envDir)) and \
                            os.path.isdir(os.path.join(config.CORPUS_PATH, subdir, envDir)) and \
                            (target_env is None or target_env[target_candidate == subdir] == envDir):
                        # print os.path.join(subdir, envDir)
                        for dirs, subdirs, files in os.walk(os.path.join(config.PREV_CORPUS_PATH, subdir, envDir), topdown=True,
                                                            onerror=None, followlinks=False):
                            #logger.debug("DEBUG os.walk dirs=%s, subdirs=%s, files=%s", dirs, subdirs, files)
                            for file in files:
                                name, exten = os.path.splitext(file)

                                if (os.path.basename(os.path.normpath(dirs)) == 'data') and (exten == ".wav"):
                                    innerArr.append(os.path.join(dirs, file))
                                    foundWav = True
                                if os.path.basename(os.path.normcase(dirs)) == 'asr-trans' and exten == '.xra':
                                    innerArr.append(os.path.join(dirs, file))

                        for dirs, subdirs, files in os.walk(os.path.join(config.CORPUS_PATH, subdir, envDir), topdown=True,
                                                            onerror=None, followlinks=False):
                            for file in files:
                                name, exten = os.path.splitext(file)

                                if (os.path.basename(os.path.normpath(dirs)) == 'Video') and (exten == ".mp4"):
                                    # print os.path.join(dirs, file)
                                    innerArr.append(os.path.join(dirs, file))
                                if (os.path.basename(os.path.normpath(dirs)) == 'Unity') and (exten == ".txt"):
                                    # print os.path.join(dirs, file)
                                    innerArr.append(os.path.join(dirs, file))
                                    # if micro HF wav was not found in first corpus, we take the one from second corpus
                                if os.path.basename(os.path.normpath(dirs)).startswith('session') and exten == '.wav':
                                    if not foundWav:
                                        innerArr.append(os.path.join(dirs, file))
                        outerArr.append(innerArr)
        return outerArr

    def get_filtered_file_paths_agent(self, target_candidate=None, target_env=None):
        """

        :param target_env:
        :return:
        """
        # The function collects 4 files for each sample from the two sources in the paths below. The 4 files are: unity coordinates, xra transcription, wav participant mic audio, and the mp4 extracted from the video. It returns an array of arrays. Each outer array corresponds to a sample (a participant-environment combination) and each inner array contains four paths, one corresponding to each of the mentioned files. The output of this function is used by the functions which compute entropies, IPU lengths, sentence lengths, and POS tags.

        outerArr = []
        agent_path = os.path.join(config.TMP_PATH)
        logger.info('filePaths_agent(target_candidate=%s, target_env=%s)' % (target_candidate, target_env))
        for root, dirs, files in os.walk(config.PREV_CORPUS_PATH, topdown=True, followlinks=False, onerror=None):
            if root.count(os.sep) - config.PREV_CORPUS_PATH.count(os.sep) >= 3:
                del dirs[:]
            else:
                subject, mode = self.extract_info(root)
                if (target_candidate is None or any(subject == filt for filt in target_candidate)) and (target_env is None or target_env[target_candidate == subject] == mode):
                    self.all_samples_ids.add((subject, mode))
        logger.debug("filePaths_agent: All samples ids %s" % str(self.all_samples_ids))
        for root, dirs, files in os.walk(agent_path):
            #logger.debug('filePaths_agent: considering %s', root)
            subject, mode = self.extract_info(root)
            #logger.debug('filePaths_agent: extracted %s / %s' % (subject, mode))
            if target_candidate is None or any(subject == filt for filt in target_candidate):
                #logger.debug('filePaths_agent: target_env[target_candidate == subject] %s' % (str(target_env[target_candidate == subject])))
                if target_env is None or target_env[target_candidate == subject] == mode:
                    #m = re.search(r'([EN]\d\d[ABCDEF])[\\/](Casque|PC|Cave)[\\/]', root)
                    #if m is None:
                    m = re.search(r'([EN][\d]{1,2}[ABCDEF])[\\/](Casque|PC|Cave)$', root)
                    if m is not None:
                        mode = m.group(2)
                        subject = m.group(1)
                        logger.debug('filePaths_agent: treating %s / %s', subject, mode)
                        innerArr = []

                        innerArr.append(os.path.join(root, 'agent.xra'))
                        innerArr.append(os.path.join(root, 'agent_sound.wav'))

                        corpus_path = os.path.join(config.CORPUS_PATH, subject, mode, 'Unity')
                        logger.debug('filePaths_agent: looking for out_record under %s', corpus_path)
                        if os.path.exists(corpus_path):
                            for file in os.listdir(corpus_path):
                                name, exten = os.path.splitext(file)
                                if exten == ".txt":
                                    # print os.path.join(dirs, file)
                                    innerArr.append(os.path.join(corpus_path, file))
                                    outerArr.append(innerArr)
                        else: logger.warn('filePaths_agent: path does not exist %s', corpus_path)

        logger.debug('filePaths_agent returns %s', str(outerArr))

        return outerArr


class DataHandler():

    def __init__(self, group=None, exp=None, iter=0):
        self._group = group if group is not None else ''
        self._exp = exp if exp is not None else ''
        self._iter = iter
        self._init_root_path()

    def _init_root_path(self):
        self.root_path = os.path.join(config.OUT_PATH, self._group, '%s-%d' % (self._exp, self._iter))

    def get_grid_name(self, presence=True, doctor=True, agent=False, phases=None, classifier='RF'):
        pres = 'presence' if presence else 'copresence'
        subject = 'doctor' if doctor else ''
        subject = subject + ('agent' if agent else '')
        if phases is None or (phases is not None and type(phases) == tuple):
            if phases in [None, (0, 1, 0)]:
                ph = 'nophase'
            else:
                ph = '%02d%02d%02d' % (phases[0] * 100, phases[1] * 100, phases[2] * 100)
        else:
            ph = phases
        return "%s_%s_%s_%s" % (pres, subject, ph, classifier)

    def save_grid(self, grid, folder, presence=True, doctor=True, agent=False, phases=None, classifier='RF'):
        name = "grid_%s" % (self.get_grid_name(presence, doctor, agent, phases, classifier))
        return self.save_obj(grid, folder, name)

    def load_grid(self, folder, presence=True, doctor=True, agent=False, phases=None, classifier='RF'):
        name = "grid_%s" % (self.get_grid_name(presence, doctor, agent, phases, classifier))
        return self.load_obj(folder, name)

    def save_obj(self, obj, folder, name):
        """
        Saves an object to a file with pickle dump (under root_path folder, under 'folder', with name 'name'.pkl).
        Makes folders accordingly if they do not exist.
        """
        try:
            file_path = os.path.join(self.root_path, folder, name + '.pkl')
            if not os.path.exists(os.path.dirname(file_path)): os.makedirs(os.path.dirname(file_path))
            f_obj = open(file_path, 'wb')
            pickle.dump(obj, f_obj)
            f_obj.close()
        except Exception as e:
            print('Could not save file %s' % (file_path))
            print(e)
            return False
        return True

    def load_obj(self, folder, name):
        try:
            file_path = os.path.join(self.root_path, folder, name + '.pkl')
            print('load_obj: opening %s' % file_path)
            f_obj = open(file_path, 'rb')
            obj = pickle.load(f_obj)
            f_obj.close()
        except Exception:
            print('file does not yet exist %s' % file_path)
            return None
        return obj

    def save_fig(self, plt, title):
        try:
            path = os.path.join(self.root_path, 'figures', title + '.png')
            if not os.path.exists(os.path.dirname(path)): os.makedirs(os.path.dirname(path))
            plt.savefig(path, bbox_inches="tight")
        except Exception as e:
            print(e)
            print('Could not save figure %s' % path)
