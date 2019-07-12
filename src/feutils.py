#from __future__ import division
import re
import logging
import os
import pickle
import pandas as pd
import numpy as np
import subprocess
import hashlib

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO


from sklearn.metrics import label_ranking_average_precision_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score

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

    def get_interval(self, intervals, begin, end):
        return intervals.get_loc((begin + end) / 2)

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

    def compute_mrr(self, y_true, y_pred, labels, verbose=0):
        """
        Compute the mean reciprocal rank (as y_true provides only one label as relevant).
        :param y_true: (nb samples)
        :param y_pred: (nb samples, nb classes)
        :param labels: (nb classes)
        :return: the mean reciprocal rank.
        """
        if verbose > 2:
            print('compute_mrr(y_true %s, y_pred %s, labels %s' % (str(y_true), str(y_pred), str(labels)))
        if type(y_true) is np.ndarray:
            y_true = pd.DataFrame(y_true)
        if type(y_pred) is np.ndarray:
            y_pred = pd.DataFrame(y_pred)
        lb = LabelBinarizer()
        lb.fit(labels)
        if y_pred.to_numpy().ndim == 1:
            y_pred = pd.DataFrame(y_pred.to_numpy()[:, np.newaxis])
        if y_true.to_numpy().ndim == 1:
            y_true = pd.DataFrame(y_true.to_numpy()[:, np.newaxis])
        y_score = np.zeros(y_pred.values.shape)
        #print(y_pred.to_numpy().shape)
        for i in np.arange(y_score.shape[1]):
            col = y_pred.iloc[:, i]
            if len(labels)> 2:
                binarized = lb.transform(col)
            else:
                binarized = np.zeros((len(col), 2))
                #print(binarized.shape)
                for irow in np.arange(len(col)):
                    if col[irow] == 0:
                        binarized[irow, 0] = 1
                    elif col[irow] == 1:
                        binarized[irow, 1] = 1
            y_score = y_score + (binarized.astype(float) / (i+1))
        y_true_bin = lb.transform(y_true.values.reshape(-1,1))
        if len(labels) <= 2:
            y_true_bin_ = np.zeros((len(y_true_bin), 2))
            for irow in np.arange(len(y_true_bin)):
                if y_true_bin[irow] == 0:
                    y_true_bin_[irow, 0] = 1
                else:
                    y_true_bin_[irow, 1] = 1
            y_true_bin = y_true_bin_
        if verbose > 5:
            print('y_true_bin %s y_score %s' % (y_true_bin.shape, y_score.shape))
            print(y_true_bin)
            print(y_score) # TODO check why even when ndim=1, when y_pred is wrong, there is 0,1 or 1,0 in binarized y_pred ???
        return label_ranking_average_precision_score(y_true_bin, y_score)

    def compute_determinacy(self, y_pred):
        if type(y_pred) is pd.DataFrame:
            y_pred = y_pred.to_numpy()
        if y_pred.ndim == 1:
            y_pred = y_pred[:, np.newaxis]
        determ = np.sum(np.sum(y_pred != 6666, axis=1) == 1) / float(len(y_pred))
        return determ

class DataHandler():

    def __init__(self, group=None, exp=None, iter=0):
        self._group = group if group is not None else ''
        self._exp = exp if exp is not None else ''
        self._iter = iter
        self._init_root_path()
        print('DataHandler.__init__(group=%s, exp=%s, iter=%d) -> root_path=%s' % (group, exp, iter, self.root_path))

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


from sklearn.base import BaseEstimator, ClassifierMixin

class JNCC2Wrapper(BaseEstimator, ClassifierMixin):
    """

    """

    logger_ = logging.getLogger(__name__)

    def __init__(self, dataHandler=None, idx=None, verbose=0):
        self.logger_.debug('JNCC2Wrapper(dataHandler=%s)' % str(dataHandler))
        if verbose > 1:
            print('JNCC2Wrapper(dataHandler=%s)' % str(dataHandler))
        self.dataHandler = dataHandler
        if dataHandler is not None:
            self.arff_root_path_ = os.path.join(self.dataHandler.root_path, 'arff')
            if not os.path.exists(os.path.dirname(self.arff_root_path_)): os.makedirs(os.path.dirname(self.arff_root_path_))
        else:
            self.arff_root_path_ = None
        self.idx = idx
        self.X_train_ = None
        self.y_train_ = None
        self.X_test_ = None
        self.y_test_ = None
        self.verbose = verbose
        self.logger_.debug('JNCC2Wrapper: initialized arff_root_path_ to %s' % self.arff_root_path_)

    def _generate_arff(self, X, y, file_name, features=None):
        """
        Generates .arff file format required by JNCC2 java-based classifier.

        :param X:
        :param y: optional
        :param file_name:
        :param features: optional (then generated as x0, x1, ..., or retrieved from columns if X is a DataFrame)
        :return:
        """
        if self.verbose > 1:
            self.logger_.debug('generate_arff(file_name=%s, features=%s, X shape=%s)'
                               % (file_name, str(features), str(X.shape)))
            print('generate_arff(fname=%s, features=%s, X shape=%s'
                  % (file_name, str(features), str(X.shape)))

        # create path if it does not exist
        if not os.path.exists(os.path.dirname(file_name)): os.makedirs(os.path.dirname(file_name))

        if features is None or len(features) == 0:
            if isinstance(X, pd.DataFrame):
                features = list(X.columns)
            else:
                features = ['x%d' % i for i in np.arange(X.shape[1])]
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=features)



        text_file = open(file_name + '.arff', "w")
        text_file.write("@relation %s\n" % file_name)
        for feature in features:
            if feature == 'Expert' or set(np.unique(X[feature])).issubset({0,1}):
                text_file.write("@attribute %s {0,1}\n" % feature.lower().replace('_', ''))
            else:
                text_file.write("@attribute %s numeric\n" % feature.lower().replace('_', ''))
        classes_str = '{'
        #classes_dict = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
        if y is not None:
            for idx, cl in enumerate(np.unique(y)):
                classes_str += str(int(cl))
                if idx < len(np.unique(y)) - 1:
                    classes_str += ','
            classes_str += '}'
            text_file.write('@attribute class %s\n' % classes_str)
        text_file.write("@DATA\n")
        Xarr = np.array(X)
        if y is None:
            Yarr = [np.nan]*Xarr.shape[0]
        else:
            Yarr = np.array(y)
        #print('XArr %s, Yarr %s' % (str(Xarr), str(Yarr)))
        for idx, (x_, y_) in enumerate(zip(Xarr, Yarr)):
            #print('   x_ %s, y_ %s' % (str(x_), str(y_)))
            line = ''
            for idx, feature in enumerate(features):
                if feature == 'Expert' or set(np.unique(X[feature])).issubset({0,1}):
                    line += str(int(x_[idx])) + ','
                else:
                    line += '%f,' % x_[idx]   #str(x_[idx]) + ','
            if y is not None:
                line += str(int(y_))
            text_file.write(line+'\n')
            """text_file.write("%d,%s,%d\n" % (int(x_[expertIndex]),
                                            ','.join(str(x__) for x__ in x_[1:]),
                                            y_))"""

        text_file.close()

    def _cv(self, train_arff_file, idx=None):
        if self.verbose > 1:
            self.logger_.debug('cv(train_arff_file=%s)' % train_arff_file)
            print('cv(train_arff_file=%s, idx=%s)' % (train_arff_file, str(idx)))
        if idx is not None:
            path = os.path.join(self.arff_root_path_, str(idx))
        else:
            path = self.arff_root_path_
        cmd = ['java', '-jar', config.JNCC2_JAR, path, train_arff_file, 'cv']
        if self.verbose > 2:
            self.logger_.debug('cv: Executing' + subprocess.list2cmdline(cmd))
            print("cv: Executing " + subprocess.list2cmdline(cmd))
        output = subprocess.check_output(cmd)
        if self.verbose > 2:
            self.logger_.info(output)
            print(output)

        # load result
        rfile = os.path.join(path, 'Predictions-CV-train.csv')
        if os.path.isfile(rfile):
            rdf = pd.read_csv(rfile, header=None)
            # first column is id of cv, 2nd column is actual, last column is NBC - rest are NCC predictions
            prev = -1
            real_id = []
            curr = -1
            for i in np.arange(len(rdf)):
                val = rdf.iloc[i,0]
                #print('cv: rdf id #%d value %d' % (i, rdf.iloc[i,0]))
                if val != prev:
                    curr += 1
                    prev = val
                real_id.append(curr)
            rdf['real_id'] = real_id
            res = []
            y_trues = []
            for i in np.arange(curr):
                res.append(rdf[rdf['real_id'] == i].iloc(axis=1)[2:-2].values)
                y_trues.append(rdf[rdf['real_id'] == i].iloc(axis=1)[1].values)
            #res = rdf.iloc(axis=1)[0:-1]
            self.logger_.debug('cv: return %s x %s' % (str(len(res)), str(res[0].shape)))
            if self.verbose > 1:
                print('cv: return %s x %s' % (str(len(res)), str(res[0].shape)))
            return res, y_trues


    def _predict(self, train_arff_file, test_arff_file, idx=None):
        if self.verbose > 1:
            self.logger_.debug('predict(train_arff_file=%s, test_arff_file=%s)' % (train_arff_file, test_arff_file))
            print('predict(train_arff_file=%s, test_arff_file=%s, idx=%s)' % (train_arff_file, test_arff_file, str(idx)))

        # is this a complete path or only a file name ?
        if os.sep in train_arff_file:
            if idx is not None:
                path = os.path.join(os.path.dirname(train_arff_file), str(idx))
            else:
                path = os.path.dirname(train_arff_file)
        else:
            path = self.arff_root_path_
        cmd = ['java', '-jar', config.JNCC2_JAR, path, os.path.basename(train_arff_file), os.path.basename(test_arff_file)]
        if self.verbose > 2:
            self.logger_.debug("predict: Executing " + subprocess.list2cmdline(cmd))
            print("predict: Executing " + subprocess.list2cmdline(cmd))
        output = subprocess.check_output(cmd)
        if self.verbose > 2:
            self.logger_.info(output)
            print(output)

        # load result
        rfile = os.path.join(path, 'Predictions-Testing-test.csv')
        if os.path.isfile(rfile):
            rdf = pd.read_csv(rfile, header=None)
            # first column is id, 2nd column is actual, last column is NBC - rest are NCC predictions
            #print(rdf)
            res = rdf.iloc(axis=1)[2:-1]
            y_true = rdf.iloc(axis=1)[1]
            if self.verbose > 1:
                self.logger_.debug('predict: return %s' % str(res.shape))
                print('predict: return %s' % str(res.shape))
            return res

    def _predict_unknownclasses(self, train_arff_file, test_arff_file):
        if self.verbose > 1:
            self.logger_.debug('_predict_unknownclasses(train_arff_file=%s, test_arff_file=%s)' % (train_arff_file, test_arff_file))
            print('_predict_unknownclasses(train_arff_file=%s, test_arff_file=%s)' % (train_arff_file, test_arff_file))

        # is this a complete path or only a file name ?
        if os.sep in train_arff_file:
            path = os.path.dirname(train_arff_file)
        else:
            path = self.arff_root_path_
        cmd = ['java', '-jar', config.JNCC2_JAR, path, os.path.basename(train_arff_file),
               os.path.basename(test_arff_file), 'unknownclasses']
        if self.verbose > 2:
            self.logger_.debug("predict: Executing " + subprocess.list2cmdline(cmd))
            print("predict: Executing " + subprocess.list2cmdline(cmd))
        output = subprocess.check_output(cmd)
        if self.verbose > 2:
            self.logger_.info(output)
            print(output)

        # load result
        # note: csv file format is different in this case for unknown reason ... We must first remove heading lines
        # (wich are not blank or commented, so we will try to find pattern 'Ncc2Prediction' in a line)
        # also in this case only 1 class is proposed (again, no reason why ...)
        rfile = os.path.join(path, 'Predictions-Testing-test.csv')
        if os.path.isfile(rfile):

            # find the heading lines to find starting row
            frfile = open(rfile, 'r')
            lines = frfile.readlines()
            nccpred_column = None
            for idx, line in enumerate(lines):
                if 'Ncc2Prediction' in line:
                    #print('line ' + str(line))
                    #print('splitted line ' + str(line.split(',')))
                    splitted = [item.replace('\n','') for item in line.split(',')]
                    nccpred_column = splitted.index('Ncc2Prediction')
                    #print('Pred starting column %d' % nccpred_column)
                    break
            csv_string = "".join(lines[idx+2:len(lines)])
            if self.verbose > 5:
                print(csv_string)
            rdf = pd.read_csv(StringIO(csv_string), header=None)
            if self.verbose > 5:
                print(rdf)
            result = rdf.iloc(axis=1)[nccpred_column:-1]
            result[result == '-'] = np.nan
            #result = result.dropna(axis='columns', how='any')
            result = result.iloc(axis=1)[0]
            if self.verbose > 1:
                print('JNCC2Wrapper._predict_unknownclasses return:')
                print(result)

            return result
        if self.verbose > 1:
            print('JNCC2Wrapper._predict_unknownclasses return None')
        return None

    def fit(self, X, y=None):
        """
        fit does nothing except storing train data.
        :param X:
        :param y:
        :return:
        """
        if self.verbose > 1:
            print('JNCC2Wrapper.fit(X=%s, y=%s)' % (X.to_numpy().shape if isinstance(X, pd.DataFrame) else X.shape,
                                                y.shape if y is not None else 'None'))
        m = hashlib.md5()
        if y is None:
            arr = np.array(X)
        else:
            arr = np.hstack((np.array(X), np.array(y).reshape(-1, 1)))
        m.update(str(arr).encode('utf-8'))
        hash = m.hexdigest()
        if self.arff_root_path_ is None:
            self.__init__(self.dataHandler, self.idx)
        if self.verbose > 2:
            print('JNCC2Wrapper.fit: root_path %s, hash %s' % (self.arff_root_path_, str(hash)))
        self.arff_train_file_ = os.path.join(self.arff_root_path_, hash, 'train')
        if not os.path.isfile(self.arff_train_file_):
            self._generate_arff(X, y, self.arff_train_file_)
        self.X_train_ = X
        self.y_train_ = y
        return self

    def predict(self, X, y=None):
        """

        :param X:
        :param y:
        :return:
        """
        if self.X_train_ is None:
            logger.error('JNCC2Wrapper.predict: not fitted !')
            print('JNCC2Wrapper.predict: not fitted !')
        if self.verbose > 1:
            print('JNCC2Wrapper.predict(X=%s, y=%s)' % (X.to_numpy().shape if isinstance(X, pd.DataFrame) else X.shape,
                                                y.shape if y is not None else 'None'))
        self.arff_test_file_ = os.path.join(os.path.dirname(self.arff_train_file_), 'test')
        if os.path.isfile(self.arff_test_file_+'.arff'):
            if self.verbose > 2:
                print('JNCC2Wrapper.predict: removing %s' % (self.arff_test_file_+'.arff'))
            os.remove(self.arff_test_file_ + '.arff')
        self._generate_arff(X, y, self.arff_test_file_)
        self.X_test_ = X
        self.y_test_ = y
        if y is not None:
            res = self._predict(self.arff_train_file_+'.arff', self.arff_test_file_+'.arff')
        else:
            res = self._predict_unknownclasses(self.arff_train_file_+'.arff', self.arff_test_file_+'.arff')
        self.predict_result_ = res
        return res

    def score(self, X, y=None):
        """
        Returns f1 macro (weighted) score for X and y.
        :param X:
        :param y:
        :return:
        """
        if self.verbose > 1:
            print('JNCC2Wrapper.score(X=%s, y=%s)' % (X.to_numpy().shape if isinstance(X, pd.DataFrame) else X.shape,
                                                y.shape if y is not None else 'None'))
        res = self.predict(X, y)
        y_true = y
        return f1_score(y_true, res.iloc(axis=1)[0], average='weighted')

