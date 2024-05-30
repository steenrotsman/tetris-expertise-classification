from uuid import uuid1
from subprocess import run

import pandas as pd
from aeon.datasets import load_from_tsfile_to_dataframe
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from .load import clear

CP = '../petsc/petsc-0.9.0-jar-with-dependencies.jar'
MAX_HEAP_SIZE = '-Xmx12G'


class PETSC:
    def __init__(self, input, input_test, bins=4, paa_win=15, window=0, stride=1, k=200, max_size=None, min_size=5, duration=1.1, sort_alpha=None, soft=None, tau=None, discriminative=False, multiresolution=False, output_patterns=None):
        self.input = input
        self.input_test = input_test
        self.output = f'temp/{uuid1()}.txt'
        self.output_test = f'temp/{uuid1()}_test.txt'
        self.bins = bins
        self.paa_win = paa_win
        self.window = window
        self.stride = stride
        self.k = k
        self.max_size = max_size if max_size is not None else paa_win
        self.min_size = min_size
        self.duration = duration
        self.sort_alpha = sort_alpha
        self.soft = soft
        self.tau = tau
        self.discriminative = discriminative
        self.multiresolution = multiresolution
        self.output_patterns = output_patterns

        self.miner = 'MineTopKDiscriminativeSequentialPatternsNew' if discriminative else 'MineTopKSequentialPatterns'

    def petsc(self, prob=False):
        if isinstance(self.input, list):
            X_train, y_train = load_from_tsfile_to_dataframe(self.input[0])
        else:
            X_train, y_train = load_from_tsfile_to_dataframe(self.input)
        window = min(X_train.var_0.str.len()) // 2

        if isinstance(self.input, list):
            X_embtrain, X_embtest = self.mine_mr_multivariate(window)
        elif self.multiresolution:
            X_embtrain, X_embtest = self.mine_mr(window)
        else:
            X_embtrain, X_embtest = self.mine()
        clear(self.output, self.output_test)

        clf = make_pipeline(StandardScaler(), SGDClassifier(loss='log_loss', penalty='elasticnet', n_jobs=-1))
        clf.fit(X_embtrain, y_train)
        y_pred = clf.predict(X_embtest)
        if prob:
            y_prob = clf.predict_proba(X_embtest)
            return y_pred, y_prob
        return y_pred

    def mine_mr(self, window):
        X_embtrain = []
        X_embtest = []

        self.window = window
        while self.window > self.paa_win:
            X_embtrain_, X_embtest_ = self.mine()
            X_embtrain.append(X_embtrain_)
            X_embtest.append(X_embtest_)
            self.window = self.window // 2

        X_embtrain = pd.concat(X_embtrain, axis=1)
        X_embtest = pd.concat(X_embtest, axis=1)

        return X_embtrain, X_embtest

    def mine(self):
        cmd = ['java', MAX_HEAP_SIZE, '-cp', CP, f'be.uantwerpen.mining.{self.miner}']
        for flag, value in vars(self).items():
            if value is not None and flag not in ['miner', 'discriminative', 'multiresolution']:
                cmd += [f'-{flag}', str(value)]

        # Subprocess returns 0 on success and 1 on error
        if run(cmd, capture_output=True).returncode:
            raise ChildProcessError

        X_embtrain = pd.read_csv(self.output, header=None)
        X_embtest = pd.read_csv(self.output_test, header=None)

        clear(self.output, self.output_test)

        return X_embtrain, X_embtest

    def mine_mr_multivariate(self, window):
        input = self.input
        input_test = self.input_test

        # Embedding
        X_embtrain = []
        X_embtest = []
        for dimension, (train_filename, test_filename) in enumerate(zip(input, input_test)):
            self.input = train_filename
            self.input_test = test_filename
            X_embtrain_, X_embtest_ = self.mine_mr(window)
            X_embtrain.append(X_embtrain_)
            X_embtest.append(X_embtest_)

        # Concat
        train_concat = pd.concat(X_embtrain, axis=1)
        test_concat = pd.concat(X_embtest, axis=1)

        return train_concat, test_concat
