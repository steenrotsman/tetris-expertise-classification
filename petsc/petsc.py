from subprocess import run
from uuid import uuid1

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aeon.datasets import load_from_tsfile
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from plot import LABELS, remove_spines

from .load import clear

CP = 'petsc/petsc-0.9.0-jar-with-dependencies.jar'
MAX_HEAP_SIZE = '-Xmx32G'
EXCLUDE = ['miner', 'discriminative', 'multiresolution']


class PETSC:
    def __init__(
        self,
        input,
        input_test,
        bins=4,
        paa_win=15,
        window=0,
        stride=1,
        k=200,
        max_size=None,
        min_size=5,
        duration=1.1,
        sort_alpha=None,
        soft=None,
        tau=None,
        discriminative=False,
        multiresolution=False,
        output_patterns=None,
        output_discretisation_train=None,
        output_discretisation_test=None,
    ):
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
        self.output_discretisation_train = output_discretisation_train
        self.output_discretisation_test = output_discretisation_test

        self.miner = (
            'MineTopKDiscriminativeSequentialPatternsNew'
            if discriminative
            else 'MineTopKSequentialPatterns'
        )

        self.window_ = None
        self.X_embtrain_ = None
        self.y_train_ = None
        self.X_embtest_ = None
        self.output_patterns_ = None
        self.output_discretisation_train_ = None
        self.output_discretisation_test_ = None

    def petsc(self, prob=False):
        if isinstance(self.input, list):
            X_train, y_train = load_from_tsfile(self.input[0])
        else:
            X_train, y_train = load_from_tsfile(self.input)

        self.window_ = X_train[0].shape[1] // 2

        self.y_train_ = y_train
        self.make_embeddings()

        clf = make_pipeline(
            StandardScaler(),
            SGDClassifier(loss='log_loss', penalty='elasticnet', n_jobs=-1),
        )
        clf.fit(self.X_embtrain_, y_train)
        y_pred = clf.predict(self.X_embtest_)
        if prob:
            y_prob = clf.predict_proba(self.X_embtest_)
            return y_pred, y_prob
        return y_pred

    def make_embeddings(self):
        if isinstance(self.input, list):
            miner = self.mine_mr if self.multiresolution else self.mine
            self.mine_multivariate(miner)
        elif self.multiresolution:
            self.mine_mr()
        else:
            self.mine()
        clear(self.output, self.output_test)

    def mine(self):
        cmd = ['java', MAX_HEAP_SIZE, '-cp', CP, f'be.uantwerpen.mining.{self.miner}']
        for flag, value in vars(self).items():
            if value is not None and flag not in EXCLUDE and flag[-1] != '_':
                cmd += [f'-{flag}', str(value)]

        run(cmd, check=True)

        self.X_embtrain_ = pd.read_csv(self.output, header=None)
        self.X_embtest_ = pd.read_csv(self.output_test, header=None)

    def mine_mr(self):
        X_embtrain = []
        X_embtest = []

        self.window = self.window_
        while self.window > self.paa_win:
            self.mine()
            X_embtrain.append(self.X_embtrain_)
            X_embtest.append(self.X_embtest_)
            self.window = self.window // 2

        self.X_embtrain_ = pd.concat(X_embtrain, axis=1)
        self.X_embtest_ = pd.concat(X_embtest, axis=1)

    def mine_multivariate(self, miner):
        self.input_ = self.input
        self.input_test_ = self.input_test
        self.output_patterns_ = self.output_patterns
        self.output_discretisation_train_ = self.output_discretisation_train
        self.output_discretisation_test_ = self.output_discretisation_test

        X_embtrain = []
        X_embtest = []
        x = enumerate(zip(self.input_, self.input_test_))
        for dim, (train_filename, test_filename) in x:
            self.input = train_filename
            self.input_test = test_filename
            if self.output_patterns is not None:
                self.output_patterns = self.output_patterns_ + str(dim)

            if self.output_discretisation_train is not None:
                self.output_discretisation_train = (
                    self.output_discretisation_train_ + str(dim)
                )
            if self.output_discretisation_test is not None:
                self.output_discretisation_test = (
                    self.output_discretisation_test_ + str(dim)
                )
            miner()
            X_embtrain.append(self.X_embtrain_)
            X_embtest.append(self.X_embtest_)

        self.X_embtrain_ = pd.concat(X_embtrain, axis=1)
        self.X_embtest_ = pd.concat(X_embtest, axis=1)

    def show_attribution(self):
        # Refit classifier without standard scaling for interpretation
        clf = SGDClassifier(
            tol=1e-3,
            alpha=1,
            penalty='elasticnet',
            n_jobs=-1,
            loss='log_loss',
            random_state=0,
        )
        clf.fit(self.X_embtrain_, self.y_train_)

        # Match coefficients to pattern occurrences
        window_data_test = parse_discrete_file(self.output_discretisation_test)
        coef_occ_by_dim = []
        coefficients = clf.coef_[2]
        coefficients /= max(coefficients.max(), coefficients.min() * -1)

        # Filter to only have top 10% largest (pos or neg) weights
        min_coef_val = sorted(coefficients, key=abs)[-len(coefficients) // 20]

        for dim in range(3):
            coef_occ = []
            patterns = parse_patterns(self.output_patterns_ + str(dim))

            coefs = coefficients[self.k * dim : self.k * (dim + 1)]
            for pattern, coef in zip(patterns, coefs):
                if abs(coef) >= min_coef_val:
                    length = int(len(pattern) * self.window / self.paa_win)
                    occ = get_occurrences(pattern, window_data_test)
                    coef_occ.append((coef, length, occ))
            coef_occ_by_dim.append(coef_occ)
        coef_occ_by_dim

        # Load data in (participants, channels, frames) shape
        X_test_dims = []
        for fname in self.input_test_:
            X_test, y_test = load_from_tsfile(fname)
            X_test_dims.append(X_test)
        X_test_dims = np.stack(X_test_dims).squeeze().transpose((1, 0, 2))

        # Create plot for each participant
        for participant, data in enumerate(X_test_dims):
            fix, axs = plt.subplots(3, sharex='all')
            for channel, coef_occ, ax, label in zip(data, coef_occ_by_dim, axs, LABELS):
                ax.plot(channel, c='k', lw=0.5)

                # Pattern attribution
                for coef, length, occurrences in coef_occ:
                    pattern_indices = occurrences[participant]
                    color = 'blue' if coef > 0 else 'red'
                    linewidth = 1 + abs(coef)
                    for idx in pattern_indices:
                        ax.plot(
                            range(idx, idx + len(channel[idx : idx + length])),
                            channel[idx : idx + length],
                            c=color,
                            lw=linewidth,
                        )

                yticks = [round(min(channel), 2), round(max(channel), 2)]
                ax.set(ylabel=label, yticks=yticks, xlim=[0, 1800])
                remove_spines(ax)
            ax.set_xlabel('Frames')

            plt.savefig(f'figs/{participant}_{y_test[participant]}.png')


def parse_patterns(fname):
    patterns = []
    with open(fname) as fp:
        for line in fp:
            if line.strip() == '':
                continue
            pattern, support = line.strip().split(';')
            pattern = [int(x) for x in pattern.split(',')]
            patterns.append((pattern))
    print(f'Parsed {fname}, length: {len(patterns)}, first: {patterns[0]}')
    return patterns


def parse_discrete_file(fname):
    data = []
    with open(fname) as fp:
        for line in fp:
            if line.strip() == '':
                continue
            ts, label = line.strip().split(':')
            label = int(label)
            while len(data) < label + 1:
                data.append([])
            data[label].append(np.array(ts.split(','), dtype='int'))
    return data


def get_occurrences(pattern, window_data):
    occurrences = []
    for windows in window_data:
        occ = []
        for i, window in enumerate(windows):
            if match_seq_pattern_faster(pattern, window) != -1:
                occ.append(i)
        occurrences.append(occ)
    return occurrences


def match_seq_pattern_faster(pattern, window_ts):
    window_ts = window_ts.tolist()
    for i in range(len(window_ts) - len(pattern) + 1):
        if window_ts[i : i + len(pattern)] == pattern:
            return i
    return -1
