from os.path import join
from time import perf_counter

import numpy as np
from aeon.classification import DummyClassifier
from aeon.classification.convolution_based import RocketClassifier
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.classification.hybrid import HIVECOTEV2
from aeon.datasets import load_from_tsfile

from petsc.petsc import PETSC

PATH = join('data', 'ts')
COLS = ['gaze_angle_x', 'gaze_angle_y', 'ear']
OBS = 23400
K = 5


def main():
    petsc()
    classify(RocketClassifier(rocket_transform="minirocket", n_jobs=-1))
    classify(HIVECOTEV2(n_jobs=-1))
    classify(DummyClassifier())
    classify(KNeighborsTimeSeriesClassifier(n_jobs=-1))


def petsc():
    print('PETSC')
    y_tests = []
    y_preds = []
    y_probs = []
    time = 0
    for i in range(K):
        train_filenames = [
            join(PATH, f'{OBS}_{col}_{i}_{K-1}_TRAIN.ts') for col in COLS
        ]
        test_filenames = [join(PATH, f'{OBS}_{col}_{i}_{K-1}_TEST.ts') for col in COLS]
        _, y_test = load_from_tsfile(test_filenames[0])

        start = perf_counter()
        petsc = PETSC(train_filenames, test_filenames, stride=4, multiresolution=True)
        y_pred, y_prob = petsc.petsc(prob=True)
        end = perf_counter()
        time += end - start

        y_tests.append(y_test)
        y_preds.append(y_pred)
        y_probs.append(y_prob)

    print(time)
    save(type(petsc).__name__, y_tests, y_preds)
    save('prob' + type(petsc).__name__, y_tests, y_probs)


def classify(clsf):
    print(str(type(clsf)))
    y_tests = []
    y_preds = []
    y_probs = []
    time = 0
    for i in range(K):
        X_train, y_train, X_test, y_test = get_multivariate_data(i)
        start = perf_counter()
        clsf.fit(X_train, y_train)
        y_pred = clsf.predict(X_test)
        y_prob = clsf.predict_proba(X_test)
        end = perf_counter()
        time += end - start

        y_tests.append(y_test)
        y_preds.append(y_pred)
        y_probs.append(y_prob)

    print(time)
    save(type(clsf).__name__, y_tests, y_preds)
    save('prob' + type(clsf).__name__, y_tests, y_probs)


def get_multivariate_data(fold):
    # Make list with for each col a list with an array for each participant
    X_train, y_train, X_test, y_test = [], [], [], []
    for col in COLS:
        _X_train, y_train = load_from_tsfile(
            join(PATH, f'{OBS}_{col}_{fold}_{K-1}_TRAIN.ts')
        )
        _X_test, y_test = load_from_tsfile(
            join(PATH, f'{OBS}_{col}_{fold}_{K-1}_TEST.ts')
        )
        X_train.append(_X_train)
        X_test.append(_X_test)

    # Change to list with 2D array for each participant
    X_train = transform(X_train)
    X_test = transform(X_test)

    return X_train, y_train, X_test, y_test


def transform(X):
    X_transformed = X[0].tolist()

    for var in X[1:]:
        for i, row in enumerate(var):
            X_transformed[i] = np.vstack((X_transformed[i], row))

    X_transformed = np.array([row[:, :OBS] for row in X_transformed])

    return X_transformed


def save(name, y_tests, y_preds):
    with open(join('results', f'{OBS}_{name}.csv'), 'w') as fp:
        for y_test, y_pred in zip(y_tests, y_preds):
            for test, pred in zip(y_test, y_pred):
                try:
                    fp.write(f'{test},{pred.tolist()}\n')
                except SyntaxError:
                    fp.write(f'{test},{pred}\n')
    return


if __name__ == '__main__':
    main()
