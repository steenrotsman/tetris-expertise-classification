from os import listdir
from os.path import join
import pickle
import random

import numpy as np
from tqdm import tqdm

from petsc.writer import from_lists_to_tsfile

COLS = ['gaze_angle_x', 'gaze_angle_y', 'ear']
K = 5
OBS = 23400
EXCLUDE = 'B152030042'


def main():
    for col in COLS:
        random.seed(0)
        labels = get_labels()
        data = get_data(col, labels)
        parts = split(data, k=K)
        save(parts, col)


def get_data(column, labels=None):
    name = column
    if column == 'ear':
        dir = 'ear'
        column = '0'
    else:
        dir = 'csv'
        column = " " + column
    path = join('data', dir)
    files = listdir(path)
    files.sort()
    try:
        with open(join('data', 'pickled', name), 'rb') as fp:
            data = pickle.load(fp)
    except FileNotFoundError:
        cols = np.loadtxt(join(path, files[0]), delimiter=',', max_rows=1, dtype='str')
        col = np.where(cols == column)[0][0]

        data = []
        for file in tqdm(files):
            if file[:10] == EXCLUDE:
                continue
            row = np.loadtxt(join(path, file), delimiter=',', skiprows=1, usecols=col)
            if OBS:
                row = row[:OBS]
            data.append(row.tolist())

        with open(join('data', 'pickled', f'{OBS}_{name}'), 'wb') as fp:
            pickle.dump(data, fp)

    if labels:
        groups = [[], [], []]
        for file, row in zip(files, data):
            id_ = file[:10]
            if id_ in labels:
                groups[labels[id_]].append(row)
        return groups

    return data


def get_labels():
    file = join('data', 'labels_tetris.csv')
    ids = np.loadtxt(file, delimiter=',', skiprows=1, usecols=1, dtype=str)
    labels = np.loadtxt(file, delimiter=',', skiprows=1, usecols=2, dtype=int)
    return {id: label for id, label in zip(ids, labels)}


def split(data, k=2):
    # Each row corresponds to one of the three labels
    parts = [[[], [], []] for i in range(k)]
    for i, rows in enumerate(data):
        random.shuffle(rows)
        part = 0
        for row in rows:
            parts[part][i].append(row)
            part += 1
            if part == k:
                part = 0

    return parts


def save(parts, name):
    # Save the parts into K train and tests sets
    for i, part in enumerate(parts):
        train = [[], [], []]
        for j, part2 in enumerate(parts):
            if i == j:
                test = part2
            else:
                for k, rows in enumerate(part2):
                    for row in rows:
                        train[k].append(row)

        for temp, t in zip([train, test], ['TRAIN', 'TEST']):
            from_lists_to_tsfile(f'data/ts/{OBS}_{name}_{i}_{K-1}_{t}.ts', temp, [],
                                 'True 0 1 2', description="Tetris",
                                 problem_name=f'tetris_{name}',
                                 time_stamps=False, missing=False, univariate=True, equal_length=True)


if __name__ == '__main__':
    main()
