from ast import literal_eval
from os import listdir
from os.path import join

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

DIR = 'results'
# CLASSES = ['Novice', 'Intermediate', 'Expert']
CLASSES = ['0', '1', '2']


WIDTH = 0.0138 * 347.12354
HEIGHT = WIDTH / 3
LETTERING_SIZE = 8
colors = ["#1f77b4", "#30c546", "#8b0072", "#efac35"]
params = {
    'axes.labelsize': LETTERING_SIZE,
    'axes.prop_cycle': cycler(color=colors),
    'figure.constrained_layout.use': True,
    'figure.figsize': [WIDTH, HEIGHT],
    'font.family': 'sans-serif',
    'font.sans-serif': 'Arial',
    'font.size': LETTERING_SIZE,
    'legend.fontsize': LETTERING_SIZE,
    'savefig.dpi': 1200,
    'text.usetex': False,
    'xtick.labelsize': LETTERING_SIZE,
    'ytick.labelsize': LETTERING_SIZE,
}
matplotlib.rcParams.update(params)


def main():
    results = listdir(DIR)
    results.sort()

    print_results(results)
    plot_results(results)


def print_results(results):
    for res in results:
        if 'prob' in res:
            continue
        print(res)
        y_true, y_pred, y_prob = get_results(res)
        names = ['accuracy', 'balanced accuracy']
        metrics = [accuracy_score, balanced_accuracy_score]
        for name, metric in zip(names, metrics):
            print(f'{name}: {metric(y_true, y_pred):.3f}')

        print(f'weighted f1: {f1_score(y_true, y_pred, average="weighted"):.3f}')
        if y_prob:
            print(
                f'auroc: {roc_auc_score(y_true, y_prob, multi_class="ovo", average="weighted"):.3f}'
            )
        print()


def plot_results(results):
    models = {
        'DummyClassifier': 'Majority class',
        'KNeighborsTimeSeriesClassifier': '1-NN DTW',
        'HIVECOTEV2': 'Hive-Cote',
        'RocketClassifier': 'MiniRocket',
        'PETSC': 'Mr-Petsc',
    }
    for obs in [1800, 23400]:
        for names, title in zip([models], ['all']):
            fig, axs = plt.subplots(1, len(names), sharey='all')
            for clf, ax in zip(names.items(), axs):
                plot_cm(obs, clf, ax)
            axs[0].set_ylabel('True label')
            axs[0].set_yticklabels(CLASSES)
            axs[2].set_xlabel('Predicted label')
            plt.savefig(f'cm_{obs}_{title}.png')


def plot_cm(obs, bsl, ax):
    y_true, y_pred, y_prob = get_results(f'{obs}_{bsl[0]}.csv')
    cm = confusion_matrix(y_true, y_pred)
    ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(bsl[1])
    tick_marks = np.arange(len(np.unique(y_true)))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(CLASSES, rotation=0, ha='right')
    ax.set_yticks(tick_marks)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = 'white' if cm[i, j] > thresh else 'black'
            ax.text(j, i, cm[i, j], ha='center', va='center', color=color)


def get_results(fn):
    y_true, y_pred, y_prob = [], [], []
    with open(join(DIR, fn)) as fp:
        for line in fp.readlines():
            row = line[:3].split(',')
            y_true.append(int(row[0]))
            y_pred.append(int(row[1]))
    try:
        _ = fn.find('_') + 1

        with open(join(DIR, fn[:_] + 'prob' + fn[_:])) as fp:
            for line in fp.readlines():
                row = literal_eval(line[2:-1])
                y_prob.append(np.array(row))
    except FileNotFoundError:
        pass
    return y_true, y_pred, y_prob


if __name__ == '__main__':
    main()
