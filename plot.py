from os.path import join

import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler

LABELS = ['gaze_angle_x', 'gaze_angle_y', 'eye_aspect_ratio']
WIDTH = 0.0138 * 372
HEIGHT = WIDTH / 3 * 2
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
    'savefig.dpi': 2400,
    'text.usetex': False,
    'xtick.labelsize': LETTERING_SIZE,
    'ytick.labelsize': LETTERING_SIZE,
}
matplotlib.rcParams.update(params)


def remove_spines(ax, remove_y=False):
    if remove_y:
        ax.axes.get_yaxis().set_visible(False)
        ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


if __name__ == '__main__':
    from train import COLS, get_multivariate_data

    X_train, y_train, X_test, y_test = get_multivariate_data(0)

    fig, axs = plt.subplots(3, sharex='all')

    for i, ax, label in zip(range(3), axs, LABELS):
        row = X_train[10, i, 1000:1300]
        ax.plot(row)
        ax.set(ylabel=label, yticks=[round(min(row), 2), round(max(row), 2)])
        remove_spines(ax)
    ax.set_xlabel('Frames')
    fig.align_labels()
    plt.savefig('figs/signal.png')
