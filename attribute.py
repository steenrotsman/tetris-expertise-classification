from os.path import join

from aeon.datasets import load_from_tsfile

from petsc.petsc import PETSC

PATH = join('data', 'ts')
COLS = ['gaze_angle_x', 'gaze_angle_y', 'ear']
OBS = 23400
K = 5


train_filenames = [join(PATH, f'{OBS}_{col}_0_4_TRAIN.ts') for col in COLS]
test_filenames = [join(PATH, f'{OBS}_{col}_0_4_TEST.ts') for col in COLS]
_, y_test = load_from_tsfile(test_filenames[0])
window = 32
kwargs = {
    'paa_win': 8,
    'bins': 7,
    'window': window,
    'output_patterns': 'temp/petsc_patterns',
    'output_discretisation_train': f'temp/mr_petsc_windows_train_{window}.txt',
    'output_discretisation_test': f'temp/mr_petsc_windows_test_{window}.txt',
}

petsc = PETSC(train_filenames, test_filenames, **kwargs)
petsc.petsc()
petsc.show_attribution()
