import os
from subprocess import run
import uuid

import pandas as pd
from sktime.datasets import load_from_tsfile_to_dataframe
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

CLASSPATH = 'target/petsc-0.9.0-jar-with-dependencies.jar'
DEFAULT_MAX_HEAP_SIZE = '-Xmx12G'


def mr_petsc(train_filename, test_filename, w=15, alphabet=4, k=200, min_len=5, rdur=1.1, soft=False, verbose=False, stride=1):
    # Load data
    X_train, y_train = load_from_tsfile_to_dataframe(train_filename)
    X_test, y_test = load_from_tsfile_to_dataframe(test_filename)

    # Embedding
    embedding_train_fname = 'temp/mr_petsc_embedding_train.txt' + str(uuid.uuid1())
    embedding_test_fname = 'temp/mr_petsc_embedding_test.txt' + str(uuid.uuid1())
    embeddings_train = []
    embeddings_test = []
    nr_series = y_train.shape[0]
    window = min([X_train['dim_0'].iloc[i].to_numpy().shape[0] for i in range(0, nr_series)])
    # window = min(window, min([X_test['dim_0'].iloc[i].to_numpy().shape[0] for i in range(0, y_test.shape[0])]))
    window //= 2
    while window > w:
        cmd = [
            # Java
            'java', DEFAULT_MAX_HEAP_SIZE, '-cp', CLASSPATH,
            'be.uantwerpen.mining.MineTopKSequentialPatterns',

            # Input
            '-input', train_filename, '-input_test', test_filename,

            # Preprocessing
            '-paa_win', str(w), '-bins', str(alphabet),

            # Constraints
            '-min_size', str(min_len), '-max_size', str(w),

            # Mining
            '-window', str(window), '-stride', str(stride), '-k', str(k),

            # Output
            '-output', embedding_train_fname, '-output_test', embedding_test_fname
        ]

        if not soft:
            cmd = cmd + ['-duration', str(rdur)]
        else:
            cmd = cmd + ['-duration', '1.0', '-soft', 'true', '-tau', '2.0']

        result = run(cmd)

        if result.returncode == 1:
            raise ChildProcessError

        embeddings_train.append(pd.read_csv(embedding_train_fname, header=None))
        embeddings_test.append(pd.read_csv(embedding_test_fname, header=None))

        clear(embedding_train_fname, embedding_test_fname)

        window = window // 2

        # Concat
        train_concat = pd.concat(embeddings_train, axis=1)
        test_concat = pd.concat(embeddings_test, axis=1)

        clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3, penalty='elasticnet', n_jobs=-1))
        clf.fit(train_concat, y_train)
        y_pred = clf.predict(test_concat)
        return y_pred


def mr_petsc_multivariate(train_filename, test_filename, w, alphabet, k, min_len, rdur, stride):
    X_train, y_train = load_from_tsfile_to_dataframe(train_filename)
    X_test, y_test = load_from_tsfile_to_dataframe(test_filename)    

    # Split each dimension into seperate file
    filenames_train = copy_ts_multivariate(train_filename, X_train, y_train)
    filenames_test = copy_ts_multivariate(test_filename, X_test, y_test)
    
    # Embedding
    embeddings_train = []
    embeddings_test = []
    nr_series = y_train.shape[0]
    for dimension in range(0, len(filenames_train)):
        window = min([X_train['dim_{}'.format(dimension)].iloc[i].to_numpy().shape[0] for i in range(0,nr_series)])
        # window //= 2
        while window > w:
            X_embtrain, X_embtest = mr_petsc_embed(filenames_train[dimension], filenames_test[dimension], window, w, alphabet, k, min_len, rdur, stride)
            if X_embtrain is None:
                continue
            embeddings_train.append(X_embtrain)
            embeddings_test.append(X_embtest)
            window = window // 2 

    # Concat
    train_concat = pd.concat(embeddings_train, axis=1)
    test_concat = pd.concat(embeddings_test, axis=1)

    clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3, penalty='elasticnet', n_jobs=-1))
    clf.fit(train_concat, y_train)
    y_pred = clf.predict(test_concat)
    return y_pred


def clear(*fnames):
    for fname in fnames:
        if os.path.exists(fname):
            os.remove(fname)


def copy_ts_multivariate(filename, data_x, data_y):
    nr_of_dimensions = 0

    # Copy header ts to memory
    lines = []
    f = open(filename, 'r')
    for line in f:
        if line.startswith('@'):
            lines.append(line)
        if line.startswith('@dimensions'):
            nr_of_dimensions = int(line[len('@dimensions '):])
    f.close()

    # Copy header for each dimension
    filenames = []
    for dimension in range(0,nr_of_dimensions):
        filename_output_dim_i = filename + '_dim_{:03d}.ts'.format(dimension)
        filenames.append(filename_output_dim_i)
        fo = open(filename_output_dim_i,'w')
        for line in lines:
            fo.write(line)
        fo.close()

    # Save data for each dimension
    nr_series = data_y.shape[0]
    for dimension in range(0, nr_of_dimensions):
        filename_output_dim_i = filename + '_dim_{:03d}.ts'.format(dimension)
        fo = open(filename_output_dim_i, 'a')
        for i in range(0, nr_series):
            arr = data_x['dim_{}'.format(dimension)].iloc[i].to_numpy().tolist()
            y = data_y[i]
            for i in range(0, len(arr)-1):
                fo.write('{:.6f}'.format(arr[i]))
                fo.write(',')
            fo.write('{:.6f}'.format(arr[len(arr)-1]))
            fo.write(':')
            fo.write(y)
            fo.write('\n')
        fo.close()
    return filenames
