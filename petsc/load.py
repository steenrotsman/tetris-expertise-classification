import os


def copy_ts(filename, filename_output, data_x, data_y):
    # Copy header ts file
    f = open(filename, 'r')
    fo = open(filename_output, 'w')
    for line in f:
        if line.startswith('@'):
            fo.write(line)
    f.close()

    # Save data
    nr_series = data_y.shape[0]
    for i in range(0, nr_series):
        arr = data_x['dim_0'].iloc[i].to_numpy().tolist()
        y = data_y[i]
        for i in range(0, len(arr) - 1):
            fo.write('{:.6f}'.format(arr[i]))
            fo.write(',')
        fo.write('{:.6f}'.format(arr[len(arr) - 1]))
        fo.write(':')
        fo.write(y)
        fo.write('\n')
    fo.close()


# i.e. to split training in validation train/test
def copy_ts_mv_single_file(filename, filename_output, data_x, data_y):
    nr_of_dimensions = 0

    # Copy header ts file
    f = open(filename, 'r')
    fo = open(filename_output, 'w')
    for line in f:
        if line.startswith('@'):
            fo.write(line)
        if line.startswith('@dimensions'):
            nr_of_dimensions = int(line[len('@dimensions ') :])
    f.close()

    # Save data
    nr_series = data_y.shape[0]
    for i in range(0, nr_series):
        for d in range(0, nr_of_dimensions):
            arr = data_x['dim_{}'.format(d)].iloc[i].to_numpy().tolist()
            y = data_y[i]
            for i in range(0, len(arr) - 1):
                fo.write('{:.6f}'.format(arr[i]))
                fo.write(',')
            fo.write('{:.6f}'.format(arr[len(arr) - 1]))
            fo.write(':')
        fo.write(y)
        fo.write('\n')
    fo.close()


def copy_ts_multivariate(filename, data_x, data_y):
    nr_of_dimensions = 0

    # Copy header ts to memory
    lines = []
    f = open(filename, 'r')
    for line in f:
        if line.startswith('@'):
            lines.append(line)
        if line.startswith('@dimensions'):
            nr_of_dimensions = int(line[len('@dimensions ') :])
    f.close()

    # Copy header for each dimension
    filenames = []
    for dimension in range(0, nr_of_dimensions):
        filename_output_dim_i = filename + '_dim_{:03d}.ts'.format(dimension)
        filenames.append(filename_output_dim_i)
        fo = open(filename_output_dim_i, 'w')
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
            for i in range(0, len(arr) - 1):
                fo.write('{:.6f}'.format(arr[i]))
                fo.write(',')
            fo.write('{:.6f}'.format(arr[len(arr) - 1]))
            fo.write(':')
            fo.write(y)
            fo.write('\n')
        fo.close()
    return filenames


def clear(*fnames):
    for fname in fnames:
        if os.path.exists(fname):
            os.remove(fname)
