"""Writer module.

PETSC expects input files in sktime .ts format, this module provides
functionality for converting a list of lists to a .ts file.
"""


def from_lists_to_tsfile(fn: str, data: list, labels: list, class_label: str,
                         description: str, problem_name: str, time_stamps: bool,
                         missing: bool, univariate: bool, equal_length: bool,
                         series_length: int = 0):
    with open(fn, 'w') as fp:
        for line in description.split(sep='\n'):
            fp.write(f'#{line}\n')
        fp.write(f'@problemName {problem_name}\n')
        fp.write(f'@timeStamps {time_stamps}\n')
        fp.write(f'@missing {missing}\n')
        fp.write(f'@univariate {univariate}\n')
        fp.write(f'@equalLength {equal_length}\n')
        if equal_length and series_length:
            fp.write(f'@seriesLength {series_length}\n')
        fp.write(f'@classLabel {class_label}\n')
        fp.write(f'@data\n')

        # TODO add functionality for multivariate data
        if labels:
            for row, label in zip(data, labels):
                fp.write(f'{",".join([str(x) for x in row])}:{label}\n')
        else:
            for label, rows in enumerate(data):
                for row in rows:
                    fp.write(f'{",".join([str(x) for x in row])}:{label}\n')
