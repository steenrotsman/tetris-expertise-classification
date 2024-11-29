This repository is set up to include reproducible code and data for the paper Expertise Prediction of Tetris Players Using Eye Tracking Information. The repository contains 5 Python files: `split.py`, `train.py`, `test.py`, `attribute.py` and `plot.py`.

`split.py` provides the code used to generate the files in the `data/ts` folder. It cannot be run for now, as we omitted the original participant data.

`train.py` runs the classification models using 5-fold cross validation. Most methods were imported from the `aeon` package, but the MR-PETSC was adapted from the repository https://bitbucket.org/len_feremans/petsc/src/master/. The results for each method are stored in a csv file in the `results` folder.

`test.py` loads the results from the `results` folder for each method and provides several metrics, it also contains the code for reproducing the confusion matrices in the paper.

`attribute.py` creates the figure with attribution.

Lastly, `plot.py` contains the code for the other figure in the paper.
