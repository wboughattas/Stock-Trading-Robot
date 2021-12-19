import matplotlib.pyplot as plt
from __init__ import ROOT_DIR
from pathlib import Path
import pandas as pd
import os


def export(dataset_dirname, scores):
    """
    export pd.Dataframe or plt.Figure to csv or png files respectively
    :param dataset_dirname: the directory name containing the dataset. e.g.: dirname of adult.data is 'adult'
    :param scores: a dictionary with keys corresponding to the name of the exported file and values corresponding to
    the content of the file
    :return: creates directory (if not exists) and exports dictionary values, being dataframes or plt.Figures, to csv or
    png files (with the option to overwrite) into Results/{dataset_name}
    """
    for idx, (filename, score) in enumerate(scores.items()):
        abspath = os.path.join(ROOT_DIR, 'Results', dataset_dirname)
        Path(abspath).mkdir(parents=True, exist_ok=True)
        if isinstance(score, pd.DataFrame):
            print('exporting', filename)
            df_abspath = Path(os.path.join(abspath, filename)).with_suffix('.csv')
            score.to_csv(df_abspath, index=False)
        elif isinstance(score, plt.Figure):
            print('exporting', filename)
            fig_abspath = Path(os.path.join(abspath, filename)).with_suffix('.png')
            score.savefig(fig_abspath)
