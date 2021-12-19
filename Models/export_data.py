from matplotlib.lines import Line2D
from sklearn.model_selection import learning_curve
from sklearn.metrics import make_scorer
from Models import groupby_type, convert
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from __init__ import ROOT_DIR
from pathlib import Path
import pandas as pd
import numpy as np
import itertools
import math
import copy
import os


def plot_estimator_scores(dataset_name, target_name, estimators, trn_scores, tst_scores, params, param_grid, scoring):
    abspath = os.path.join(ROOT_DIR, 'Results', dataset_name,
                           'Train_and_test_' + scoring.__name__ + '_VS_model_parameters')
    Path(abspath).mkdir(parents=True, exist_ok=True)
    print('Plotting estimator scores')
    unique_classes = groupby_type(estimators)
    estimators_vals = [[estimator.get_params()[i] for i in params[estimator.__class__.__name__]] for estimator in
                       estimators]
    by_type_estimators = convert(estimators, unique_classes.values())
    by_type_estimators_vals = convert(estimators_vals, unique_classes.values())
    by_type_trn_scores = [pd.DataFrame(i, columns=['mean_train_{}'.format(scoring.__name__)]) for i in
                          convert(list(trn_scores['mean_train_{}'.format(scoring.__name__)]), unique_classes.values())]
    by_type_tst_scores = [pd.DataFrame(i, columns=['mean_test_{}'.format(scoring.__name__)]) for i in
                          convert(list(tst_scores['mean_test_{}'.format(scoring.__name__)]), unique_classes.values())]

    for i1, (k1, v1) in enumerate(param_grid.items()):
        df = pd.DataFrame(by_type_estimators_vals[i1], columns=list(v1['params'].keys()))
        if len(df.columns) == 1:
            plt.figure()
            independent_values = df[str(list(v1['params'].keys())[0])].tolist()
            trn_scores_at_indices = list(by_type_trn_scores[i1]['mean_train_{}'.format(scoring.__name__)])
            tst_scores_at_indices = list(by_type_tst_scores[i1]['mean_test_{}'.format(scoring.__name__)])

            index_max_tst_score = tst_scores_at_indices.index(max(tst_scores_at_indices))

            if all(isinstance(n, (int, float)) for n in independent_values):
                lin_space = np.linspace(min(independent_values), max(independent_values), len(independent_values))
            else:
                lin_space = independent_values

            plt.plot(lin_space, trn_scores_at_indices, label='train', color='g', marker='o')
            plt.plot(lin_space, tst_scores_at_indices, label='test', color='r', marker='o')
            plt.plot(lin_space[index_max_tst_score], max(tst_scores_at_indices), color='r', marker='x',
                     markersize=15)
            plt.yticks(np.linspace(0., 1., num=6), [round(i, 1) for i in np.linspace(0., 1., num=6)])
            plt.xticks(lin_space, independent_values)
            plt.title('{}({})\n{} {} score vs {} '.format(dataset_name, target_name, v1['model'].__class__.__name__,
                                                          scoring.__name__, list(df.columns)[0]))
            plt.text(independent_values[0], 0.95, 'train = {:.3f}'.format(trn_scores_at_indices[index_max_tst_score]),
                     color='g')
            plt.text(independent_values[0], 0.90, 'test = {:.3f}'.format(tst_scores_at_indices[index_max_tst_score]),
                     color='r')
            plt.legend(loc='lower right')
            # plt.show()

            plt_abspath = Path(os.path.join(abspath, scoring.__name__ + '_VS_' + k1 + '_parameters')).with_suffix(
                '.png')
            plt.savefig(plt_abspath)

            continue
        fig_size = len(list(itertools.product(*v1['params'].values()))) * 20 / 12
        width = int(len(list(itertools.product(*v1['params'].values()))) / min([len(i) for i in v1['params'].values()]))
        height = len(v1['params'].values())
        fig, ax = plt.subplots(height, width, figsize=(fig_size, fig_size))
        plt.suptitle('{}({})\n{} {} score VS {}'.format(dataset_name, target_name, v1['model'].__class__.__name__,
                                                        scoring.__name__, list(v1['params'].keys())))
        for i2, (k2, v2) in enumerate(v1['params'].items()):
            var_excluding_values_at_index = copy.deepcopy(v1['params'])
            del var_excluding_values_at_index[k2]
            combinations_excluding_index_values = list(itertools.product(*var_excluding_values_at_index.values()))
            param_names_excluding_index = list(var_excluding_values_at_index.keys())
            max_ = int(len(list(itertools.product(*v1['params'].values()))) /
                       min([len(i) for i in v1['params'].values()]))
            count = 1

            for i3, v3 in enumerate(combinations_excluding_index_values):
                independent_values = df.loc[df[param_names_excluding_index].apply(tuple, axis=1).isin([v3])]
                trn_scores_at_indices = list(by_type_trn_scores[i1].iloc[independent_values.index.values][
                                                 'mean_train_{}'.format(scoring.__name__)])
                tst_scores_at_indices = list(by_type_tst_scores[i1].iloc[independent_values.index.values][
                                                 'mean_test_{}'.format(scoring.__name__)])
                index_max_tst_score = tst_scores_at_indices.index(max(tst_scores_at_indices))

                if all(isinstance(n, (int, float)) for n in v2):
                    lin_space = np.linspace(min(v2), max(v2), len(v2))
                else:
                    lin_space = v2

                ax[i2, i3].plot(lin_space, trn_scores_at_indices, label='train', color='g', marker='o')
                ax[i2, i3].plot(lin_space, tst_scores_at_indices, label='test', color='r', marker='o')
                ax[i2, i3].plot(lin_space[index_max_tst_score], max(tst_scores_at_indices), color='r', marker='x',
                                markersize=15)
                ax[i2, i3].set_yticks(np.linspace(0., 1., num=6))
                ax[i2, i3].set_yticklabels([round(i, 1) for i in np.linspace(0., 1., num=6)])

                ax[i2, i3].set_xticks(lin_space)
                ax[i2, i3].set_xticklabels(v2)

                title = 'score vs {} @\n{} '.format(k2, ',\n'.join([str(i) + ' = ' + str(j) for i, j in
                                                                    list(zip(param_names_excluding_index, v3))]))
                ax[i2, i3].set_title(title)
                ax[i2, i3].text(v2[0], 0.95, 'train = {:.3f}'.format(trn_scores_at_indices[index_max_tst_score]),
                                color='g')
                ax[i2, i3].text(v2[0], 0.90, 'test = {:.3f}'.format(tst_scores_at_indices[index_max_tst_score]),
                                color='r')
                ax[i2, i3].legend(loc='lower right')
                count += 1

        # plt.show()
        plt_abspath = Path(os.path.join(abspath, scoring.__name__ + '_VS_' + k1 + '_parameters')).with_suffix('.png')
        plt.savefig(plt_abspath)


def plot_learning_curve(dataset_name, target_name, estimators, X_train, y_train, params, scoring, cv=5,
                        shuffle=True, train_sizes=np.linspace(.1, 1., 10)):
    """

    :param target_name:
    :param dataset_name:
    :param estimators:
    :param X_train:
    :param y_train:
    :param params:
    :param scoring:
    :param cv:
    :param shuffle:
    :param train_sizes:
    :return:
    """
    abspath = os.path.join(ROOT_DIR, 'Results', dataset_name,
                           '_vs_'.join([i.__name__ for i in scoring]) + '_VS_Training_size')
    Path(abspath).mkdir(parents=True, exist_ok=True)

    unique_classes = groupby_type(estimators)
    by_type_estimators = convert(estimators, unique_classes.values())
    colors = [['cyan', 'royalblue', 'blue', 'green', 'purple'], ['red', 'orange', 'yellow', 'pink', 'purple']]

    def func(x, a, b, c):
        return a * x ** 2 + b * x + c

    for class_idx, class_ in enumerate(by_type_estimators):
        side_len = math.ceil(math.sqrt(len(class_)))
        fig_size = len(class_) * 20 / 8
        fig, axes = plt.subplots(side_len, side_len, figsize=(fig_size, fig_size))
        fig.suptitle('{}({})\nLearning curves for {}'.format(dataset_name, target_name, class_[0].__class__.__name__, ))
        estimator_idx = 0
        for idx1, ax_row in enumerate(axes):
            for idx2, ax in enumerate(ax_row):
                if estimator_idx == len(class_):
                    break
                print('plotting estimator', estimator_idx + 1)
                estimator = class_[estimator_idx]
                train_sizes1, train_scores1, validation_scores1 = \
                    learning_curve(estimator, X_train, y_train, cv=cv, train_sizes=train_sizes,
                                   scoring=make_scorer(scoring[0]), shuffle=shuffle)
                train_scores_mean1 = [abs(i) for i in np.mean(train_scores1, axis=1)]
                validation_scores_mean1 = [abs(i) for i in np.mean(validation_scores1, axis=1)]

                train_scores_mean1 = [0 if math.isnan(x) else x for x in train_scores_mean1]
                validation_scores_mean1 = [0 if math.isnan(x) else x for x in validation_scores_mean1]

                a1, _ = curve_fit(func, train_sizes1, train_scores_mean1)
                a2, _ = curve_fit(func, train_sizes1, validation_scores_mean1)

                ax.plot(train_sizes1, train_scores_mean1, "o", color='royalblue',
                        label='Training {}'.format(scoring[0].__name__))
                ax.plot(train_sizes1, func(train_sizes1, *a1), '-', color='royalblue')
                ax.plot(train_sizes1, validation_scores_mean1, "o", fillstyle='none', color='royalblue',
                        label='Validation {}'.format(scoring[0].__name__))
                ax.plot(train_sizes1, func(train_sizes1, *a2), linestyle='dashed', color='royalblue')

                params_val = {
                    params[estimator.__class__.__name__][idx]: estimator.get_params()[model_name] for
                    idx, model_name in enumerate(params[estimator.__class__.__name__])
                }
                params_val_formatted = ', '.join('{}: {}'.format(key, value) for key, value in params_val.items())
                title = '{}\n({})'.format(estimator.__class__.__name__, params_val_formatted)
                ax.set_title(title)
                ax.set_xlabel('Training size')
                ax.set_ylabel(scoring[0].__name__)
                ax.legend(loc="upper right")

                if len(scoring) == 2:
                    ax2 = ax.twinx()
                    train_sizes2, train_scores2, validation_scores2 = \
                        learning_curve(estimator, X_train, y_train, cv=cv, train_sizes=train_sizes,
                                       scoring=make_scorer(scoring[1]), shuffle=shuffle)
                    train_scores_mean2 = [abs(i) for i in np.mean(train_scores2, axis=1)]
                    validation_scores_mean2 = [abs(i) for i in np.mean(validation_scores2, axis=1)]

                    train_scores_mean2 = [0 if math.isnan(x) else x for x in train_scores_mean2]
                    validation_scores_mean2 = [0 if math.isnan(x) else x for x in validation_scores_mean2]

                    a3, _ = curve_fit(func, train_sizes2, train_scores_mean2)
                    a4, _ = curve_fit(func, train_sizes2, validation_scores_mean2)

                    ax2.plot(train_sizes2, train_scores_mean2, "o", color='red',
                             label='Training {}'.format(scoring[1].__name__))
                    ax2.plot(train_sizes2, func(train_sizes2, *a3), '-', color='red')
                    ax2.plot(train_sizes2, validation_scores_mean2, "o", fillstyle='none', color='red',
                             label='Validation {}'.format(scoring[1].__name__))
                    ax2.plot(train_sizes2, func(train_sizes2, *a4), linestyle='dashed', color='red')

                    ax2.set_ylabel(scoring[1].__name__)
                    ax2.legend(loc="lower right")

                estimator_idx += 1
        fig.tight_layout()
        fig_abspath = Path(
            os.path.join(abspath, class_[0].__class__.__name__ + '_'.join([i.__name__ for i in scoring]))).with_suffix(
            '.png')
        fig.savefig(fig_abspath)


def export_df(dataset_name, scores):
    abspath = os.path.join(ROOT_DIR, 'Results', dataset_name)
    Path(abspath).mkdir(parents=True, exist_ok=True)
    for idx, (filename, score) in enumerate(scores.items()):
        df_abspath = Path(os.path.join(abspath, filename)).with_suffix('.csv')
        score.to_csv(df_abspath, index=False)
