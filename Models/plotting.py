from sklearn.model_selection import learning_curve
from sklearn.metrics import make_scorer, roc_curve, roc_auc_score, plot_confusion_matrix, average_precision_score, \
    precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
from Models import groupby_type, convert
from scipy.optimize import curve_fit
import numpy as np
import itertools
import math
import copy
import matplotlib.pyplot as plt
import pandas as pd


def plot_estimator_scores(dataset_name, target_name, best_estimator, estimators, trn_scores, tst_scores, params,
                          param_grid, scoring):
    """
    plot score vs parameters plot for every estimator in the estimators list
    :param param_grid: a grid of model parameters
    :param tst_scores: a dataframe of test scores
    :param trn_scores: a dataframe of train scores
    :param best_estimator: trained estimator
    :param target_name: column name of y
    :param dataset_name: corresponds to the dirname containing the dataset
    :param estimators: list of trained models
    :param params: list of in_scope parameters
    :param scoring: sklearn scoring metric
    :return: plt.Figure
    """

    print('Plotting estimator scores')

    # finds the number of estimators for each estimator class
    unique_classes = groupby_type(estimators)
    # a list of estimator's all parameters
    estimators_vals = [[estimator.get_params()[i] for i in params[estimator.__class__.__name__]] for estimator in
                       estimators]
    # convert 1d list of estimators into 2d given the number of estimators per class
    by_type_estimators = convert(estimators, unique_classes.values())
    # convert 1d list of estimator parameters into 2d given the number of estimators per class
    by_type_estimators_vals = convert(estimators_vals, unique_classes.values())
    # convert 1d list of train scores into 2d given the number of estimators per class
    by_type_trn_scores = [pd.DataFrame(i, columns=['mean_train_{}'.format(scoring.__name__)]) for i in
                          convert(list(trn_scores['mean_train_{}'.format(scoring.__name__)]), unique_classes.values())]
    # convert 1d list of test scores into 2d given the number of estimators per class
    by_type_tst_scores = [pd.DataFrame(i, columns=['mean_test_{}'.format(scoring.__name__)]) for i in
                          convert(list(tst_scores['mean_test_{}'.format(scoring.__name__)]), unique_classes.values())]
    # get the model parameters are are of interest
    param_grid_in_scope = {best_estimator.__class__.__name__: param_grid[best_estimator.__class__.__name__]}
    for i1, (k1, v1) in enumerate(param_grid_in_scope.items()):
        # convert the lists of parameters into dataframe
        df = pd.DataFrame(by_type_estimators_vals[i1], columns=list(v1['params'].keys()))

        # one subplot is enough
        if len(df.columns) == 1:
            fig, ax = plt.figure()
            # a list of individual values per unique parameter e.g.: C: [1, 5, 10], 'loss': ['linear', 'square'], ...
            independent_values = df[str(list(v1['params'].keys())[0])].tolist()

            trn_scores_at_indices = list(by_type_trn_scores[i1]['mean_train_{}'.format(scoring.__name__)])
            tst_scores_at_indices = list(by_type_tst_scores[i1]['mean_test_{}'.format(scoring.__name__)])

            # the index of highest score
            index_max_tst_score = tst_scores_at_indices.index(max(tst_scores_at_indices))

            if all(isinstance(n, (int, float)) for n in independent_values):
                # parameter values are numbers
                lin_space = np.linspace(min(independent_values), max(independent_values), len(independent_values))
            else:
                # parameter values are strings
                lin_space = independent_values

            fig.plot(lin_space, trn_scores_at_indices, label='train', color='g', marker='o')
            fig.plot(lin_space, tst_scores_at_indices, label='test', color='r', marker='o')
            # plot a marker at highest test score
            fig.plot(lin_space[index_max_tst_score], max(tst_scores_at_indices), color='r', marker='x',
                     markersize=15)

            fig.set_yticks(np.linspace(0., 1., num=6))
            fig.set_xticks(lin_space)
            ax.set_yticklabels([round(i, 1) for i in np.linspace(0., 1., num=6)])
            ax.set_xticklabels(independent_values)

            fig.set_title('{}({})\n{} {} vs {} '.format(dataset_name, target_name, v1['model'].__class__.__name__,
                                                        scoring.__name__, list(df.columns)[0]))
            # train values when test score is highest
            fig.text(independent_values[0], 0.95, 'train = {:.3f}'.format(trn_scores_at_indices[index_max_tst_score]),
                     color='g')
            # test values when test score is highest
            fig.text(independent_values[0], 0.90, 'test = {:.3f}'.format(tst_scores_at_indices[index_max_tst_score]),
                     color='r')
            fig.legend(loc='lower right')

            return fig
            # continue # -> use when there is a list of estimators

        # the upcoming Figure will have multiple subplots
        fig_size = len(list(itertools.product(*v1['params'].values()))) * 20 / 12
        # number of horizontal subplots
        width = int(len(list(itertools.product(*v1['params'].values()))) / min([len(i) for i in v1['params'].values()]))
        # number of vertical subplots
        height = len(v1['params'].values())

        fig, ax = plt.subplots(height, width, figsize=(fig_size, fig_size))
        fig.suptitle('{}({})\n{} {} VS {}'.format(dataset_name, target_name, v1['model'].__class__.__name__,
                                                  scoring.__name__, list(v1['params'].keys())))
        for i2, (k2, v2) in enumerate(v1['params'].items()):
            # deepcopy of params
            var_excluding_values_at_index = copy.deepcopy(v1['params'])
            # goal: make every possible unique combination of parameters excluding the parameter at index k2
            del var_excluding_values_at_index[k2]
            # make combinations
            combinations_excluding_index_values = list(itertools.product(*var_excluding_values_at_index.values()))
            # list of param names excluding the parameter at index k2
            param_names_excluding_index = list(var_excluding_values_at_index.keys())

            count = 1
            for i3, v3 in enumerate(combinations_excluding_index_values):
                # a list of individual values per unique parameter e.g.: C: [1, 5, 10], 'loss': ['linear', 'square'],...
                independent_values = df.loc[df[param_names_excluding_index].apply(tuple, axis=1).isin([v3])]

                trn_scores_at_indices = list(by_type_trn_scores[i1].iloc[independent_values.index.values][
                                                 'mean_train_{}'.format(scoring.__name__)])
                tst_scores_at_indices = list(by_type_tst_scores[i1].iloc[independent_values.index.values][
                                                 'mean_test_{}'.format(scoring.__name__)])
                # the index of highest score
                index_max_tst_score = tst_scores_at_indices.index(max(tst_scores_at_indices))

                if all(isinstance(n, (int, float)) for n in v2):
                    # parameter values are numbers
                    lin_space = np.linspace(min(v2), max(v2), len(v2))
                else:
                    # parameter values are strings
                    lin_space = v2

                ax[i2, i3].plot(lin_space, trn_scores_at_indices, label='train', color='g', marker='o')
                ax[i2, i3].plot(lin_space, tst_scores_at_indices, label='test', color='r', marker='o')
                # plot a marker at highest test score
                ax[i2, i3].plot(lin_space[index_max_tst_score], max(tst_scores_at_indices), color='r', marker='x',
                                markersize=15)

                ax[i2, i3].set_yticks(np.linspace(0., 1., num=6))
                ax[i2, i3].set_xticks(lin_space)
                ax[i2, i3].set_yticklabels([round(i, 1) for i in np.linspace(0., 1., num=6)])
                ax[i2, i3].set_xticklabels(v2)

                set_title = '{} vs {} @\n{} '.format(scoring.__name__, k2,
                                                     ',\n'.join([str(i) + ' = ' + str(j) for i, j in
                                                                 list(zip(param_names_excluding_index, v3))]))
                ax[i2, i3].set_title(set_title)
                # train values when test score is highest
                ax[i2, i3].text(v2[0], 0.95, 'train = {:.3f}'.format(trn_scores_at_indices[index_max_tst_score]),
                                color='g')
                # test values when test score is highest
                ax[i2, i3].text(v2[0], 0.90, 'test = {:.3f}'.format(tst_scores_at_indices[index_max_tst_score]),
                                color='r')
                ax[i2, i3].legend(loc='lower right')
                count += 1

        return fig


def plot_learning_curve(dataset_name, target_name, estimators, X_train, y_train, params, scoring, cv=5,
                        shuffle=True, train_sizes=np.linspace(.1, 1., 100)):
    """
    plots loss vs train size plot for every estimator in the estimators list
    :param target_name: column name of y
    :param dataset_name: corresponds to the dirname containing the dataset
    :param estimators: list of trained models
    :param X_train: X_train
    :param y_train: train target column
    :param params: list of in_scope parameters
    :param scoring: sklearn scoring metric
    :param cv: cross validation. default=5
    :param shuffle: shuffle the data before splitting into batches
    :param train_sizes: a list/array of train sizes
    :return: plt.Figure
    """
    print('plotting learning curves')
    # finds the number of estimators for each estimator class
    unique_classes = groupby_type(estimators)
    # convert 1d list of estimators into 2d given the number of estimators per class
    by_type_estimators = convert(estimators, unique_classes.values())

    # curve_fit function (polynomial of degree 2)
    def func(x, a, b, c):
        return a * x ** 2 + b * x + c

    for class_idx, class_ in enumerate(by_type_estimators):
        # number of plots within the figure
        side_len = math.ceil(math.sqrt(len(class_)))
        # size of figure
        fig_size = len(class_) * 20 / 8

        fig, axes = plt.subplots(side_len, side_len, figsize=(fig_size, fig_size))
        fig.suptitle('{}({})\nLearning curves for {}'.format(dataset_name, target_name, class_[0].__class__.__name__, ))

        estimator_idx = 0
        for idx1, ax_row in enumerate(axes):
            for idx2, ax in enumerate(ax_row):
                if estimator_idx == len(class_):
                    break
                print('plotting subplot', estimator_idx + 1)
                estimator = class_[estimator_idx]

                # calculate learning curve variables: train_sizes, train_scores, validation_scores
                train_sizes1, train_scores1, validation_scores1 = \
                    learning_curve(estimator, X_train, y_train, cv=cv, train_sizes=train_sizes,
                                   scoring=make_scorer(scoring[0]), shuffle=shuffle)

                # apply abs() to negative scores (e.g. r2_score)
                train_scores_mean1 = [abs(i) for i in np.mean(train_scores1, axis=1)]
                validation_scores_mean1 = [abs(i) for i in np.mean(validation_scores1, axis=1)]

                # fill NAs with 0
                train_scores_mean1 = [0 if math.isnan(x) else x for x in train_scores_mean1]
                validation_scores_mean1 = [0 if math.isnan(x) else x for x in validation_scores_mean1]

                # find a curve_fit for the points scatter so it's easy to analyze
                a1, _ = curve_fit(func, train_sizes1, train_scores_mean1)
                a2, _ = curve_fit(func, train_sizes1, validation_scores_mean1)

                ax.plot(train_sizes1, train_scores_mean1, "o", color='royalblue',
                        label='Training {}'.format(scoring[0].__name__))
                ax.plot(train_sizes1, func(train_sizes1, *a1), '-', color='royalblue')
                ax.plot(train_sizes1, validation_scores_mean1, "o", fillstyle='none', color='royalblue',
                        label='Validation {}'.format(scoring[0].__name__))
                ax.plot(train_sizes1, func(train_sizes1, *a2), linestyle='dashed', color='royalblue')

                # get parameters in a dict
                params_val = {
                    params[estimator.__class__.__name__][idx]: estimator.get_params()[model_name] for
                    idx, model_name in enumerate(params[estimator.__class__.__name__])
                }
                # join the parameters and convert to a string
                params_val_formatted = ', '.join('{}: {}'.format(key, value) for key, value in params_val.items())

                set_title = '{}\n({})'.format(estimator.__class__.__name__, params_val_formatted)
                ax.set_title(set_title)
                ax.set_xlabel('Training size')
                ax.set_ylabel(scoring[0].__name__)
                ax.legend(loc="upper right")

                # can also plot in 2nd y axis
                if len(scoring) == 2:
                    ax2 = ax.twinx()

                    # calculate learning curve variables: train_sizes, train_scores, validation_scores
                    train_sizes2, train_scores2, validation_scores2 = \
                        learning_curve(estimator, X_train, y_train, cv=cv, train_sizes=train_sizes,
                                       scoring=make_scorer(scoring[1]), shuffle=shuffle)

                    # apply abs() to negative scores (e.g. r2_score)
                    train_scores_mean2 = [abs(i) for i in np.mean(train_scores2, axis=1)]
                    validation_scores_mean2 = [abs(i) for i in np.mean(validation_scores2, axis=1)]

                    # fill NAs with 0
                    train_scores_mean2 = [0 if math.isnan(x) else x for x in train_scores_mean2]
                    validation_scores_mean2 = [0 if math.isnan(x) else x for x in validation_scores_mean2]

                    # find a curve_fit for the points scatter so it's easy to analyze
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

        # discard empty space and make the individual plots larger
        fig.tight_layout()

        return fig


def plot_ROC_curve(model, X_test, y_test):
    """
        plot a ROC curve
        :param model: trained model
        :param X_test: X_test
        :param y_test: target values
        :return: plt.Figure
    """

    pred = model.predict(X_test)
    # find false-positive rate, true-positive-rate, thresholds values to plot
    fpr, tpr, thresholds = roc_curve(y_test, pred, pos_label=1)

    # calculates the Receiver operating characteristic area under curve
    # should equal the value in scores table corresponding to the model
    auc = roc_auc_score(y_test, pred)

    fig, ax = plt.subplots()
    # plot false-positive-rate vs true-positive-rate
    ax.plot(fpr, tpr, color='orange', label='ROC: {}'.format(round(auc, 2)))
    # plot y=x
    ax.plot([0, 1], [0, 1], color='navy', linestyle='--')

    fig.suptitle('ROC AUC')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    fig.legend(loc="lower right")

    return fig


def plot_PR_curve(model, X_test, y_test):
    """
        plot a PR curve
        :param model: trained model
        :param X_test: X_test
        :param y_test: target values
        :return: plt.Figure
    """

    pred = model.predict(X_test)

    # find recall, precision, thresholds values to plot
    r, p, thresholds = precision_recall_curve(y_test, pred, pos_label=1)

    # calculates the average_precision_score area under curve
    # should equal the value in scores table corresponding to the model
    apr = average_precision_score(y_test, pred)

    fig, ax = plt.subplots()

    # plot recall vs precision
    ax.plot(r, p, color='orange', label='APR: {}'.format(round(apr, 2)))
    # extend the recall vs precision line to y axis
    ax.plot([0, min(r)], [1, 1], color='orange')
    # plot a y=x
    ax.plot([0, 1], [0, 1], color='navy', linestyle='--')

    fig.suptitle('PR AUC')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    fig.legend(loc="lower right")

    return fig


def plot_conf_matrix(model, X_test, y_test, labels):
    """
    plot a confusion matrix figure
    :param model: trained model
    :param X_test: X_test
    :param y_test: target values
    :param labels: y labels
    :return: plt.Figure
    """

    cm = ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        display_labels=labels,
        cmap=plt.cm.Blues,
    )
    cm.ax_.set_title(model.__class__.__name__)

    return cm.figure_
