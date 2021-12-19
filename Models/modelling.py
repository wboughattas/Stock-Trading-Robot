from sklearn.metrics import make_scorer, f1_score
from itertools import islice
import pandas as pd
import itertools
import sys


def sklearn_grid_search(X, y, search_type, param_grid, scoring, **kwargs):
    """
    Search over specified parameter values for an estimator
    :param scoring:
    :param X: X_train
    :param y: y_train
    :param search_type: GridSearchCV or RandomizedSearchCV
    :param param_grid: Dictionary with parameters names (str) as keys and lists of parameter settings to try as values
    :param kwargs: attributes corresponding to search_type
    :return: a dataframe with best_score and best_params per model
    """

    dfs = []

    print('Running sklearn_grid_search')
    # f1_score is excluded because we cannot set average = 'weighted'
    estimator_scoring = {i.__name__: make_scorer(i) for i in scoring if i is not f1_score}
    for model_name, val in param_grid.items():
        clf = search_type(val['model'], val['params'], refit=False, return_train_score=True, scoring=estimator_scoring,
                          **kwargs)
        clf.fit(X, y)

        results = pd.DataFrame(clf.cv_results_)
        scores_columns = [i for i in list(clf.cv_results_.keys()) if 'mean' in i and 'score' in i and 'time' not in i]
        results = results[['params', *scores_columns]]
        results = results.astype({'params': str})
        results['params'] = results['params'].replace({'{': '', '}': '', "'": ''}, regex=True)
        results['model'] = [val['model'].__class__.__name__] * len(results)

        dfs.append(results)

    df = pd.concat(dfs).fillna(0)
    cols = list(df.columns)
    cols = [cols[-1]] + cols[:-1]
    df = df[cols]
    return df


def score_estimators(X, y, estimators, score_metric, custom_params, type_reference):
    """
    outputs best_score and params per model
    :param type_reference:
    :param score_metric: https://scikit-learn.org/stable/modules/model_evaluation.html
    :param custom_params: a dictionary of estimator class names as keys and list of parameter names as values
    :param X: X_train/test
    :param y: y_train/test
    :param estimators: list of sklearn classifiers
    :return: a dataframe with scores and params per model
    """

    results = []
    for index, estimator in enumerate(estimators):
        try:
            score = abs(score_metric(y, estimator.predict(X)))
        except ValueError as exp:
            print(exp)
            print('A weighted setting has been used')
            score = abs(score_metric(y, estimator.predict(X), average='weighted'))
        params = {
            custom_params[estimator.__class__.__name__][idx]: estimator.get_params()[model_name] for
            idx, model_name in enumerate(custom_params[estimator.__class__.__name__])
        }
        params_formatted = ', '.join('{}: {}'.format(key, value) for key, value in params.items())
        estimator_info = {'model': estimator.__class__.__name__,
                          '_'.join(['mean', type_reference, score_metric.__name__]): score,
                          'params': estimator.get_params() if custom_params is None else params_formatted}
        results.append(pd.DataFrame(estimator_info, index=[0]))
    df = pd.concat(results).fillna(0)
    # df = df.drop(df[df.best_score < 1.].index)
    # return df, df.loc[df.groupby(['model']).idxmax()['best_score']].reset_index().drop('index', axis=1)
    return df


def convert(listA, len_2d):
    """

    :param listA:
    :param len_2d:
    :return:
    """
    res = iter(listA)
    return [list(islice(res, i)) for i in len_2d]


def groupby_type(estimators):
    """

    :param estimators:
    :return:
    """
    unique_classes = {}
    for estimator in estimators:
        clf_name = estimator.__class__.__name__
        if clf_name not in list(unique_classes.keys()):
            unique_classes[clf_name] = 0
        unique_classes[clf_name] += 1
    return unique_classes


def cleanup_scores(*args, comments=False):
    model_col = pd.DataFrame(args[0]['model']).iloc[:, 0]
    params_col = pd.DataFrame(args[0]['params']).iloc[:, 0]
    df = pd.concat(([i.drop(['model', 'params'], axis=1) for i in args]), axis=1)
    df = df.round(2)
    df = pd.concat([model_col, params_col, df], axis=1).fillna(0)
    df.reset_index(drop=True, inplace=True)

    if comments is True:
        mean_test_columns = [i for i in list(df.columns) if 'mean_test' in i]
        for i in mean_test_columns:
            large_score_est = df.loc[df.groupby(['model'])[i].idxmax()]
            large_score_indices = large_score_est.index
            new_column = i.replace('mean', 'ranking')
            df[new_column] = ['-']*len(df)
            df[new_column].loc[large_score_indices] = 'highest of its class'

            small_score_est = df.loc[df.groupby(['model'])[i].idxmin()]
            small_score_indices = small_score_est.index
            df[new_column].loc[small_score_indices] = 'lowest of its class'
    return df


def custom_grid_search(X, y, search_grid):
    """
    Exhaustive search over specified parameter values for an estimator. Tries every combination of parameters
    :param X: X_train
    :param y: y_train
    :param search_grid: Dictionary with parameters names (str) as keys and lists of parameter settings to try as values
    :return: a dataframe with best_score, and best_params per model and a list of parameters
    """

    clf = []
    lst_lst_param_keys = {}
    for estimator_name, model_params in search_grid.items():
        lst_param_values, lst_param_keys = [], []
        for param_keys, param_val in model_params['params'].items():
            try:
                if all(isinstance(n, (int, float)) for n in param_val):
                    if not all(param_val[i] <= param_val[i + 1] for i in range(len(param_val) - 1)):
                        raise ValueError('Numerical values must be increasing')
            except ValueError as exp:
                print("\nError", exp)
                sys.exit()
            lst_param_values.append(param_val)
            lst_param_keys.append(param_keys)
        lst_lst_param_keys[model_params['model'].__class__.__name__] = lst_param_keys
        unique_combinations = list(itertools.product(*lst_param_values))
        for val in unique_combinations:
            params = {lst_param_keys[i]: val[i] for i in range(len(lst_param_keys))}
            class_ = model_params['model'].__class__
            initial_attributes = model_params['model'].get_params()
            model = class_(**{**initial_attributes, **params})
            print('Training {}'.format(model))
            try:
                model.fit(X, y)
                clf.append(model)
            except ValueError as exp2:
                print("An error was encountered while running this model: ", exp2)
                print("Must exclude this model from the param_grid: ", model)
                sys.exit()
    return clf, lst_lst_param_keys
